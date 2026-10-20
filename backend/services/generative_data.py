"""Bounded, geometry-aware voxel patches for generative CFD experiments."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkMultiBlockDataSet
from vtkmodules.vtkIOParallel import vtkPOpenFOAMReader
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.util.numpy_support import vtk_to_numpy

from backend.services.cfd_analysis import CfdDataError, get_case_field_options
from backend.services.dataset_inspector import load_manifest


GRID_SIZE = 32
PATCH_SIZE = 8
PATCH_STRIDE = 4
MINIMUM_PATCH_OCCUPANCY = 0.25
SUPPORTED_REGION = "coolant"
SUPPORTED_FIELD = "T"
SUPPORTED_UNITS = "K"


class GenerativeDataError(Exception):
    """A verified generative training patch could not be prepared."""


def _find_region_mesh(data: Any, region: str, path: tuple[str, ...] = ()) -> Any | None:
    if isinstance(data, vtkMultiBlockDataSet):
        for index in range(data.GetNumberOfBlocks()):
            metadata = data.GetMetaData(index)
            name = (
                metadata.Get(vtkCompositeDataSet.NAME())
                if metadata and metadata.Has(vtkCompositeDataSet.NAME())
                else str(index)
            )
            found = _find_region_mesh(data.GetBlock(index), region, path + (name,))
            if found is not None:
                return found
    elif len(path) >= 2 and path[-2:] == (region, "internalMesh"):
        return data
    return None


def split_patch_origins(
    origins: NDArray[np.int64], *, grid_size: int = GRID_SIZE, patch_size: int = PATCH_SIZE
) -> NDArray[np.str_]:
    """Split by a spatial slab; leave patches between the regions out as a buffer."""
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise GenerativeDataError("Patch origins must have x, y, and z indexes.")
    if patch_size <= 0 or grid_size < patch_size * 3:
        raise GenerativeDataError("Grid and patch sizes do not allow a spatial holdout.")
    labels = np.full(origins.shape[0], "buffer", dtype="<U10")
    labels[origins[:, 0] <= patch_size // 2] = "train"
    labels[origins[:, 0] >= grid_size - patch_size] = "validation"
    if not np.any(labels == "train") or not np.any(labels == "validation"):
        raise GenerativeDataError("The selected case has no train or validation patches.")
    return labels


def extract_spatial_patches(
    grid_values: NDArray[np.float64],
    grid_mask: NDArray[np.bool_],
    *,
    case_id: str,
    reynolds_number: int,
    mesh_resolution: str,
    solution_time: str,
    bounds: list[list[float]],
    grid_size: int = GRID_SIZE,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
) -> dict[str, Any]:
    """Create overlapping physical subvolumes without flattening away their layout."""
    if (
        grid_values.shape != (grid_size, grid_size, grid_size)
        or grid_mask.shape != grid_values.shape
        or not np.isfinite(grid_values[grid_mask]).all()
    ):
        raise GenerativeDataError("The resampled scalar field or validity mask is malformed.")
    if patch_size <= 0 or patch_size > grid_size or stride <= 0:
        raise GenerativeDataError("Patch size and stride must fit within the spatial grid.")

    occupancy_limit = int(np.ceil(patch_size**3 * MINIMUM_PATCH_OCCUPANCY))
    patches: list[NDArray[np.float32]] = []
    masks: list[NDArray[np.bool_]] = []
    origins: list[tuple[int, int, int]] = []
    for x in range(0, grid_size - patch_size + 1, stride):
        for y in range(0, grid_size - patch_size + 1, stride):
            for z in range(0, grid_size - patch_size + 1, stride):
                patch_mask = grid_mask[
                    x : x + patch_size, y : y + patch_size, z : z + patch_size
                ]
                if int(patch_mask.sum()) < occupancy_limit:
                    continue
                patch_values = grid_values[
                    x : x + patch_size, y : y + patch_size, z : z + patch_size
                ]
                patches.append(np.where(patch_mask, patch_values, 0).astype(np.float32))
                masks.append(patch_mask.copy())
                origins.append((x, y, z))

    if not patches:
        raise GenerativeDataError(
            "No spatial patch contains enough verified coolant cells at this grid resolution."
        )
    origin_array = np.asarray(origins, dtype=np.int64)
    split_labels = split_patch_origins(
        origin_array, grid_size=grid_size, patch_size=patch_size
    )
    spacing = [
        (float(bounds[axis][1]) - float(bounds[axis][0])) / grid_size
        for axis in range(3)
    ]
    sample_metadata = []
    for origin, mask, split in zip(origin_array, masks, split_labels, strict=True):
        center = [
            float(bounds[axis][0]) + (int(origin[axis]) + patch_size / 2) * spacing[axis]
            for axis in range(3)
        ]
        sample_metadata.append(
            {
                "sample_id": f"x{origin[0]}_y{origin[1]}_z{origin[2]}",
                "case_id": case_id,
                "reynolds_number": reynolds_number,
                "mesh_resolution": mesh_resolution,
                "solution_time": solution_time,
                "patch_index": origin.tolist(),
                "patch_center": center,
                "patch_center_units": None,
                "valid_voxel_count": int(mask.sum()),
                "split": str(split),
            }
        )

    counts = {
        split: int(np.count_nonzero(split_labels == split))
        for split in ("train", "validation", "buffer")
    }
    if counts["train"] < 32 or counts["validation"] < 8:
        raise GenerativeDataError(
            "The spatially separated patch set is too small for a meaningful pilot experiment."
        )
    return {
        "values": np.stack(patches),
        "masks": np.stack(masks),
        "origins": origin_array,
        "split_labels": split_labels,
        "samples": sample_metadata,
        "counts": counts,
        "bounds": bounds,
        "grid_size": grid_size,
        "patch_size": patch_size,
        "stride": stride,
        "representation": {
            "method": "uniform voxel centers assigned the scalar value of the containing coolant cell",
            "interpolation": "none; piecewise-constant cell-value assignment",
            "mask": "voxels not contained in a coolant cell are marked invalid and excluded from loss/metrics",
            "grid_size": [grid_size, grid_size, grid_size],
            "patch_size": [patch_size, patch_size, patch_size],
            "stride": stride,
            "coordinate_units": None,
            "limitations": [
                "Resampling to a regular grid reduces mesh resolution and does not preserve original cell topology.",
                "Overlapping patches are split by a spatial slab; the gap between training and validation patches is excluded.",
                "This single-case experiment cannot establish generalization to another case or operating condition.",
            ],
        },
    }


def _cache_key(case_id: str, manifest_sha256: str) -> str:
    return hashlib.sha256(f"{case_id}:{manifest_sha256}:{GRID_SIZE}:{PATCH_SIZE}:{PATCH_STRIDE}".encode()).hexdigest()


def _write_cache(cache_path: Path, dataset: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(".json")
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as stream:
        temporary_npz = Path(stream.name)
    try:
        np.savez_compressed(
            temporary_npz,
            values=dataset["values"],
            masks=dataset["masks"],
            origins=dataset["origins"],
            split_labels=dataset["split_labels"],
        )
        temporary_npz.replace(cache_path)
    finally:
        if temporary_npz.exists():
            temporary_npz.unlink()
    payload = {
        key: value
        for key, value in dataset.items()
        if key not in {"values", "masks", "origins", "split_labels"}
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=metadata_path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary_json = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary_json.replace(metadata_path)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()


def load_prepared_dataset(cache_path: Path) -> dict[str, Any]:
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise GenerativeDataError("Prepared real-data patches are unavailable; prepare the selected case again.")
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("values", "masks", "origins", "split_labels")}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise GenerativeDataError("The prepared patch cache is malformed.") from error
    return {**metadata, **arrays}


def prepare_case_patches(
    dataset_path: Path,
    manifest_path: Path,
    cache_root: Path,
    *,
    case_id: str,
) -> tuple[dict[str, Any], str]:
    """Load one verified OpenFOAM case, then cache its bounded voxel-patch sample."""
    manifest = load_manifest(manifest_path)
    if manifest.get("status") != "scanned":
        raise GenerativeDataError("Run a successful dataset scan before preparing generative samples.")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    case = next((item for item in manifest.get("cases", []) if item.get("case_id") == case_id), None)
    if case is None:
        raise GenerativeDataError("The selected case is not present in the current dataset manifest.")
    case_metadata = case.get("metadata", {})
    reynolds_number = case_metadata.get("reynolds_number")
    mesh_resolution = case_metadata.get("mesh_resolution")
    solution_time = case_metadata.get("solution_time")
    if not isinstance(reynolds_number, int) or not isinstance(mesh_resolution, str) or not isinstance(solution_time, str):
        raise GenerativeDataError("The selected case is missing verified Reynolds, mesh, or solution-time metadata.")
    field_record = next(
        (
            field
            for field in case.get("fields", [])
            if field.get("region") == SUPPORTED_REGION
            and field.get("name") == SUPPORTED_FIELD
            and field.get("association") == "cell"
            and field.get("field_class") == "volScalarField"
            and field.get("units") == SUPPORTED_UNITS
        ),
        None,
    )
    if field_record is None:
        raise GenerativeDataError("The selected case does not contain the verified coolant/T scalar field in kelvin.")

    source_root = dataset_path.expanduser().resolve()
    case_path = (source_root / Path(case_id)).resolve()
    try:
        case_path.relative_to(source_root)
    except ValueError as error:
        raise GenerativeDataError("The selected case path escapes the configured dataset directory.") from error
    if not case_path.is_dir():
        raise GenerativeDataError("The selected CFD case directory is unavailable.")
    try:
        options = get_case_field_options(case_path)
    except CfdDataError as error:
        raise GenerativeDataError(str(error)) from error
    if str(options.get("time")) != solution_time:
        raise GenerativeDataError("The selected solution time changed since the dataset scan.")
    if not any(
        option.get("region") == SUPPORTED_REGION
        and option.get("name") == SUPPORTED_FIELD
        and option.get("field_class") == "volScalarField"
        and option.get("units") == SUPPORTED_UNITS
        for option in options.get("fields", [])
    ):
        raise GenerativeDataError("The scanned coolant/T field could not be reverified in the selected case.")

    cache_path = cache_root / f"{_cache_key(case_id, manifest_sha256)}.npz"
    if cache_path.is_file() and cache_path.with_suffix(".json").is_file():
        return load_prepared_dataset(cache_path), manifest_sha256

    reader = vtkPOpenFOAMReader()
    reader.SetFileName(str(case_path / f"{mesh_resolution}.foam"))
    reader.SetCaseType(0)
    reader.UpdateInformation()
    if reader.GetErrorCode() != 0:
        raise GenerativeDataError("The OpenFOAM reader could not inspect the selected case.")
    time_values = reader.GetTimeValues()
    if time_values is None or not any(
        np.isclose(time_values.GetValue(index), float(solution_time))
        for index in range(time_values.GetNumberOfTuples())
    ):
        raise GenerativeDataError("The selected solution time is unavailable in the OpenFOAM reader.")
    reader.DisableAllCellArrays()
    reader.SetCellArrayStatus(SUPPORTED_FIELD, 1)
    reader.SetTimeValue(float(solution_time))
    reader.Update()
    if reader.GetErrorCode() != 0:
        raise GenerativeDataError("The OpenFOAM reader could not load coolant/T for the selected case.")
    mesh = _find_region_mesh(reader.GetOutput(), SUPPORTED_REGION)
    if mesh is None or mesh.GetNumberOfCells() == 0:
        raise GenerativeDataError("The selected coolant internal mesh is empty or unavailable.")
    vtk_field = mesh.GetCellData().GetArray(SUPPORTED_FIELD)
    if vtk_field is None or vtk_field.GetNumberOfComponents() != 1:
        raise GenerativeDataError("The selected coolant/T data is not a scalar cell field.")
    cell_values = vtk_to_numpy(vtk_field)
    if cell_values.shape != (mesh.GetNumberOfCells(),) or not np.isfinite(cell_values).all():
        raise GenerativeDataError("The coolant/T values do not align with mesh cells or contain invalid values.")

    mesh_bounds = mesh.GetBounds()
    bounds = [[float(mesh_bounds[2 * axis]), float(mesh_bounds[2 * axis + 1])] for axis in range(3)]
    if any(not np.isfinite(pair).all() or pair[1] <= pair[0] for pair in bounds):
        raise GenerativeDataError("The selected mesh has invalid spatial bounds.")
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    axes = [
        np.linspace(pair[0], pair[1], GRID_SIZE, endpoint=False)
        + (pair[1] - pair[0]) / (2 * GRID_SIZE)
        for pair in bounds
    ]
    grid_values = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    grid_mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
    for x_index, x_value in enumerate(axes[0]):
        for y_index, y_value in enumerate(axes[1]):
            for z_index, z_value in enumerate(axes[2]):
                cell_id = locator.FindCell((float(x_value), float(y_value), float(z_value)))
                if cell_id >= 0:
                    grid_values[x_index, y_index, z_index] = float(cell_values[cell_id])
                    grid_mask[x_index, y_index, z_index] = True
    if int(grid_mask.sum()) < PATCH_SIZE**3:
        raise GenerativeDataError("The selected coolant region occupies too few voxels at the pilot resolution.")

    prepared = extract_spatial_patches(
        grid_values,
        grid_mask,
        case_id=case_id,
        reynolds_number=reynolds_number,
        mesh_resolution=mesh_resolution,
        solution_time=solution_time,
        bounds=bounds,
    )
    prepared.update(
        {
            "target": {
                "region": SUPPORTED_REGION,
                "field": SUPPORTED_FIELD,
                "units": SUPPORTED_UNITS,
                "field_class": "volScalarField",
                "association": "cell",
            },
            "cell_count": int(mesh.GetNumberOfCells()),
            "manifest_sha256": manifest_sha256,
        }
    )
    _write_cache(cache_path, prepared)
    reader.SetFileName("")
    return prepared, manifest_sha256


def prepared_cache_path(cache_root: Path, case_id: str, manifest_sha256: str) -> Path:
    return cache_root / f"{_cache_key(case_id, manifest_sha256)}.npz"

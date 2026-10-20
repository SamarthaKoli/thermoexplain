"""Read verified OpenFOAM cell fields and matching cell-center coordinates."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkMultiBlockDataSet
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.vtkIOParallel import vtkPOpenFOAMReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from backend.services.cfd_analysis import CfdDataError, get_case_field_options
from backend.services.dataset_inspector import load_manifest


DEFAULT_REGION = "coolant"
DEFAULT_FIELD = "T"
DEFAULT_MAX_SAMPLES_PER_CASE = 5_000
MAX_SAMPLES_PER_CASE = 100_000
FEATURE_NAMES = ("x", "y", "z", "reynolds_number")


class SurrogateDataError(Exception):
    """A verified target or coordinate-to-cell mapping is unavailable."""


def validate_sample_limit(sample_limit: int) -> int:
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int):
        raise SurrogateDataError("The per-case sample limit must be an integer.")
    if sample_limit < 100 or sample_limit > MAX_SAMPLES_PER_CASE:
        raise SurrogateDataError(
            f"The per-case sample limit must be between 100 and {MAX_SAMPLES_PER_CASE}."
        )
    return sample_limit


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


def read_case_training_samples(
    case_path: Path,
    *,
    case_id: str,
    reynolds_number: int,
    mesh_resolution: str,
    max_samples: int,
    seed: int,
    region: str = DEFAULT_REGION,
    field_name: str = DEFAULT_FIELD,
) -> dict[str, Any]:
    """Load one case, verify VTK cell/value alignment, then retain a reproducible sample."""
    max_samples = validate_sample_limit(max_samples)
    try:
        options = get_case_field_options(case_path)
    except CfdDataError as error:
        raise SurrogateDataError(str(error)) from error
    field = next(
        (
            option
            for option in options["fields"]
            if option["region"] == region and option["name"] == field_name
        ),
        None,
    )
    if (
        field is None
        or field["field_class"] != "volScalarField"
        or field["association"] != "cell"
        or not field["statistics_supported"]
    ):
        raise SurrogateDataError(
            f"Target {region}/{field_name} is not a supported scalar cell field in {case_id}."
        )

    try:
        solution_time = float(options["time"])
    except (TypeError, ValueError) as error:
        raise SurrogateDataError(
            f"The selected solution time for {case_id} is not numeric."
        ) from error

    reader = vtkPOpenFOAMReader()
    reader.SetFileName(str(case_path / f"{mesh_resolution}.foam"))
    reader.SetCaseType(0)
    reader.UpdateInformation()
    if reader.GetErrorCode() != 0:
        raise SurrogateDataError(
            f"The OpenFOAM reader could not inspect case {case_id}."
        )
    available_times = reader.GetTimeValues()
    if available_times is None or not any(
        np.isclose(available_times.GetValue(index), solution_time)
        for index in range(available_times.GetNumberOfTuples())
    ):
        raise SurrogateDataError(
            f"The reader does not expose the verified solution time for {case_id}."
        )

    reader.DisableAllCellArrays()
    reader.SetCellArrayStatus(field_name, 1)
    reader.SetTimeValue(solution_time)
    reader.Update()
    if reader.GetErrorCode() != 0:
        raise SurrogateDataError(
            f"The OpenFOAM reader could not load {region}/{field_name} for {case_id}."
        )
    mesh = _find_region_mesh(reader.GetOutput(), region)
    if mesh is None:
        raise SurrogateDataError(
            f"The reader did not return the {region} internal mesh for {case_id}."
        )
    vtk_values = mesh.GetCellData().GetArray(field_name)
    if vtk_values is None or vtk_values.GetNumberOfComponents() != 1:
        raise SurrogateDataError(
            f"The reader did not return a scalar {region}/{field_name} cell field for {case_id}."
        )
    values = vtk_to_numpy(vtk_values)
    centers_filter = vtkCellCenters()
    centers_filter.SetInputData(mesh)
    centers_filter.Update()
    centers = vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData())
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or centers.shape[0] != mesh.GetNumberOfCells()
        or values.ndim != 1
        or values.shape[0] != mesh.GetNumberOfCells()
    ):
        raise SurrogateDataError(
            f"Cell-center coordinates and field values do not align for {case_id}."
        )
    if not np.isfinite(centers).all() or not np.isfinite(values).all():
        raise SurrogateDataError(
            f"Non-finite coordinates or target values were found in {case_id}."
        )

    cell_count = int(mesh.GetNumberOfCells())
    bounds = np.column_stack((centers.min(axis=0), centers.max(axis=0)))
    count = min(max_samples, cell_count)
    rng = np.random.default_rng(seed)
    indexes = (
        np.arange(cell_count)
        if count == cell_count
        else np.sort(rng.choice(cell_count, size=count, replace=False))
    )
    sampled_centers = np.asarray(centers[indexes], dtype=np.float64)
    sampled_values = np.asarray(values[indexes], dtype=np.float64)
    result = {
        "case_id": case_id,
        "reynolds_number": reynolds_number,
        "mesh_resolution": mesh_resolution,
        "solution_time": options["time"],
        "region": region,
        "field": field_name,
        "target_units": field["units"],
        "cell_count": cell_count,
        "sample_count": count,
        "coordinates": sampled_centers,
        "values": sampled_values,
        "bounds": bounds.astype(float).tolist(),
        "coordinate_units": None,
        "coordinate_method": "VTK cell centers from the OpenFOAM internal mesh",
    }
    reader.SetFileName("")
    del centers_filter, vtk_values, mesh, reader
    gc.collect()
    return result


def collect_verified_samples(
    dataset_path: Path,
    manifest_path: Path,
    *,
    max_samples_per_case: int = DEFAULT_MAX_SAMPLES_PER_CASE,
    seed: int = 42,
    region: str = DEFAULT_REGION,
    field_name: str = DEFAULT_FIELD,
) -> dict[str, Any]:
    """Collect one bounded, seeded sample from every compatible scanned case."""
    max_samples_per_case = validate_sample_limit(max_samples_per_case)
    manifest = load_manifest(manifest_path)
    if manifest.get("status") != "scanned":
        raise SurrogateDataError("Run a successful dataset scan before training a model.")
    source_root = dataset_path.expanduser().resolve()
    cases = manifest.get("cases", [])
    compatible_cases = []
    for case in cases:
        metadata = case.get("metadata", {})
        reynolds_number = metadata.get("reynolds_number")
        mesh_resolution = metadata.get("mesh_resolution")
        if not isinstance(reynolds_number, int) or not isinstance(mesh_resolution, str):
            continue
        case_path = (source_root / Path(case["case_id"])).resolve()
        try:
            case_path.relative_to(source_root)
        except ValueError as error:
            raise SurrogateDataError(
                "A scanned case path is outside the configured dataset root."
            ) from error
        compatible_cases.append(
            (case, case_path, reynolds_number, mesh_resolution)
        )
    if len(compatible_cases) < 2:
        raise SurrogateDataError(
            "At least two verified CFD cases with Reynolds number and mesh metadata are required."
        )

    results = []
    available_units: set[str | None] = set()
    for index, (case, case_path, reynolds_number, mesh_resolution) in enumerate(
        sorted(compatible_cases, key=lambda item: item[0]["case_id"])
    ):
        result = read_case_training_samples(
            case_path,
            case_id=case["case_id"],
            reynolds_number=reynolds_number,
            mesh_resolution=mesh_resolution,
            max_samples=max_samples_per_case,
            seed=seed + index,
            region=region,
            field_name=field_name,
        )
        available_units.add(result["target_units"])
        results.append(result)
    if len(available_units) != 1:
        raise SurrogateDataError(
            "The selected target has inconsistent or unknown units across the scanned cases."
        )
    if len({result["reynolds_number"] for result in results}) < 2:
        raise SurrogateDataError(
            "The selected cases do not contain Reynolds-number variation."
        )
    return {
        "cases": results,
        "region": region,
        "field": field_name,
        "target_units": next(iter(available_units)),
        "coordinate_units": None,
        "coordinate_method": "VTK cell centers from the OpenFOAM internal mesh",
        "dataset_version": {
            "manifest_generated_at": manifest.get("generated_at"),
            "case_count": len(results),
        },
    }

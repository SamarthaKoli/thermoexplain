"""Build bounded, field-mapped boundary surfaces from scanned OpenFOAM cases."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkMultiBlockDataSet
from vtkmodules.vtkFiltersCore import (
    vtkCellDataToPointData,
    vtkDecimatePro,
    vtkTriangleFilter,
)
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkIOParallel import vtkPOpenFOAMReader

from backend.services.cfd_analysis import CfdDataError, get_case_field_options

MAX_RENDER_TRIANGLES = 12_000
RENDER_VALUE_ARRAY = "__thermoexplain_render_value"
MAX_VTK_SURFACE_BYTES = 100 * 1024 * 1024


def _find_region_mesh(
    data: Any, region: str, path: tuple[str, ...] = ()
) -> Any | None:
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


def _scalar_values(values: np.ndarray, value_type: str, component: str) -> np.ndarray:
    if values.ndim == 1:
        if component != "value":
            raise CfdDataError("The selected scalar field only supports its stored value.", 422)
        return values
    if values.ndim != 2:
        raise CfdDataError("The selected field has an unsupported data shape.", 422)
    if component == "magnitude":
        if value_type == "symmTensor" and values.shape[1] == 6:
            return np.sqrt(
                values[:, 0] ** 2
                + 2 * values[:, 1] ** 2
                + 2 * values[:, 2] ** 2
                + values[:, 3] ** 2
                + 2 * values[:, 4] ** 2
                + values[:, 5] ** 2
            )
        return np.linalg.norm(values, axis=1)
    if value_type == "vector" and component in {"x", "y", "z"}:
        component_index = {"x": 0, "y": 1, "z": 2}[component]
        if values.shape[1] == 3:
            return values[:, component_index]
    raise CfdDataError("The selected field component cannot be rendered.", 422)


def _read_surface(
    case_path: Path, region: str, field_name: str, time_name: str
) -> tuple[Any, str]:
    reader = vtkPOpenFOAMReader()
    reader.SetFileName(str(case_path / f"{case_path.name}.foam"))
    reader.SetCaseType(0)
    reader.UpdateInformation()
    if reader.GetErrorCode() != 0:
        raise CfdDataError("The OpenFOAM reader could not inspect the selected case.", 422)
    time_values = reader.GetTimeValues()
    if time_values is None or not any(
        np.isclose(time_values.GetValue(index), float(time_name))
        for index in range(time_values.GetNumberOfTuples())
    ):
        raise CfdDataError("The selected solution time is unavailable to the OpenFOAM reader.", 422)

    reader.DisableAllCellArrays()
    reader.DisableAllPointArrays()
    reader.SetCellArrayStatus(field_name, 1)
    reader.SetPointArrayStatus(field_name, 1)
    reader.SetTimeValue(float(time_name))
    reader.Update()
    if reader.GetErrorCode() != 0:
        raise CfdDataError("The OpenFOAM reader could not load the selected field.", 422)

    mesh = _find_region_mesh(reader.GetOutput(), region)
    if mesh is None:
        raise CfdDataError(
            "The reader did not return the selected region's internal mesh.", 422
        )
    cell_values = mesh.GetCellData().GetArray(field_name)
    point_values = mesh.GetPointData().GetArray(field_name)
    if cell_values is not None:
        association = "cell"
    elif point_values is not None:
        association = "point"
    else:
        raise CfdDataError(
            "The selected field is not available on the selected region's mesh.", 422
        )

    surface = vtkDataSetSurfaceFilter()
    surface.SetInputData(mesh)
    surface.Update()
    if surface.GetErrorCode() != 0 or surface.GetOutput().GetNumberOfCells() == 0:
        raise CfdDataError("The selected region did not produce a renderable surface.", 422)

    source = surface.GetOutput()
    if association == "cell":
        cell_to_point = vtkCellDataToPointData()
        cell_to_point.SetInputData(source)
        cell_to_point.PassCellDataOff()
        cell_to_point.Update()
        source = cell_to_point.GetOutput()
    else:
        cell_to_point = None

    vtk_values = source.GetPointData().GetArray(field_name)
    if vtk_values is None:
        raise CfdDataError(
            "The selected field could not be mapped to the rendered surface points.", 422
        )
    source_values = vtk_to_numpy(vtk_values)
    values_array = vtkDoubleArray()
    values_array.SetName(RENDER_VALUE_ARRAY)
    values_array.SetNumberOfComponents(vtk_values.GetNumberOfComponents())
    values_array.SetNumberOfTuples(source_values.shape[0])
    for index, row in enumerate(source_values):
        if np.ndim(row) == 0:
            values_array.SetValue(index, float(row))
        else:
            values_array.SetTuple(index, row)
    source.GetPointData().AddArray(values_array)
    source.GetPointData().SetActiveScalars(RENDER_VALUE_ARRAY)

    triangles = vtkTriangleFilter()
    triangles.SetInputData(source)
    triangles.PassVertsOff()
    triangles.PassLinesOff()
    triangles.Update()
    poly_data = triangles.GetOutput()
    if triangles.GetErrorCode() != 0 or poly_data.GetNumberOfPolys() == 0:
        raise CfdDataError("The selected surface could not be triangulated for rendering.", 422)
    reader.SetFileName("")
    del mesh, reader, surface, triangles
    if cell_to_point is not None:
        del cell_to_point
    gc.collect()
    return poly_data, association


def serialize_surface(
    poly_data: Any,
    *,
    case_id: str,
    region: str,
    field_name: str,
    units: str | None,
    association: str,
    component: str,
    value_type: str,
    mapping_method: str,
) -> dict[str, Any]:
    triangle_count = int(poly_data.GetNumberOfPolys())
    if triangle_count == 0:
        raise CfdDataError("No renderable triangles were found in the selected surface.", 422)

    render_data = poly_data.GetPointData().GetArray(RENDER_VALUE_ARRAY)
    if render_data is None:
        raise CfdDataError("The selected physical field was not mapped to the surface.", 422)
    raw_values = vtk_to_numpy(render_data)
    scalar_values = _scalar_values(raw_values, value_type, component)
    points = vtk_to_numpy(poly_data.GetPoints().GetData())
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] != scalar_values.shape[0]
        or not np.isfinite(points).all()
        or not np.isfinite(scalar_values).all()
    ):
        raise CfdDataError(
            "The surface geometry or mapped field contains invalid or mismatched values.",
            422,
        )

    display_mesh = poly_data
    decimator = None
    if triangle_count > MAX_RENDER_TRIANGLES:
        decimator = vtkDecimatePro()
        decimator.SetInputData(poly_data)
        decimator.SetTargetReduction(1 - MAX_RENDER_TRIANGLES / triangle_count)
        decimator.PreserveTopologyOn()
        decimator.Update()
        display_mesh = decimator.GetOutput()
        if (
            decimator.GetErrorCode() != 0
            or display_mesh.GetNumberOfPolys() == 0
            or display_mesh.GetPointData().GetArray(RENDER_VALUE_ARRAY) is None
        ):
            raise CfdDataError(
                "The field-mapped surface could not be safely simplified for display.",
                422,
            )

    points = vtk_to_numpy(display_mesh.GetPoints().GetData()).astype(float, copy=False)
    render_data = display_mesh.GetPointData().GetArray(RENDER_VALUE_ARRAY)
    values = _scalar_values(vtk_to_numpy(render_data), value_type, component)
    if (
        values.shape[0] != points.shape[0]
        or not np.isfinite(points).all()
        or not np.isfinite(values).all()
    ):
        raise CfdDataError(
            "The simplified surface field no longer matches its geometry.", 422
        )

    polygons = display_mesh.GetPolys()
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    connectivity = vtk_to_numpy(polygons.GetConnectivityArray())
    face_count = int(display_mesh.GetNumberOfPolys())
    if (
        offsets.size != face_count + 1
        or not np.all(np.diff(offsets) == 3)
        or connectivity.size != face_count * 3
    ):
        raise CfdDataError("The rendered surface contains unsupported cell topology.", 422)
    faces = connectivity.reshape(-1, 3).astype(np.int64, copy=False)
    if faces.shape[0] > MAX_RENDER_TRIANGLES:
        raise CfdDataError("The surface exceeded the safe rendering limit.", 422)
    return {
        "case_id": case_id,
        "region": region,
        "field": field_name,
        "units": units,
        "component": component,
        "geometry_type": "boundary_surface",
        "source_format": "processor-decomposed OpenFOAM",
        "source_association": association,
        "render_association": "point",
        "mapping_method": mapping_method,
        "triangle_count": int(faces.shape[0]),
        "source_triangle_count": triangle_count,
        "simplified": int(faces.shape[0]) < triangle_count,
        "points": points.tolist(),
        "triangles": faces.tolist(),
        "values": values.astype(float, copy=False).tolist(),
        "scalar_range": [float(values.min()), float(values.max())],
        "bounds": [
            float(points[:, axis].min()) for axis in range(3)
        ] + [float(points[:, axis].max()) for axis in range(3)],
        "diagnostic": (
            "This is the extracted boundary surface of the selected OpenFOAM region. "
            "It is not a volumetric rendering."
        ),
    }


def _serialize_geometry_only(
    poly_data: Any, *, case_id: str, available_arrays: list[str], diagnostic: str
) -> dict[str, Any]:
    triangles = vtkTriangleFilter()
    triangles.SetInputData(poly_data)
    triangles.PassVertsOff()
    triangles.PassLinesOff()
    triangles.Update()
    display_mesh = triangles.GetOutput()
    if triangles.GetErrorCode() != 0 or display_mesh.GetNumberOfPolys() == 0:
        raise CfdDataError("The available VTK surface could not be triangulated.", 422)
    source_triangle_count = int(display_mesh.GetNumberOfPolys())
    if source_triangle_count > MAX_RENDER_TRIANGLES:
        decimator = vtkDecimatePro()
        decimator.SetInputData(display_mesh)
        decimator.SetTargetReduction(1 - MAX_RENDER_TRIANGLES / source_triangle_count)
        decimator.PreserveTopologyOn()
        decimator.Update()
        if decimator.GetErrorCode() != 0 or decimator.GetOutput().GetNumberOfPolys() == 0:
            raise CfdDataError("The VTK surface could not be simplified for display.", 422)
        display_mesh = decimator.GetOutput()

    points = vtk_to_numpy(display_mesh.GetPoints().GetData()).astype(float, copy=False)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CfdDataError("The VTK surface contains invalid point coordinates.", 422)
    polygons = display_mesh.GetPolys()
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    connectivity = vtk_to_numpy(polygons.GetConnectivityArray())
    triangle_count = int(display_mesh.GetNumberOfPolys())
    if (
        offsets.size != triangle_count + 1
        or not np.all(np.diff(offsets) == 3)
        or connectivity.size != triangle_count * 3
    ):
        raise CfdDataError("The VTK surface contains unsupported cell topology.", 422)
    faces = connectivity.reshape(-1, 3).astype(np.int64, copy=False)
    return {
        "case_id": case_id,
        "region": None,
        "field": None,
        "units": None,
        "component": None,
        "geometry_type": "surface_geometry",
        "source_format": "Legacy VTK POLYDATA",
        "source_association": "none",
        "render_association": "none",
        "mapping_method": "No selected physical field is mapped to this geometry.",
        "triangle_count": triangle_count,
        "source_triangle_count": source_triangle_count,
        "simplified": triangle_count < source_triangle_count,
        "points": points.tolist(),
        "triangles": faces.tolist(),
        "values": [],
        "scalar_range": None,
        "bounds": [
            float(points[:, axis].min()) for axis in range(3)
        ] + [float(points[:, axis].max()) for axis in range(3)],
        "available_arrays": available_arrays,
        "diagnostic": diagnostic,
    }


def _find_layer_info_surface(case_path: Path) -> tuple[Any, list[str]]:
    layer_info = case_path / "layerInfo"
    if not layer_info.is_dir():
        raise CfdDataError("No VTK layer-info surface is available for this case.", 422)
    for path in sorted(layer_info.rglob("*.vtk")):
        try:
            if path.stat().st_size > MAX_VTK_SURFACE_BYTES:
                continue
        except OSError:
            continue
        try:
            with path.open("rb") as stream:
                header = stream.read(1024)
        except OSError:
            continue
        if b"DATASET POLYDATA" not in header:
            continue
        reader = vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        if reader.GetErrorCode() != 0:
            continue
        poly_data = reader.GetOutput()
        if poly_data.GetNumberOfPoints() == 0 or poly_data.GetNumberOfPolys() == 0:
            continue
        arrays = [
            name
            for attributes in (poly_data.GetPointData(), poly_data.GetCellData())
            for name in (
                attributes.GetArrayName(index)
                for index in range(attributes.GetNumberOfArrays())
            )
            if name
        ]
        return poly_data, sorted(set(arrays))
    raise CfdDataError("No valid VTK POLYDATA surface is available for this case.", 422)


def _render_layer_info_surface(
    case_path: Path,
    *,
    case_id: str,
    region: str,
    field: dict[str, Any],
    field_name: str,
    component: str,
    reader_diagnostic: str,
) -> dict[str, Any]:
    poly_data, available_arrays = _find_layer_info_surface(case_path)
    association = (
        "point"
        if poly_data.GetPointData().GetArray(field_name) is not None
        else "cell"
        if poly_data.GetCellData().GetArray(field_name) is not None
        else None
    )
    if association is not None and association == field["association"]:
        triangles = vtkTriangleFilter()
        triangles.SetInputData(poly_data)
        triangles.PassVertsOff()
        triangles.PassLinesOff()
        triangles.Update()
        source = triangles.GetOutput()
        if association == "cell":
            cell_to_point = vtkCellDataToPointData()
            cell_to_point.SetInputData(source)
            cell_to_point.PassCellDataOff()
            cell_to_point.Update()
            source = cell_to_point.GetOutput()
        values = source.GetPointData().GetArray(field_name)
        if values is not None:
            value_array = vtkDoubleArray()
            value_array.SetName(RENDER_VALUE_ARRAY)
            raw_values = vtk_to_numpy(values)
            value_array.SetNumberOfComponents(values.GetNumberOfComponents())
            value_array.SetNumberOfTuples(raw_values.shape[0])
            for index, row in enumerate(raw_values):
                if np.ndim(row) == 0:
                    value_array.SetValue(index, float(row))
                else:
                    value_array.SetTuple(index, row)
            source.GetPointData().AddArray(value_array)
            try:
                return serialize_surface(
                    source,
                    case_id=case_id,
                    region=region,
                    field_name=field_name,
                    units=field["units"],
                    association=association,
                    component=component,
                    value_type=field["value_type"],
                    mapping_method=(
                        "Cell-to-point interpolation on the VTK surface"
                        if association == "cell"
                        else "Stored point values on the VTK surface"
                    ),
                )
            except CfdDataError as error:
                reader_diagnostic = f"{reader_diagnostic} VTK field mapping also failed: {error}"

    reason = (
        f"The VTK POLYDATA surface contains geometry but no compatible "
        f"{region}/{field_name} physical field values. {reader_diagnostic}"
    )
    return _serialize_geometry_only(
        poly_data,
        case_id=case_id,
        available_arrays=available_arrays,
        diagnostic=reason,
    )


def get_case_spatial_visualization(
    case_path: Path,
    *,
    case_id: str,
    region: str,
    field_name: str,
    component: str,
) -> dict[str, Any]:
    options = get_case_field_options(case_path)
    field = next(
        (
            item for item in options["fields"]
            if item["region"] == region and item["name"] == field_name
        ),
        None,
    )
    if field is None or not field["statistics_supported"]:
        raise CfdDataError(
            "The selected physical field is not available for spatial visualization.",
            404,
        )
    if component not in field["components"]:
        raise CfdDataError(
            "The selected component is not supported for this physical field.", 400
        )
    try:
        if field["association"] not in {"cell", "point"}:
            raise CfdDataError(
                "Only cell- and point-associated OpenFOAM fields can be mapped to this surface.",
                422,
            )
        poly_data, association = _read_surface(
            case_path, region, field_name, options["time"]
        )
        mapping_method = (
            "Cell-to-point interpolation on the extracted boundary surface"
            if association == "cell"
            else "Stored point values on the extracted boundary surface"
        )
        return serialize_surface(
            poly_data,
            case_id=case_id,
            region=region,
            field_name=field_name,
            units=field["units"],
            association=association,
            component=component,
            value_type=field["value_type"],
            mapping_method=mapping_method,
        )
    except CfdDataError as error:
        try:
            return _render_layer_info_surface(
                case_path,
                case_id=case_id,
                region=region,
                field=field,
                field_name=field_name,
                component=component,
                reader_diagnostic=str(error),
            )
        except CfdDataError:
            raise error

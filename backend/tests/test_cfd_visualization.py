import json

from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

from backend.app import create_app
from backend.config import Settings
from backend.routes import dataset as dataset_routes
from backend.services import cfd_visualization
from backend.services.cfd_analysis import CfdDataError, get_case_field_options
from backend.services.cfd_visualization import (
    RENDER_VALUE_ARRAY,
    _serialize_geometry_only,
    get_case_spatial_visualization,
    serialize_surface,
)


def make_surface():
    points = vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(1, 1, 0)
    points.InsertNextPoint(0, 1, 0)
    polygons = vtkCellArray()
    polygons.InsertNextCell(3)
    polygons.InsertCellPoint(0)
    polygons.InsertCellPoint(1)
    polygons.InsertCellPoint(2)
    polygons.InsertNextCell(3)
    polygons.InsertCellPoint(0)
    polygons.InsertCellPoint(2)
    polygons.InsertCellPoint(3)
    surface = vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polygons)
    values = vtkDoubleArray()
    values.SetName(RENDER_VALUE_ARRAY)
    values.SetNumberOfValues(4)
    for index, value in enumerate((300.0, 301.0, 302.0, 303.0)):
        values.SetValue(index, value)
    surface.GetPointData().AddArray(values)
    return surface


def test_surface_serialization_preserves_geometry_and_field_values():
    result = serialize_surface(
        make_surface(),
        case_id="Re560/coarse",
        region="cell1",
        field_name="T",
        units="K",
        association="cell",
        component="value",
        value_type="scalar",
        mapping_method="Cell-to-point interpolation on the extracted boundary surface",
    )

    assert result["geometry_type"] == "boundary_surface"
    assert result["source_association"] == "cell"
    assert result["render_association"] == "point"
    assert result["triangle_count"] == 2
    assert result["scalar_range"] == [300.0, 303.0]
    assert result["points"][1] == [1.0, 0.0, 0.0]
    assert result["triangles"] == [[0, 1, 2], [0, 2, 3]]
    assert result["units"] == "K"
    assert "not a volumetric rendering" in result["diagnostic"]


def test_vector_component_serialization_uses_selected_component():
    surface = make_surface()
    vectors = np.asarray(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    vtk_vectors = numpy_to_vtk(vectors, deep=True)
    vtk_vectors.SetNumberOfComponents(3)
    vtk_vectors.SetName(RENDER_VALUE_ARRAY)
    surface.GetPointData().RemoveArray(RENDER_VALUE_ARRAY)
    surface.GetPointData().AddArray(vtk_vectors)

    result = serialize_surface(
        surface,
        case_id="Re560/coarse",
        region="cell1",
        field_name="U",
        units="m/s",
        association="cell",
        component="magnitude",
        value_type="vector",
        mapping_method="Cell-to-point interpolation on the extracted boundary surface",
    )

    assert result["values"] == [5.0, 12.0, 1.0, 2.0]
    assert result["scalar_range"] == [1.0, 12.0]


def test_surface_geometry_without_a_physical_field_has_no_fabricated_values():
    result = _serialize_geometry_only(
        make_surface(),
        case_id="Re560/coarse",
        available_arrays=["numLayers", "level", "fch"],
        diagnostic="The selected temperature is not present on the layer-info surface.",
    )

    assert result["geometry_type"] == "surface_geometry"
    assert result["source_format"] == "Legacy VTK POLYDATA"
    assert result["field"] is None
    assert result["values"] == []
    assert result["scalar_range"] is None
    assert result["available_arrays"] == ["numLayers", "level", "fch"]
    assert result["triangle_count"] == 2


def test_spatial_visualization_falls_back_to_vtk_geometry_only(tmp_path, monkeypatch):
    case_path = tmp_path / "Re560" / "coarse"
    layer_info = case_path / "layerInfo"
    layer_info.mkdir(parents=True)
    (layer_info / "layerInfo.vtk").write_text(
        "# vtk DataFile Version 3.0\n"
        "layer surface\n"
        "ASCII\n"
        "DATASET POLYDATA\n"
        "POINTS 4 float\n"
        "0 0 0 1 0 0 1 1 0 0 1 0\n"
        "POLYGONS 2 8\n"
        "3 0 1 2\n"
        "3 0 2 3\n"
        "POINT_DATA 4\n"
        "SCALARS level float 1\n"
        "LOOKUP_TABLE default\n"
        "0 1 1 0\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        cfd_visualization,
        "get_case_field_options",
        lambda _: {
            "fields": [
                {
                    "region": "coolant",
                    "name": "phi",
                    "statistics_supported": True,
                    "components": ["value"],
                    "association": "face",
                    "units": None,
                    "value_type": "scalar",
                }
            ],
            "time": "200",
        },
    )

    def reject_face_surface(*_args):
        raise CfdDataError(
            "Face-associated fields cannot map to the volume boundary.", 422
        )

    monkeypatch.setattr(
        cfd_visualization, "_read_surface", reject_face_surface
    )

    result = get_case_spatial_visualization(
        case_path,
        case_id="Re560/coarse",
        region="coolant",
        field_name="phi",
        component="value",
    )

    assert result["geometry_type"] == "surface_geometry"
    assert result["field"] is None
    assert result["values"] == []
    assert result["scalar_range"] is None
    assert result["available_arrays"] == ["level"]
    assert result["triangle_count"] == 2
    assert "cell- and point-associated OpenFOAM fields" in result["diagnostic"]


def test_spatial_visualization_rejects_field_not_available_in_selected_case(tmp_path):
    case_path = tmp_path / "Re560" / "coarse"
    case_path.mkdir(parents=True)
    (case_path / "processor0").mkdir()
    (case_path / "processor0" / "200").mkdir()
    (case_path / "processor0" / "200" / "cell1").mkdir()
    (case_path / "processor0" / "200" / "cell1" / "T").write_text(
        "FoamFile\n{\n class volScalarField;\n}\ndimensions [0 0 0 1 0 0 0];\n"
        "internalField nonuniform List<scalar> 1\n(\n300\n);",
        encoding="ascii",
    )
    with pytest.raises(CfdDataError, match="not available"):
        get_case_spatial_visualization(
            case_path,
            case_id="Re560/coarse",
            region="cell1",
            field_name="p",
            component="value",
        )


def test_spatial_visualization_route_uses_validated_scanned_case_path(
    tmp_path, monkeypatch
):
    raw_path = tmp_path / "raw"
    case_path = raw_path / "Re560" / "coarse"
    case_path.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "Re560/coarse",
                        "metadata": {
                            "reynolds_number": 560,
                            "mesh_resolution": "coarse",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw_path,
        manifest_path=manifest_path,
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    calls = {}

    def fake_visualization(selected_path, **kwargs):
        calls.update(path=selected_path, **kwargs)
        return {"geometry_type": "boundary_surface", "case_id": kwargs["case_id"]}

    monkeypatch.setattr(
        dataset_routes, "get_case_spatial_visualization", fake_visualization
    )
    response = app.test_client().get(
        "/api/dataset/spatial-visualization"
        "?case_id=Re560%2Fcoarse&region=cell1&field=T&component=value"
    )

    assert response.status_code == 200
    assert response.get_json()["geometry_type"] == "boundary_surface"
    assert calls["path"] == case_path
    assert calls["region"] == "cell1"
    assert calls["field_name"] == "T"


def test_spatial_visualization_route_rejects_unselected_case(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "cases": []}), encoding="utf-8"
    )
    app = create_app(
        Settings(
            cors_origins=("http://localhost:5173",),
            dataset_path=tmp_path / "raw",
            manifest_path=manifest_path,
        )
    )
    app.config.update(TESTING=True)

    response = app.test_client().get(
        "/api/dataset/spatial-visualization"
        "?case_id=..%2Foutside&region=cell1&field=T"
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == (
        "Case was not found in the latest dataset scan."
    )

import json

import pytest

from backend.app import create_app
from backend.config import Settings
from backend.services.cfd_analysis import (
    CfdDataError,
    calculate_field_statistics,
    compare_mesh_cases,
    get_case_field_options,
)


CASE_ID = "Re560/coarse"


def write_field(path, field_class, value_type, dimensions, internal_field):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile\n{\n"
        f"    class {field_class};\n"
        f"    object {path.name};\n"
        "}\n"
        f"dimensions [{dimensions}];\n"
        f"internalField {internal_field};\n",
        encoding="ascii",
    )


def make_case(root, case_id=CASE_ID):
    case_path = root / case_id
    write_field(
        case_path / "processor0" / "200" / "coolant" / "T",
        "volScalarField",
        "scalar",
        "0 0 0 1 0 0 0",
        "nonuniform List<scalar> 2\n(\n300\n301\n)",
    )
    write_field(
        case_path / "processor1" / "200" / "coolant" / "T",
        "volScalarField",
        "scalar",
        "0 0 0 1 0 0 0",
        "nonuniform List<scalar> 1\n(\n302\n)",
    )
    write_field(
        case_path / "processor0" / "200" / "coolant" / "U",
        "volVectorField",
        "vector",
        "0 1 -1 0 0 0 0",
        "nonuniform List<vector> 1\n(\n(3 4 0)\n)",
    )
    write_field(
        case_path / "processor1" / "200" / "coolant" / "U",
        "volVectorField",
        "vector",
        "0 1 -1 0 0 0 0",
        "nonuniform List<vector> 1\n(\n(0 0 12)\n)",
    )
    write_field(
        case_path / "processor0" / "200" / "coolant" / "stress",
        "volSymmTensorField",
        "symmTensor",
        "1 -1 -2 0 0 0 0",
        "nonuniform List<symmTensor> 1\n(\n(1 2 3 4 5 6)\n)",
    )
    write_field(
        case_path / "processor1" / "200" / "coolant" / "stress",
        "volSymmTensorField",
        "symmTensor",
        "1 -1 -2 0 0 0 0",
        "nonuniform List<symmTensor> 1\n(\n(0 0 0 0 0 0)\n)",
    )
    write_field(
        case_path / "processor0" / "200" / "coolant" / "alphaConv",
        "volScalarField",
        "scalar",
        "1 0 -3 -1 0 0 0",
        "uniform 0",
    )
    write_field(
        case_path / "processor1" / "200" / "coolant" / "alphaConv",
        "volScalarField",
        "scalar",
        "1 0 -3 -1 0 0 0",
        "uniform 0",
    )
    return case_path


def settings_for(tmp_path, case_path, cases=None, gpt_oss_api_key=None, gpt_oss_model=None):
    raw = tmp_path / "raw"
    case_path.relative_to(tmp_path)
    metadata_cases = cases if cases is not None else [
        {
            "case_id": CASE_ID,
            "metadata": {
                "reynolds_number": 560,
                "mesh_resolution": "coarse",
            },
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "cases": metadata_cases}), encoding="utf-8"
    )
    return Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=manifest_path,
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        gpt_oss_api_key=gpt_oss_api_key,
        gpt_oss_model=gpt_oss_model,
    )


def test_case_fields_and_units_are_discovered_from_complete_partitions(tmp_path):
    case_path = make_case(tmp_path / "raw")
    options = get_case_field_options(case_path)

    assert options["time"] == "200"
    assert options["processor_count"] == 2
    temperature = next(
        item for item in options["fields"] if item["region"] == "coolant" and item["name"] == "T"
    )
    velocity = next(
        item for item in options["fields"] if item["region"] == "coolant" and item["name"] == "U"
    )
    assert temperature["units"] == "K"
    assert temperature["association"] == "cell"
    assert velocity["components"] == ["magnitude", "x", "y", "z"]


def test_statistics_are_computed_over_all_partition_values(tmp_path):
    case_path = make_case(tmp_path / "raw")
    stats = calculate_field_statistics(
        case_path,
        case_id=CASE_ID,
        reynolds_number=560,
        mesh_resolution="coarse",
        region="coolant",
        field_name="T",
    )

    assert stats["minimum"] == 300
    assert stats["maximum"] == 302
    assert stats["mean"] == 301
    assert stats["sample_count"] == 3
    assert stats["units"] == "K"
    assert stats["statistic_scope"] == "all internal cell values across processor partitions"
    assert stats["visualization"]["status"] == "boundary_surface_on_request"
    assert "not a volumetric rendering" in stats["visualization"]["reason"]


def test_empty_nonuniform_partition_is_ignored_when_aggregating_field_statistics(tmp_path):
    case_path = make_case(tmp_path / "raw")
    write_field(
        case_path / "processor0" / "200" / "coolant" / "T",
        "volScalarField",
        "scalar",
        "0 0 0 1 0 0 0",
        "nonuniform 0()",
    )

    stats = calculate_field_statistics(
        case_path,
        case_id=CASE_ID,
        reynolds_number=560,
        mesh_resolution="coarse",
        region="coolant",
        field_name="T",
    )

    assert stats["minimum"] == 302
    assert stats["maximum"] == 302
    assert stats["mean"] == 302
    assert stats["sample_count"] == 1


def test_vector_magnitude_is_calculated_from_components(tmp_path):
    case_path = make_case(tmp_path / "raw")
    stats = calculate_field_statistics(
        case_path,
        case_id=CASE_ID,
        reynolds_number=560,
        mesh_resolution="coarse",
        region="coolant",
        field_name="U",
        component="magnitude",
    )

    assert stats["minimum"] == 5
    assert stats["maximum"] == 12
    assert stats["mean"] == 8.5
    assert stats["units"] == "m/s"


def test_symmetric_tensor_statistics_use_frobenius_magnitude(tmp_path):
    stats = calculate_field_statistics(
        make_case(tmp_path / "raw"),
        case_id=CASE_ID,
        reynolds_number=560,
        mesh_resolution="coarse",
        region="coolant",
        field_name="stress",
        component="magnitude",
    )

    assert stats["minimum"] == 0
    assert stats["maximum"] == pytest.approx(129**0.5)
    assert stats["mean"] == pytest.approx((129**0.5) / 2)


def test_uniform_field_statistics_do_not_invent_mesh_element_count(tmp_path):
    stats = calculate_field_statistics(
        make_case(tmp_path / "raw"),
        case_id=CASE_ID,
        reynolds_number=560,
        mesh_resolution="coarse",
        region="coolant",
        field_name="alphaConv",
    )

    assert stats["minimum"] == stats["maximum"] == stats["mean"] == 0
    assert stats["sample_count"] is None
    assert "mesh-element count is not inferred" in stats["statistic_scope"]


def test_unknown_field_and_bad_vector_component_are_rejected(tmp_path):
    case_path = make_case(tmp_path / "raw")
    arguments = {
        "case_id": CASE_ID,
        "reynolds_number": 560,
        "mesh_resolution": "coarse",
        "region": "coolant",
    }
    with pytest.raises(CfdDataError, match="not available"):
        calculate_field_statistics(case_path, field_name="imaginary", **arguments)
    with pytest.raises(CfdDataError, match="not supported"):
        calculate_field_statistics(
            case_path, field_name="U", component="pressure", **arguments
        )


def test_missing_case_data_and_truncated_field_fail_explicitly(tmp_path):
    missing_path = tmp_path / "missing"
    with pytest.raises(CfdDataError, match="unavailable"):
        get_case_field_options(missing_path)

    case_path = make_case(tmp_path / "raw")
    (case_path / "processor1" / "200" / "coolant" / "T").write_text(
        "FoamFile { class volScalarField; }\n"
        "dimensions [0 0 0 1 0 0 0];\n"
        "internalField nonuniform List<scalar> 2\n(\n302\n)",
        encoding="ascii",
    )
    with pytest.raises(CfdDataError, match="truncated"):
        calculate_field_statistics(
            case_path,
            case_id=CASE_ID,
            reynolds_number=560,
            mesh_resolution="coarse",
            region="coolant",
            field_name="T",
        )


def test_mesh_comparison_requires_same_reynolds_and_distinct_meshes():
    with pytest.raises(CfdDataError, match="same verified Reynolds"):
        compare_mesh_cases(
            case_id="Re11240/coarse",
            reference_case_id="Re560/fine",
            case_metadata={"reynolds_number": 11240, "mesh_resolution": "coarse"},
            reference_metadata={"reynolds_number": 560, "mesh_resolution": "fine"},
        )

    result = compare_mesh_cases(
        case_id="Re560/coarse",
        reference_case_id="Re560/fine",
        case_metadata={"reynolds_number": 560, "mesh_resolution": "coarse"},
        reference_metadata={"reynolds_number": 560, "mesh_resolution": "fine"},
    )
    assert result["status"] == "unavailable"
    assert "array index" in result["reason"]


def test_api_case_selection_invalid_field_and_empty_dataset_states(tmp_path):
    raw = tmp_path / "raw"
    case_path = make_case(raw)
    fine_case_path = make_case(raw, "Re560/fine")
    settings = settings_for(
        tmp_path,
        case_path,
        cases=[
            {
                "case_id": CASE_ID,
                "metadata": {"reynolds_number": 560, "mesh_resolution": "coarse"},
            },
            {
                "case_id": "Re560/fine",
                "metadata": {"reynolds_number": 560, "mesh_resolution": "fine"},
            },
        ],
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    options_response = client.get(f"/api/dataset/cfd-fields?case_id={CASE_ID}")
    assert options_response.status_code == 200
    assert options_response.get_json()["fields"]
    assert client.get("/api/dataset/cfd-fields?case_id=missing").status_code == 404
    assert client.get(
        f"/api/dataset/field-statistics?case_id={CASE_ID}&region=coolant&field=missing"
    ).status_code == 404
    assert client.get("/api/dataset/field-statistics").status_code == 400
    comparison = client.post(
        "/api/dataset/mesh-comparison",
        json={"case_id": CASE_ID, "reference_case_id": "Re560/fine"},
    )
    assert comparison.status_code == 200
    assert comparison.get_json()["status"] == "unavailable"
    assert (fine_case_path / "processor0" / "200" / "coolant" / "T").is_file()

    empty_settings = settings_for(
        tmp_path / "empty",
        tmp_path / "empty" / "raw" / "placeholder",
        cases=[],
    )
    empty_app = create_app(empty_settings)
    empty_app.config.update(TESTING=True)
    empty_client = empty_app.test_client()
    assert empty_client.get("/api/dataset/cases").get_json() == {"cases": []}

import json
import tarfile
import zipfile

from backend.services.dataset_inspector import scan_dataset, write_audit_report, write_manifest


def test_empty_directory_manifest_does_not_claim_expected_cases(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / ".gitkeep").touch()

    manifest = scan_dataset(source)

    assert manifest["status"] == "no_dataset_files"
    assert manifest["summary"]["candidate_count"] == 0
    assert manifest["cases"] == []
    assert manifest["field_names"] == []
    assert manifest["warnings"]


def test_vtk_xml_header_reports_only_explicit_fields_and_mesh_counts(tmp_path):
    source = tmp_path / "case" / "mesh.vtu"
    source.parent.mkdir()
    source.write_text(
        '<?xml version="1.0"?><VTKFile type="UnstructuredGrid">'
        '<Piece NumberOfPoints="12" NumberOfCells="4">'
        '<PointData><DataArray Name="Temperature" units="K" format="appended"/>'
        "</PointData></Piece></VTKFile>",
        encoding="utf-8",
    )

    manifest = scan_dataset(source.parent.parent)

    assert manifest["status"] == "scanned"
    assert manifest["cases"][0]["case_id"] == "case"
    assert manifest["cases"][0]["mesh_info"]["point_count"] == 12
    assert manifest["cases"][0]["mesh_info"]["cell_count"] == 4
    assert manifest["cases"][0]["fields"] == [
        {"name": "Temperature", "association": "point", "units": "K"}
    ]
    assert manifest["cases"][0]["metadata"]["reynolds_number"] is None


def test_case_keeps_distinct_mesh_counts_attached_to_each_file(tmp_path):
    case = tmp_path / "case-a"
    case.mkdir()
    (case / "coarse.vtu").write_text(
        '<VTKFile><Piece NumberOfPoints="10" NumberOfCells="4"/></VTKFile>',
        encoding="utf-8",
    )
    (case / "fine.vtu").write_text(
        '<VTKFile><Piece NumberOfPoints="100" NumberOfCells="40"/></VTKFile>',
        encoding="utf-8",
    )

    result = scan_dataset(tmp_path)["cases"][0]

    assert result["mesh_info"] is None
    assert [item["mesh_info"]["cell_count"] for item in result["mesh_info_by_file"]] == [
        4,
        40,
    ]


def test_malformed_legacy_vtk_file_is_reported(tmp_path):
    source = tmp_path / "broken.vtk"
    source.write_text("not a VTK file\n", encoding="utf-8")

    manifest = scan_dataset(tmp_path)

    assert manifest["files"][0]["inspection_error"] is True
    assert "missing its version signature" in manifest["files"][0]["warnings"][0]


def test_csv_fields_are_inspected_but_units_are_not_invented(tmp_path):
    source = tmp_path / "points.csv"
    source.write_text("x [mm],y [mm],z [mm],temperature\n1,2,3,300\n", encoding="utf-8")

    manifest = scan_dataset(tmp_path)
    item = manifest["files"][0]

    assert item["preprocessing_supported"] is True
    assert item["fields"][0]["units"] == "mm"
    assert item["fields"][3]["units"] is None
    assert item["mesh_info"]["point_count"] is None


def test_malformed_csv_rows_are_reported_without_loading_the_whole_file(tmp_path):
    source = tmp_path / "malformed.csv"
    source.write_text("x,y,z\n1,2,3\n4,5\n", encoding="utf-8")

    manifest = scan_dataset(tmp_path)

    assert manifest["files"][0]["format"] == "csv_point_table"
    assert any("Malformed CSV row 3" in warning for warning in manifest["files"][0]["warnings"])
    assert manifest["files"][0]["preprocessing_supported"] is False


def test_openfoam_case_inventory_does_not_invent_units_or_mesh_counts(tmp_path):
    case = tmp_path / "run-a"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
    field = case / "processor0" / "10" / "coolant" / "T"
    field.parent.mkdir(parents=True)
    field.write_text(
        "FoamFile { class volScalarField; }\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        "internalField nonuniform List<scalar> 1\n(\n300\n)",
        encoding="utf-8",
    )

    manifest = scan_dataset(tmp_path)

    assert len(manifest["cases"]) == 1
    assert manifest["cases"][0]["case_id"] == "run-a"
    assert manifest["cases"][0]["fields"][0]["name"] == "T"
    assert manifest["cases"][0]["fields"][0]["units"] is None
    assert manifest["cases"][0]["mesh_info"]["cell_count"] is None


def test_archive_is_listed_without_extracting(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Re_560/coarse/solution.vtu", "<VTKFile/>")

    manifest = scan_dataset(archive_path)

    assert manifest["status"] == "archive_listed"
    assert manifest["files"][0]["path"] == "Re_560/coarse/solution.vtu"
    assert manifest["files"][0]["size_bytes"] == len("<VTKFile/>")
    assert not (tmp_path / "Re_560").exists()
    assert any("no files were extracted" in warning for warning in manifest["warnings"])


def test_tar_archive_listing_uses_archive_metadata_without_extraction(tmp_path):
    member = tmp_path / "member.csv"
    member.write_text("x,y,z\n", encoding="utf-8")
    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(member, arcname="case/points.csv")
    member.unlink()

    manifest = scan_dataset(archive_path)

    assert manifest["status"] == "archive_listed"
    assert manifest["files"][0]["path"] == "case/points.csv"
    assert not (tmp_path / "case").exists()


def test_manifest_and_report_are_written_as_requested(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "reports" / "audit.md"
    manifest = scan_dataset(tmp_path / "missing")

    write_manifest(manifest, manifest_path)
    write_audit_report(manifest, report_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "source_missing"
    assert "No cases have been verified" in report_path.read_text(encoding="utf-8")

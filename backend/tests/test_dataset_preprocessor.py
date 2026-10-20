import csv

import pytest

from backend.services.dataset_preprocessor import preprocess_csv


def write_point_csv(path, rows=10):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x [m]", "y [m]", "z [m]", "temperature [K]"])
        for index in range(rows):
            writer.writerow([index, index + 1, index + 2, 290 + index])


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_preprocessing_preserves_source_and_adds_case_mesh_and_location(tmp_path):
    source = tmp_path / "raw" / "case-a" / "points.csv"
    source.parent.mkdir(parents=True)
    write_point_csv(source, rows=4)
    original = source.read_bytes()

    result = preprocess_csv(
        source,
        tmp_path / "processed",
        case_id="case-a",
        mesh_id="medium",
    )
    rows = read_rows(tmp_path / "processed" / result["output_file"])

    assert source.read_bytes() == original
    assert result["source_rows"] == 4
    assert result["output_rows"] == 4
    assert result["split_status"] == "not_assigned"
    assert rows[0]["_case_id"] == "case-a"
    assert rows[0]["_mesh_id"] == "medium"
    assert rows[0]["_source_row_index"] == "1"
    assert rows[0]["_split_group_id"] == "case-a"
    assert rows[0]["x [m]"] == "0"
    assert rows[0]["temperature [K]"] == "290"


def test_reservoir_sampling_is_bounded_and_reproducible(tmp_path):
    source = tmp_path / "points.csv"
    write_point_csv(source, rows=25)
    output = tmp_path / "processed"

    first = preprocess_csv(source, output, sample_size=5, seed=17)
    second = preprocess_csv(source, output, sample_size=5, seed=17)

    first_indices = [
        row["_source_row_index"]
        for row in read_rows(output / first["output_file"])
    ]
    second_indices = [
        row["_source_row_index"]
        for row in read_rows(output / second["output_file"])
    ]
    assert first["output_rows"] == 5
    assert first["sampling"] == "reservoir sampling"
    assert first["random_seed"] == 17
    assert first_indices == second_indices


def test_preprocessing_requires_explicit_xyz_columns(tmp_path):
    source = tmp_path / "table.csv"
    source.write_text("temperature,pressure\n300,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explicit x, y, and z"):
        preprocess_csv(source, tmp_path / "processed")


def test_preprocessing_rejects_ragged_csv(tmp_path):
    source = tmp_path / "points.csv"
    source.write_text("x,y,z\n1,2,3\n4,5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed CSV row"):
        preprocess_csv(source, tmp_path / "processed")


def test_preprocessing_rejects_output_inside_source_directory(tmp_path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source = source_dir / "points.csv"
    write_point_csv(source, rows=1)

    with pytest.raises(ValueError, match="Output must be separate"):
        preprocess_csv(source, source_dir)


def test_preprocessing_rejects_non_csv_source(tmp_path):
    source = tmp_path / "points.vtu"
    source.write_text("<VTKFile/>", encoding="utf-8")

    with pytest.raises(ValueError, match="supports one existing CSV"):
        preprocess_csv(source, tmp_path / "processed")

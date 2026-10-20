"""Bounded, reproducible preprocessing for explicitly spatial CSV point tables."""

import csv
import json
import random
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.dataset_inspector import MAX_SAMPLE_SIZE


def _coordinate_columns(header: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, name in enumerate(header):
        normalized = re.sub(r"\s*\[[^\]]+\]|\s*\([^)]+\)", "", name).strip().lower()
        if normalized in {"x", "y", "z"}:
            if normalized in found:
                raise ValueError(f"CSV has multiple coordinate columns named {normalized}.")
            found[normalized] = index
    missing = {"x", "y", "z"} - found.keys()
    if missing:
        raise ValueError(
            "CSV preprocessing requires explicit x, y, and z columns; missing: "
            + ", ".join(sorted(missing))
        )
    return found


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def preprocess_csv(
    source: Path,
    output_directory: Path,
    *,
    sample_size: int | None = None,
    seed: int = 42,
    case_id: str | None = None,
    mesh_id: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    output_directory = output_directory.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError("Preprocessing supports one existing CSV file at a time.")
    if sample_size is not None and (
        sample_size < 1 or sample_size > MAX_SAMPLE_SIZE
    ):
        raise ValueError(
            f"sample_size must be between 1 and {MAX_SAMPLE_SIZE:,}."
        )
    if _is_within(source, output_directory):
        raise ValueError("Output must be separate from the source data directory.")

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        stream.seek(0)
        reader = csv.reader(stream, dialect)
        header = next(reader, [])
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV must have a non-empty header row.")
        _coordinate_columns(header)
        rng = random.Random(seed)
        reservoir: list[int] = []
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"Malformed CSV row {row_count + 1}: expected {len(header)} columns, got {len(row)}."
                )
            if sample_size is None:
                continue
            elif len(reservoir) < sample_size:
                reservoir.append(row_count)
            else:
                replacement_index = rng.randint(0, row_count - 1)
                if replacement_index < sample_size:
                    reservoir[replacement_index] = row_count

    selected_indices = None if sample_size is None else set(reservoir)
    run_id = uuid.uuid4().hex[:12]
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"point-samples-{run_id}.csv"
    metadata_path = output_directory / f"point-samples-{run_id}.json"
    resolved_case_id = case_id or source.parent.name or "."
    output_header = [
        "_case_id",
        "_mesh_id",
        "_source_file",
        "_source_row_index",
        "_split_group_id",
        *header,
    ]
    temporary_output = output_path.with_suffix(".csv.tmp")
    expected_output_rows = row_count if selected_indices is None else len(selected_indices)
    written_rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
        input_reader = csv.reader(input_stream, dialect)
        next(input_reader, None)
        with temporary_output.open("w", encoding="utf-8", newline="") as output_stream:
            writer = csv.writer(output_stream)
            writer.writerow(output_header)
            for source_row_index, row in enumerate(input_reader, start=1):
                if selected_indices is not None and source_row_index not in selected_indices:
                    continue
                if len(row) != len(header):
                    temporary_output.unlink(missing_ok=True)
                    raise ValueError(
                        f"CSV changed or became malformed at row {source_row_index + 1}."
                    )
                writer.writerow(
                    [
                        resolved_case_id,
                        mesh_id or "",
                        source_id or source.name,
                        source_row_index,
                        resolved_case_id,
                        *row,
                    ]
                )
                written_rows += 1
    if written_rows != expected_output_rows:
        temporary_output.unlink(missing_ok=True)
        raise ValueError("Source file changed between validation and output passes.")
    temporary_output.replace(output_path)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name,
        "case_id": resolved_case_id,
        "mesh_id": mesh_id,
        "source_rows": row_count,
        "output_rows": written_rows,
        "sample_size": sample_size,
        "sampling": "reservoir sampling" if sample_size is not None and row_count > sample_size else "all rows",
        "random_seed": seed if sample_size is not None and row_count > sample_size else None,
        "coordinate_columns": {
            axis: header[index] for axis, index in _coordinate_columns(header).items()
        },
        "field_names": header,
        "units": "Preserved only when stated in source column names; no unit conversion performed.",
        "transformations": [
            "Validated CSV header and consistent row widths.",
            "Added case, explicitly supplied mesh ID (if any), source file ID, source row index, and split group identifiers.",
            "Preserved all source columns without unit conversion.",
            *(
                [f"Applied seeded reservoir sampling with seed {seed}."]
                if sample_size is not None and row_count > sample_size
                else []
            ),
        ],
        "split_status": "not_assigned",
        "split_safety": (
            "No train/test split was created. Treat _split_group_id as an indivisible group; "
            "provide verified family metadata and spatial blocks before model evaluation."
        ),
        "output_file": output_path.name,
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    latest_path = output_directory / "latest.json"
    temporary_latest = output_directory / "latest.json.tmp"
    temporary_latest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_latest.replace(latest_path)
    return metadata

"""Read-only inventory and bounded metadata inspection for research data."""

import csv
import json
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


VTK_XML_EXTENSIONS = {".vtu", ".vtp", ".vti", ".vtr", ".vts"}
RECOGNIZED_FORMATS = {
    "csv_point_table",
    "vtk_legacy",
    "vtk_xml",
    "openfoam_case_marker",
    "openfoam_case",
}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".7z"}
OPENFOAM_CASE_MARKERS = {
    "controlDict",
    "regionProperties",
    "boundary",
    "points",
    "owner",
    "faces",
    "neighbour",
    "fvSchemes",
    "fvSolution",
    "blockMeshDict",
    "triSurface",
    "turbulenceProperties",
    "materialProperties",
    "caseSetupDict",
    "decomposeParDict",
    "geometryDict",
    "meshDict",
    "projectDict",
    "topoSetDict",
    "p",
    "T",
    "U",
    "phi",
    "alphaConv",
    "rho",
    "k",
    "epsilon",
    "omega",
}
MAX_METADATA_BYTES = 1024 * 1024
MAX_SAMPLE_SIZE = 1_000_000
VTK_XML_TAG = re.compile(
    rb"<(/?)(Piece|DataArray|PointData|CellData|FieldData|Points|Cells)\b([^>]*)>",
    re.IGNORECASE,
)
XML_ATTRIBUTE = re.compile(rb"""([A-Za-z_:][\w:.-]*)\s*=\s*["']([^"']*)["']""")


def _xml_attributes(raw: bytes) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in XML_ATTRIBUTE.findall(raw)
    }


def _unit_from_name(name: str) -> str | None:
    match = re.search(r"(?:\[([^\]]+)\]|\(([^)]+)\))\s*$", name)
    return (match.group(1) or match.group(2)).strip() if match else None


def _inspect_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    if not sample:
        raise ValueError("CSV file is empty.")
    try:
        dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    warnings = ["CSV row count and geometry were not inferred from the header."]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, dialect)
        header = next(reader, [])
        if not header or any(not column.strip() for column in header):
            raise ValueError("CSV header is missing or contains an empty column name.")
        if len(set(header)) != len(header):
            warnings.append("CSV header contains duplicate column names.")
        try:
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    warnings.append(
                        f"Malformed CSV row {row_number}: expected {len(header)} columns, got {len(row)}."
                    )
                    break
                if row_number >= 101:
                    break
        except csv.Error as error:
            warnings.append(f"Malformed CSV data near row {reader.line_num}: {error}")
    fields = [
        {
            "name": column.strip(),
            "association": "row",
            "units": _unit_from_name(column.strip()),
        }
        for column in header
    ]
    coordinate_names = [
        re.sub(r"\s*\[[^\]]+\]|\s*\([^)]+\)", "", column).strip().lower()
        for column in header
    ]
    return {
        "fields": fields,
        "mesh_info": {
            "point_count": None,
            "cell_count": None,
            "geometry": "tabular rows; mesh topology not encoded",
        },
        "metadata": {"delimiter": dialect.delimiter, "header_columns": len(header)},
        "preprocessing_supported": (
            all(coordinate_names.count(axis) == 1 for axis in ("x", "y", "z"))
            and not any(warning.startswith("Malformed CSV") for warning in warnings)
        ),
        "warnings": warnings,
    }


def _inspect_vtk_xml(path: Path) -> dict[str, Any]:
    # Read only the bounded XML header. VTK payloads may be binary or multi-gigabyte.
    with path.open("rb") as stream:
        header = stream.read(MAX_METADATA_BYTES)
    if not header.lstrip().startswith(b"<?xml") and b"<VTKFile" not in header:
        raise ValueError("File does not contain a recognizable VTK XML header.")
    file_size = path.stat().st_size
    if file_size <= MAX_METADATA_BYTES and b'encoding="raw"' not in header:
        try:
            ElementTree.fromstring(header)
        except ElementTree.ParseError as error:
            raise ValueError(f"Malformed VTK XML: {error}") from error
    fields = []
    pieces = []
    association = None
    for match in VTK_XML_TAG.finditer(header):
        closing, tag, raw_attributes = match.groups()
        tag_name = tag.lower()
        if tag_name in {b"pointdata", b"celldata", b"fielddata", b"points", b"cells"}:
            if closing:
                association = None
            else:
                association = {
                    b"pointdata": "point",
                    b"celldata": "cell",
                    b"fielddata": "field",
                    b"points": "points",
                    b"cells": "cells",
                }[tag_name]
            continue
        attributes = _xml_attributes(raw_attributes)
        if tag_name == b"piece":
            pieces.append(attributes)
        elif attributes.get("Name"):
            fields.append(
                {
                    "name": attributes["Name"],
                    "association": association or "array",
                    "units": attributes.get("units") or attributes.get("Units"),
                }
            )
    return {
        "fields": fields,
        "mesh_info": {
            "point_count": sum(int(p["NumberOfPoints"]) for p in pieces if p.get("NumberOfPoints", "").isdigit())
            if pieces else None,
            "cell_count": sum(int(p["NumberOfCells"]) for p in pieces if p.get("NumberOfCells", "").isdigit())
            if pieces else None,
            "piece_count": len(pieces) or None,
            "geometry": path.suffix.lower()[1:],
        },
        "metadata": {"reader": "bounded VTK XML header inspection"},
        "warnings": (
            ["VTK metadata is limited to the first 1 MiB of the file."]
            if file_size > MAX_METADATA_BYTES
            else []
        ),
    }


def _inspect_legacy_vtk(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(MAX_METADATA_BYTES)
    text = header.decode("ascii", errors="replace")
    if not re.match(r"(?im)^#\s* vtk DataFile Version\s+", text):
        raise ValueError("Legacy VTK header is missing its version signature.")
    is_binary = bool(re.search(r"(?im)^BINARY\s*$", text))
    point_match = re.search(r"(?im)^POINTS\s+(\d+)\s+", text)
    cell_match = re.search(r"(?im)^CELLS\s+(\d+)\s+", text)
    fields = []
    for match in re.finditer(
        r"(?im)^(SCALARS|VECTORS|NORMALS|TENSORS|FIELD)\s+([^\s]+)",
        text,
    ):
        previous_association = list(
            re.finditer(r"(?im)^(POINT_DATA|CELL_DATA)\s+\d+", text[: match.start()])
        )
        association = (
            {
                "POINT_DATA": "point",
                "CELL_DATA": "cell",
            }[previous_association[-1].group(1)]
            if previous_association
            else "array"
        )
        fields.append(
            {"name": match.group(2), "association": association, "units": None}
        )
    return {
        "fields": fields,
        "mesh_info": {
            "point_count": int(point_match.group(1)) if point_match else None,
            "cell_count": int(cell_match.group(1)) if cell_match else None,
            "geometry": "legacy VTK",
        },
        "metadata": {
            "dataset_type": (re.search(r"(?im)^DATASET\s+(\S+)", text) or [None, None])[1],
            "encoding": "binary" if is_binary else "ascii or unspecified",
        },
        "warnings": (
            ["Legacy VTK binary payload is not decoded by the built-in inspector."]
            if is_binary
            else ["Legacy VTK inspection reads only the bounded file header."]
        ),
    }


def _infer_openfoam_case_metadata(case_path: Path) -> dict[str, Any]:
    metadata = {"reynolds_number": None, "mesh_resolution": None, "units": None}
    for part in reversed(case_path.parts):
        normalized = part.lower()
        if normalized.startswith("re") and normalized[2:].isdigit():
            metadata["reynolds_number"] = int(normalized[2:])
        if normalized in {"coarse", "medium", "fine"}:
            metadata["mesh_resolution"] = normalized
    return metadata


def _infer_case_id(relative_path: str) -> str:
    parts = Path(relative_path).parts
    for index, part in enumerate(parts):
        normalized = part.lower()
        if normalized in {"system", "constant", "polymesh", "0"}:
            return Path(*parts[:index]).as_posix() if index else "."
        if normalized in {"coarse", "medium", "fine"}:
            return Path(*parts[: index + 1]).as_posix()
        if normalized == "layerinfo":
            return Path(*parts[:index]).as_posix() if index else "."
        if normalized.startswith("processor"):
            return Path(*parts[:index]).as_posix() if index else "."
    if len(parts) > 1:
        return str(Path(*parts[:-1]))
    return "."


def _inspect_openfoam_directory(case_path: Path) -> dict[str, Any]:
    metadata = _infer_openfoam_case_metadata(case_path)
    from backend.services.cfd_analysis import CfdDataError, get_case_field_options

    try:
        field_options = get_case_field_options(case_path)
        fields = [
            {
                "name": field["name"],
                "region": field["region"],
                "association": field["association"],
                "units": field["units"],
                "field_class": field["field_class"],
                "dimensions": field["dimensions"],
            }
            for field in field_options["fields"]
        ]
        metadata["solution_time"] = field_options["time"]
        metadata["processor_count"] = field_options["processor_count"]
        warnings = [
            "OpenFOAM mesh point/cell counts are not computed; binary mesh topology is retained as source data."
        ]
    except CfdDataError as error:
        fields = []
        warnings = [str(error)]
    return {
        "fields": fields,
        "mesh_info": {
            "point_count": None,
            "cell_count": None,
            "geometry": "OpenFOAM case; mesh counts not parsed",
        },
        "metadata": {
            "reader": "OpenFOAM case and partition field-header inspection",
            **metadata,
        },
        "warnings": warnings,
    }


def _detect_format(path: Path) -> str:
    if path.name == ".gitkeep":
        return "workspace_placeholder"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv_point_table"
    if suffix == ".vtk":
        return "vtk_legacy"
    if suffix in VTK_XML_EXTENSIONS:
        return "vtk_xml"
    if suffix == ".foam":
        return "openfoam_case_marker"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    if path.name == "controlDict" and path.parent.name == "system":
        return "openfoam_case"
    if path.name in OPENFOAM_CASE_MARKERS or (
        path.parent.name in {"system", "constant", "polyMesh"}
        or any(part.lower().startswith("processor") for part in path.parts)
        or path.name in {"T", "U", "p", "phi", "alphaConv", "rho", "k", "epsilon", "omega"}
    ):
        return "openfoam_file"
    try:
        with path.open("rb") as stream:
            prefix = stream.read(512)
    except OSError:
        return "unreadable"
    if prefix.startswith(b"PK\x03\x04"):
        return "zip_archive"
    if b"# vtk DataFile" in prefix:
        return "vtk_legacy"
    if prefix.lstrip().startswith(b"<?xml") and b"VTKFile" in prefix:
        return "vtk_xml"
    return "unknown"


def _inspect_file(path: Path, detected_format: str) -> dict[str, Any]:
    if detected_format == "csv_point_table":
        return _inspect_csv(path)
    if detected_format == "vtk_xml":
        return _inspect_vtk_xml(path)
    if detected_format == "vtk_legacy":
        return _inspect_legacy_vtk(path)
    if detected_format == "openfoam_case":
        return _inspect_openfoam_directory(path.parent.parent)
    if detected_format == "workspace_placeholder":
        return {
            "fields": [],
            "mesh_info": None,
            "metadata": {},
            "warnings": [],
        }
    return {
        "fields": [],
        "mesh_info": None,
        "metadata": {},
        "warnings": ["No compatible metadata reader is enabled for this file format."],
    }


def _archive_members(path: Path) -> list[dict[str, Any]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                {"path": item.filename, "size_bytes": item.file_size}
                for item in archive.infolist()
                if not item.is_dir()
            ]
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            return [
                {"path": item.name, "size_bytes": item.size}
                for item in archive.getmembers()
                if item.isfile()
            ]
    raise ValueError("Archive format is not readable with Python's standard library.")


def _file_record(path: Path, relative_path: str, size: int, source_file: Path | None) -> dict[str, Any]:
    detected_format = _detect_format(path)
    case_id = _infer_case_id(relative_path)
    if case_id == ".":
        case_id = "."
    record: dict[str, Any] = {
        "path": relative_path.replace("\\", "/"),
        "size_bytes": size,
        "format": detected_format,
        "case_id": case_id,
        "fields": [],
        "mesh_info": None,
        "metadata": {},
        "warnings": [],
        "preprocessing_supported": False,
    }
    if source_file and detected_format in {
        "csv_point_table",
        "vtk_xml",
        "vtk_legacy",
        "openfoam_case",
        "workspace_placeholder",
    }:
        try:
            inspected = _inspect_file(path, detected_format)
            record.update(inspected)
            record["preprocessing_supported"] = inspected.get(
                "preprocessing_supported", False
            )
        except (OSError, UnicodeError, ValueError, csv.Error, ElementTree.ParseError) as error:
            record["warnings"] = [f"Metadata inspection failed: {error}"]
            record["inspection_error"] = True
    return record


def _case_records(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for file_record in files:
        if file_record["format"] not in RECOGNIZED_FORMATS:
            continue
        case_id = file_record["case_id"]
        case = cases.setdefault(
            case_id,
            {
                "case_id": case_id,
                "files": [],
                "fields": [],
                "mesh_info": None,
                "mesh_info_by_file": [],
                "_mesh_infos": [],
                "metadata": {
                    "reynolds_number": None,
                    "mesh_resolution": None,
                    "units": None,
                },
            },
        )
        case["files"].append(file_record["path"])
        for key, value in (file_record.get("metadata") or {}).items():
            if key in {
                "reynolds_number",
                "mesh_resolution",
                "units",
                "solution_time",
                "processor_count",
            } and value is not None:
                case["metadata"][key] = value
        for field in file_record["fields"]:
            if field not in case["fields"]:
                case["fields"].append(field)
        if file_record["format"] in {"vtk_legacy", "vtk_xml", "openfoam_case"} and file_record["mesh_info"] is not None:
            case["_mesh_infos"].append(
                {"file": file_record["path"], "mesh_info": file_record["mesh_info"]}
            )
    for case in cases.values():
        case["mesh_info_by_file"] = case.pop("_mesh_infos")
        if not case["mesh_info_by_file"]:
            case["mesh_info"] = {
                "point_count": None,
                "cell_count": None,
                "geometry": "OpenFOAM case; mesh counts not parsed",
            }
        elif len(case["mesh_info_by_file"]) == 1:
            case["mesh_info"] = case["mesh_info_by_file"][0]["mesh_info"]
        else:
            case["mesh_info"] = None
    return sorted(cases.values(), key=lambda case: case["case_id"])


def scan_dataset(source: Path) -> dict[str, Any]:
    source = source.expanduser()
    files: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not source.exists():
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "source_missing",
            "source_kind": "missing",
            "files": [],
            "cases": [],
            "field_names": [],
            "summary": {"file_count": 0, "candidate_count": 0, "total_bytes": 0},
            "warnings": ["Dataset source path does not exist."],
        }

    source_kind = "directory" if source.is_dir() else "file"
    archive_path = source if source.is_file() and (
        zipfile.is_zipfile(source)
        or tarfile.is_tarfile(source)
        or source.suffix.lower() in ARCHIVE_EXTENSIONS
    ) else None
    if archive_path:
        source_kind = "archive"
        try:
            for member in _archive_members(archive_path):
                member_path = Path(member["path"])
                files.append(
                    _file_record(
                        member_path,
                        member["path"],
                        member["size_bytes"],
                        None,
                    )
                )
            warnings.append(
                "Archive contents were listed only; no files were extracted or read."
            )
        except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
            warnings.append(f"Archive listing failed: {error}")
    elif source.is_dir():
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative_path = path.relative_to(source).as_posix()
            files.append(
                _file_record(path, relative_path, path.stat().st_size, path)
            )
        openfoam_case_ids = {
            record["case_id"]
            for record in files
            if record["format"] == "openfoam_case"
        }
        for record in list(files):
            relative_path = Path(record["path"])
            if relative_path.name != "controlDict":
                continue
            control_dict = source / relative_path
            case_root = control_dict.parent.parent
            relative_case = case_root.relative_to(source).as_posix()
            relative_case = relative_case if relative_case != "." else "."
            if relative_case not in openfoam_case_ids:
                result = _inspect_openfoam_directory(case_root)
                files.append(
                    {
                        "path": relative_path.as_posix(),
                        "size_bytes": record["size_bytes"],
                        "format": "openfoam_case",
                        "case_id": relative_case,
                        **result,
                        "preprocessing_supported": False,
                    }
                )
                openfoam_case_ids.add(relative_case)
    else:
        files.append(_file_record(source, source.name, source.stat().st_size, source))

    cases = _case_records(files)
    field_names = sorted(
        {field["name"] for case in cases for field in case["fields"]}
    )
    candidate_count = sum(
        file["format"] in {"csv_point_table", "vtk_legacy", "vtk_xml"}
        for file in files
    ) + len(
        {
            file["case_id"]
            for file in files
            if file["format"] in {"openfoam_case_marker", "openfoam_case"}
        }
    )
    total_bytes = sum(file["size_bytes"] for file in files)
    if source_kind == "archive":
        status = "archive_listed" if files else "archive_listing_failed"
    elif candidate_count:
        status = "scanned"
    else:
        status = "no_dataset_files"
        warnings.append("No supported CFD or tabular data files were found.")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_kind": source_kind,
        "files": files,
        "cases": cases,
        "field_names": field_names,
        "summary": {
            "file_count": len(files),
            "candidate_count": candidate_count,
            "total_bytes": total_bytes,
        },
        "warnings": warnings,
    }


def write_audit_report(manifest: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CFD Dataset Audit",
        "",
        f"- Scan status: `{manifest['status']}`",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Source kind: `{manifest['source_kind']}`",
        f"- Files listed: {manifest['summary']['file_count']}",
        f"- Candidate data files: {manifest['summary']['candidate_count']}",
        f"- Total listed bytes: {manifest['summary']['total_bytes']}",
        f"- Cases discovered: {len(manifest['cases'])}",
        "",
        "## Discovered cases",
        "",
    ]
    if manifest["cases"]:
        lines.extend(
            [
                "| Case ID | Files | Reynolds number | Mesh resolution | Solution time | Partitions | Fields | Points | Cells |",
                "|---|---:|---:|---|---:|---:|---|---:|---:|",
            ]
        )
        for case in manifest["cases"]:
            mesh = case["mesh_info"] or {}
            metadata = case["metadata"]
            reynolds_number = metadata.get("reynolds_number")
            mesh_resolution = metadata.get("mesh_resolution")
            fields = ", ".join(
                (
                    f"{field['region']}/" if field.get("region") else ""
                )
                + field["name"]
                + (f" [{field['units']}]" if field.get("units") else "")
                for field in case["fields"]
            ) or "Not detected"
            lines.append(
                f"| `{case['case_id']}` | {len(case['files'])} | "
                f"{reynolds_number if reynolds_number is not None else 'Unknown'} | "
                f"{mesh_resolution if mesh_resolution is not None else 'Unknown'} | "
                f"{metadata.get('solution_time', 'Unknown')} | "
                f"{metadata.get('processor_count', 'Unknown')} | {fields} | "
                f"{mesh.get('point_count', 'Unknown')} | {mesh.get('cell_count', 'Unknown')} |"
            )
    else:
        lines.append("No cases have been verified in the configured source.")
    lines.extend(
        [
            "",
            "## File inventory",
            "",
            "| Relative path | Format | Size (bytes) | Inspection |",
            "|---|---|---:|---|",
        ]
    )
    if manifest["files"]:
        for record in manifest["files"]:
            inspection = (
                "Warnings"
                if record.get("warnings")
                else "Metadata inspected"
                if record.get("fields") or record.get("mesh_info")
                else "Inventory only"
            )
            lines.append(
                f"| `{record['path']}` | `{record['format']}` | "
                f"{record['size_bytes']} | {inspection} |"
            )
    else:
        lines.append("| No files listed | — | 0 | No source files |")
    lines.extend(
        [
            "",
            "## Mesh metadata by file",
            "",
            "| Case ID | File | Points | Cells | Geometry |",
            "|---|---|---:|---:|---|",
        ]
    )
    mesh_rows = [
        (case["case_id"], mesh_record)
        for case in manifest["cases"]
        for mesh_record in case["mesh_info_by_file"]
    ]
    if mesh_rows:
        for case_id, mesh_record in mesh_rows:
            mesh_info = mesh_record["mesh_info"] or {}
            lines.append(
                f"| `{case_id}` | `{mesh_record['file']}` | "
                f"{mesh_info.get('point_count', 'Unknown')} | "
                f"{mesh_info.get('cell_count', 'Unknown')} | "
                f"{mesh_info.get('geometry', 'Unknown')} |"
            )
    else:
        lines.append("| No mesh metadata detected | — | Unknown | Unknown | Unknown |")
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- Reynolds numbers, mesh resolutions, and units are reported only when explicit source metadata is detected; otherwise they remain unknown.",
            "- Expected configurations (Re 560 and 11240; coarse, medium, and fine meshes) have not been verified against a dataset.",
            "- Archive entries are inventoried without extraction.",
            "- The built-in inspector reads bounded metadata; it does not decode large CFD field arrays.",
            "- Preprocessing currently supports only CSV point tables with explicit x, y, and z columns.",
            "- No train/test split is created. Grouped splitting requires verified case-family metadata and spatial block configuration.",
            "",
            "## Warnings",
            "",
        ]
    )
    all_warnings = list(manifest["warnings"])
    all_warnings.extend(
        f"{record['path']}: {warning}"
        for record in manifest["files"]
        for warning in record["warnings"]
    )
    lines.extend([f"- {warning}" for warning in all_warnings] or ["- None."])
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def write_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return scan_dataset(Path("__thermoexplain_unscanned_source__"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported dataset manifest schema.")
        return manifest
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Dataset manifest cannot be read: {error}") from error

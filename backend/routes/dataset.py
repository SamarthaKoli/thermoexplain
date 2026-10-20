import json
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from backend.config import Settings
from backend.services.cfd_analysis import (
    CfdDataError,
    calculate_field_statistics,
    compare_mesh_cases,
    get_case_field_options,
)
from backend.services.cfd_visualization import get_case_spatial_visualization
from backend.services.dataset_inspector import (
    MAX_SAMPLE_SIZE,
    load_manifest,
    scan_dataset,
    write_audit_report,
    write_manifest,
)
from backend.services.dataset_preprocessor import preprocess_csv


def create_dataset_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("dataset", __name__)

    def manifest() -> dict[str, Any]:
        return load_manifest(settings.manifest_path)

    def selected_case(case_id: str | None) -> tuple[dict[str, Any], Path]:
        if not case_id:
            raise CfdDataError("case_id query parameter is required.")
        case = next(
            (item for item in manifest()["cases"] if item["case_id"] == case_id),
            None,
        )
        if case is None:
            raise CfdDataError("Case was not found in the latest dataset scan.", 404)
        source_root = settings.dataset_path.expanduser().resolve()
        case_path = (source_root / Path(case_id)).resolve()
        try:
            case_path.relative_to(source_root)
        except ValueError as error:
            raise CfdDataError("Case path is outside the configured dataset.", 400) from error
        return case, case_path

    @blueprint.get("/scan")
    def scan_status():
        current = manifest()
        return jsonify(
            {
                "status": current["status"],
                "generated_at": current["generated_at"],
                "source_kind": current["source_kind"],
                "summary": current["summary"],
                "warnings": current["warnings"],
            }
        )

    @blueprint.post("/scan")
    def scan_now():
        try:
            current = scan_dataset(settings.dataset_path)
            write_manifest(current, settings.manifest_path)
            write_audit_report(current, settings.audit_report_path)
        except (OSError, ValueError) as error:
            current_app.logger.warning("Dataset scan could not be completed: %s", error)
            return jsonify({"error": "Dataset scan could not be completed."}), 500
        return jsonify(
            {
                "status": current["status"],
                "generated_at": current["generated_at"],
                "source_kind": current["source_kind"],
                "summary": current["summary"],
                "case_count": len(current["cases"]),
                "warnings": current["warnings"],
            }
        )

    @blueprint.get("/cases")
    def cases():
        return jsonify({"cases": manifest()["cases"]})

    @blueprint.get("/case-metadata")
    def case_metadata():
        case_id = request.args.get("case_id")
        if not case_id:
            return jsonify({"error": "case_id query parameter is required."}), 400
        case = next(
            (item for item in manifest()["cases"] if item["case_id"] == case_id),
            None,
        )
        if case is None:
            return jsonify({"error": "Case was not found in the latest scan."}), 404
        return jsonify(case)

    @blueprint.get("/fields")
    def fields():
        current = manifest()
        case_id = request.args.get("case_id")
        if case_id:
            try:
                _case, case_path = selected_case(case_id)
                options = get_case_field_options(case_path)
            except CfdDataError as error:
                return jsonify({"error": str(error)}), error.status_code
            return jsonify({"fields": options["fields"]})
        cases_to_check = current["cases"]
        field_map: dict[tuple[str, str | None], dict[str, Any]] = {}
        for case in cases_to_check:
            for field in case["fields"]:
                field_map[(field["name"], field.get("units"))] = field
        return jsonify({"fields": list(field_map.values())})

    @blueprint.get("/cfd-fields")
    def cfd_fields():
        try:
            _case, case_path = selected_case(request.args.get("case_id"))
            options = get_case_field_options(case_path)
        except CfdDataError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(options)

    @blueprint.get("/field-statistics")
    def field_statistics():
        try:
            case, case_path = selected_case(request.args.get("case_id"))
            region = request.args.get("region")
            field_name = request.args.get("field")
            if not region or not field_name:
                raise CfdDataError("region and field query parameters are required.")
            result = calculate_field_statistics(
                case_path,
                case_id=case["case_id"],
                reynolds_number=case["metadata"].get("reynolds_number"),
                mesh_resolution=case["metadata"].get("mesh_resolution"),
                region=region,
                field_name=field_name,
                component=request.args.get("component", "value"),
            )
        except CfdDataError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.get("/spatial-visualization")
    def spatial_visualization():
        try:
            case, case_path = selected_case(request.args.get("case_id"))
            region = request.args.get("region")
            field_name = request.args.get("field")
            component = request.args.get("component", "value")
            if not region or not field_name:
                raise CfdDataError("region and field query parameters are required.")
            result = get_case_spatial_visualization(
                case_path,
                case_id=case["case_id"],
                region=region,
                field_name=field_name,
                component=component,
            )
        except CfdDataError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.post("/mesh-comparison")
    def mesh_comparison():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        try:
            case, _case_path = selected_case(payload.get("case_id"))
            reference, _reference_path = selected_case(payload.get("reference_case_id"))
            result = compare_mesh_cases(
                case_id=case["case_id"],
                reference_case_id=reference["case_id"],
                case_metadata=case["metadata"],
                reference_metadata=reference["metadata"],
            )
        except CfdDataError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.get("/mesh-info")
    def mesh_info():
        case_id = request.args.get("case_id")
        if not case_id:
            return jsonify({"error": "case_id query parameter is required."}), 400
        case = next(
            (item for item in manifest()["cases"] if item["case_id"] == case_id),
            None,
        )
        if case is None:
            return jsonify({"error": "Case was not found in the latest scan."}), 404
        return jsonify(
            {
                "case_id": case_id,
                "mesh_info": case["mesh_info"],
                "mesh_info_by_file": case["mesh_info_by_file"],
            }
        )

    @blueprint.get("/preprocessing")
    def preprocessing_status():
        latest_path = settings.processed_path / "latest.json"
        if not latest_path.is_file():
            return jsonify({"status": "not_run", "message": "Preprocessing has not been run."})
        try:
            result = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            current_app.logger.warning("Preprocessing status is unreadable: %s", error)
            return jsonify({"error": "Preprocessing status is unreadable."}), 500
        return jsonify(result)

    @blueprint.get("/audit")
    def audit_report():
        try:
            report = settings.audit_report_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return jsonify({"status": "unavailable", "report": None})
        except OSError as error:
            current_app.logger.warning("Audit report cannot be read: %s", error)
            return jsonify({"error": "Audit report cannot be read."}), 500
        return jsonify({"status": "available", "report": report})

    @blueprint.post("/preprocess")
    def preprocess():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        relative_source = payload.get("file")
        if not isinstance(relative_source, str) or not relative_source.strip():
            return jsonify({"error": "A scanned CSV relative file path is required."}), 400
        try:
            scanned_files = {
                item["path"]: item
                for item in manifest()["files"]
                if item["preprocessing_supported"]
            }
        except (KeyError, TypeError) as error:
            current_app.logger.warning("Dataset manifest has an invalid file index: %s", error)
            return jsonify({"error": "Dataset manifest has an invalid file index."}), 500
        normalized_source = Path(relative_source).as_posix()
        if normalized_source not in scanned_files:
            return jsonify({"error": "File is not a supported CSV in the latest scan."}), 400
        source_root = settings.dataset_path.expanduser().resolve()
        source = (source_root / Path(normalized_source)).resolve()
        try:
            source.relative_to(source_root)
        except ValueError:
            return jsonify({"error": "Source path is outside the configured dataset."}), 400

        sample_size = payload.get("sample_size")
        seed = payload.get("seed", 42)
        if sample_size is not None and (
            not isinstance(sample_size, int)
            or isinstance(sample_size, bool)
            or sample_size < 1
            or sample_size > MAX_SAMPLE_SIZE
        ):
            return jsonify(
                {"error": f"sample_size must be between 1 and {MAX_SAMPLE_SIZE:,}."}
            ), 400
        if not isinstance(seed, int) or isinstance(seed, bool):
            return jsonify({"error": "seed must be an integer."}), 400
        try:
            result = preprocess_csv(
                source,
                settings.processed_path,
                sample_size=sample_size,
                seed=seed,
                case_id=scanned_files[normalized_source]["case_id"],
                source_id=normalized_source,
            )
        except (OSError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(result), 201

    return blueprint

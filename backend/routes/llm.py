import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from backend.config import Settings
from backend.services.cfd_analysis import CfdDataError, calculate_field_statistics
from backend.services.dataset_inspector import load_manifest
from backend.services.gpt_oss_service import (
    GptOssServiceError,
    SUPPORTED_GPT_OSS_MODELS,
    check_gpt_oss_service,
    interpret_feature_context,
    interpret_cfd_statistics,
    require_gpt_oss_configuration,
    test_gpt_oss_connection,
)


def create_llm_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("llm", __name__)

    @blueprint.get("/status")
    def status():
        model_configured = settings.gpt_oss_model in SUPPORTED_GPT_OSS_MODELS
        configured = bool(settings.gpt_oss_api_key and model_configured)
        if not settings.gpt_oss_api_key:
            return jsonify(
                {
                    "provider": "GPT-OSS",
                    "configured": False,
                    "model": settings.gpt_oss_model,
                    "status": "not_verified",
                    "diagnostic": "Set GPT_OSS_API_KEY in the backend environment to verify service availability.",
                }
            )
        try:
            result = check_gpt_oss_service(settings)
            result["configured"] = configured
            return jsonify(result)
        except GptOssServiceError as error:
            current_app.logger.warning(
                "GPT-OSS availability check failed: status=%s diagnostic=%s",
                error.status_code,
                error.diagnostic,
            )
            return jsonify(
                {
                    "provider": "GPT-OSS",
                    "configured": configured,
                    "model": settings.gpt_oss_model,
                    "status": "unavailable",
                    "diagnostic": error.diagnostic,
                }
            )

    @blueprint.post("/connection-test")
    def connection_test():
        try:
            result = test_gpt_oss_connection(settings)
        except GptOssServiceError as error:
            current_app.logger.warning(
                "GPT-OSS inference probe failed: status=%s diagnostic=%s",
                error.status_code,
                error.diagnostic,
            )
            return jsonify(error.to_payload()), error.status_code
        return jsonify(result)

    @blueprint.post("/analysis")
    def analysis():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        required = ("case_id", "region", "field")
        if any(not isinstance(payload.get(key), str) or not payload[key].strip() for key in required):
            return jsonify({"error": "case_id, region, and field are required."}), 400
        case_id = payload["case_id"]
        try:
            require_gpt_oss_configuration(settings)
            case = next(
                (
                    item
                    for item in load_manifest(settings.manifest_path)["cases"]
                    if item["case_id"] == case_id
                ),
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
            statistics = calculate_field_statistics(
                case_path,
                case_id=case_id,
                reynolds_number=case["metadata"].get("reynolds_number"),
                mesh_resolution=case["metadata"].get("mesh_resolution"),
                region=payload["region"],
                field_name=payload["field"],
                component=payload.get("component", "value"),
            )
            interpretation = interpret_cfd_statistics(settings, statistics)
        except CfdDataError as error:
            return jsonify({"error": str(error)}), error.status_code
        except GptOssServiceError as error:
            current_app.logger.warning(
                "GPT-OSS CFD interpretation failed: status=%s diagnostic=%s",
                error.status_code,
                error.diagnostic,
            )
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(
            {
                "statistics": statistics,
                "interpretation": interpretation,
            }
        )

    @blueprint.post("/interpretation")
    def interpretation():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        feature = payload.get("feature")
        context = payload.get("context")
        if (
            not isinstance(feature, str)
            or feature not in {"generative", "validation"}
            or not isinstance(context, dict)
            or not context
        ):
            return jsonify(
                {"error": "A supported feature and non-empty evidence context are required."}
            ), 400
        try:
            serialized_context = json.dumps(context, allow_nan=False)
        except (TypeError, ValueError):
            return jsonify({"error": "Evidence context must contain valid JSON values."}), 400
        if len(serialized_context) > 12_000:
            return jsonify({"error": "Evidence context exceeds the 12,000-character limit."}), 400
        try:
            interpretation_result = interpret_feature_context(settings, feature, context)
        except GptOssServiceError as error:
            current_app.logger.warning(
                "GPT-OSS %s interpretation failed: status=%s diagnostic=%s",
                feature,
                error.status_code,
                error.diagnostic,
            )
            return jsonify(error.to_payload()), error.status_code
        return jsonify(interpretation_result)

    return blueprint

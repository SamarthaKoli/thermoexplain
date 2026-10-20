"""Model discovery, verified prediction inputs, explanations, and GPT-OSS endpoints."""

import math
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from backend.config import Settings
from backend.ml.surrogate import (
    SurrogateError,
    explain_lime_local,
    explain_shap_local,
    list_saved_models,
    load_model_bundle,
    manifest_digest,
    predict_one,
    validate_feature_matrix,
)
from backend.services.dataset_inspector import load_manifest
from backend.services.gpt_oss_service import (
    GptOssServiceError,
    interpret_surrogate_explanation,
    require_gpt_oss_configuration,
)


class MlRequestError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _model(settings: Settings, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle, metadata = load_model_bundle(settings.models_path, model_id)
    if manifest_digest(settings.manifest_path) != metadata["dataset"]["manifest_sha256"]:
        raise MlRequestError(
            "The scanned dataset manifest changed after training; retrain this model before using it.",
            409,
        )
    return bundle, metadata


def _prediction_input(
    settings: Settings, payload: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[float]]:
    if not isinstance(payload, dict):
        raise MlRequestError("A JSON request body is required.")
    model_id = payload.get("model_id")
    case_id = payload.get("case_id")
    coordinates = payload.get("coordinates")
    if not isinstance(model_id, str) or not model_id.strip():
        raise MlRequestError("model_id is required.")
    if not isinstance(case_id, str) or not case_id.strip():
        raise MlRequestError("case_id is required.")
    if not isinstance(coordinates, dict) or any(
        key not in coordinates for key in ("x", "y", "z")
    ):
        raise MlRequestError("Finite x, y, and z coordinates are required.")
    numeric_coordinates = []
    for key in ("x", "y", "z"):
        value = coordinates[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MlRequestError(f"Coordinate {key} must be a numeric value.")
        if not math.isfinite(value):
            raise MlRequestError(f"Coordinate {key} must be finite.")
        numeric_coordinates.append(float(value))

    bundle, model_metadata = _model(settings, model_id)
    saved_case = next(
        (item for item in model_metadata["cases"] if item["case_id"] == case_id),
        None,
    )
    if saved_case is None:
        raise MlRequestError(
            "The selected case was not part of this model's verified training dataset."
        )
    manifest_case = next(
        (
            item
            for item in load_manifest(settings.manifest_path)["cases"]
            if item["case_id"] == case_id
        ),
        None,
    )
    if manifest_case is None:
        raise MlRequestError("The selected case is no longer in the dataset scan.", 404)
    source_root = settings.dataset_path.expanduser().resolve()
    case_path = (source_root / Path(case_id)).resolve()
    try:
        case_path.relative_to(source_root)
    except ValueError as error:
        raise MlRequestError(
            "The selected case is outside the configured dataset.", 400
        ) from error
    if not case_path.is_dir():
        raise MlRequestError("The selected CFD case directory is unavailable.", 404)

    bounds = saved_case["bounds"]
    for index, value in enumerate(numeric_coordinates):
        minimum, maximum = bounds[index]
        if value < minimum or value > maximum:
            raise MlRequestError(
                f"Coordinate {('x', 'y', 'z')[index]} is outside the selected case's verified cell-center bounds."
            )
    reynolds_number = manifest_case["metadata"].get("reynolds_number")
    if reynolds_number != saved_case["reynolds_number"]:
        raise MlRequestError(
            "The selected case metadata changed after model training; retrain before predicting.",
            409,
        )
    features = validate_feature_matrix(
        [*numeric_coordinates, float(reynolds_number)]
    )[0]
    return bundle, model_metadata, saved_case, features.tolist()


def create_ml_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("ml", __name__)

    @blueprint.get("/models")
    def models():
        try:
            current_digest = manifest_digest(settings.manifest_path)
            available_models = list_saved_models(settings.models_path)
        except (OSError, SurrogateError) as error:
            current_app.logger.warning("Saved model inventory is unavailable.")
            return jsonify({"error": str(error)}), 500
        for model in available_models:
            model["available"] = (
                model.get("dataset", {}).get("manifest_sha256") == current_digest
            )
            if not model["available"]:
                model["unavailable_reason"] = (
                    "The dataset manifest changed after this model was trained."
                )
        return jsonify(
            {
                "models": available_models,
                "status": "available" if available_models else "unavailable",
                "empty_reason": (
                    None
                    if available_models
                    else "No trained surrogate is saved. Run python -m backend.scripts.train_surrogate after verifying the dataset."
                ),
            }
        )

    @blueprint.post("/predict")
    def predict():
        try:
            bundle, model_metadata, selected_case, features = _prediction_input(
                settings, request.get_json(silent=True)
            )
            value = predict_one(bundle, features)
        except (MlRequestError, SurrogateError) as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(
            {
                "model_id": model_metadata["model_id"],
                "case_id": selected_case["case_id"],
                "target": model_metadata["target"],
                "coordinates": {
                    "x": features[0],
                    "y": features[1],
                    "z": features[2],
                    "units": model_metadata["coordinate_units"],
                },
                "reynolds_number": selected_case["reynolds_number"],
                "prediction": value,
                "units": model_metadata["target"]["units"],
                "status": "exploratory",
                "limitations": model_metadata["limitations"],
            }
        )

    def local_explanation(method: str):
        bundle, model_metadata, selected_case, features = _prediction_input(
            settings, request.get_json(silent=True)
        )
        if method == "shap":
            explanation = explain_shap_local(bundle, features)
        else:
            explanation = explain_lime_local(bundle, features)
        return bundle, model_metadata, selected_case, features, explanation

    @blueprint.get("/shap/global")
    def shap_global():
        model_id = request.args.get("model_id", "")
        try:
            _bundle, model_metadata = _model(settings, model_id)
            global_data = model_metadata["explainability"]["shap_global"]
        except (MlRequestError, SurrogateError, KeyError) as error:
            return jsonify({"error": str(error)}), getattr(error, "status_code", 400)
        return jsonify(
            {
                "model_id": model_id,
                "target": model_metadata["target"],
                "baseline_definition": global_data["baseline_definition"],
                "sample_count": global_data["sample_count"],
                "additivity_checked": global_data["additivity_checked"],
                "features": global_data["features"],
            }
        )

    @blueprint.post("/shap/local")
    def shap_local():
        try:
            _bundle, model_metadata, selected_case, _features, explanation = (
                local_explanation("shap")
            )
        except (MlRequestError, SurrogateError) as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(
            {
                "model_id": model_metadata["model_id"],
                "case_id": selected_case["case_id"],
                "target": model_metadata["target"],
                "explanation": explanation,
            }
        )

    @blueprint.post("/lime/local")
    def lime_local():
        try:
            _bundle, model_metadata, selected_case, _features, explanation = (
                local_explanation("lime")
            )
        except (MlRequestError, SurrogateError) as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(
            {
                "model_id": model_metadata["model_id"],
                "case_id": selected_case["case_id"],
                "target": model_metadata["target"],
                "explanation": explanation,
            }
        )

    @blueprint.post("/gpt-oss/interpretation")
    def gpt_oss_interpretation():
        payload = request.get_json(silent=True)
        method = payload.get("method") if isinstance(payload, dict) else None
        if method not in {"shap", "lime"}:
            return jsonify({"error": "method must be either 'shap' or 'lime'."}), 400
        try:
            require_gpt_oss_configuration(settings)
            _bundle, model_metadata, selected_case, features, explanation = (
                local_explanation(method)
            )
            context = {
                "target": model_metadata["target"],
                "model_name": model_metadata["model_name"],
                "case": {
                    "case_id": selected_case["case_id"],
                    "reynolds_number": selected_case["reynolds_number"],
                    "mesh_resolution": selected_case["mesh_resolution"],
                    "solution_time": selected_case["solution_time"],
                },
                "features": {
                    name: float(features[index])
                    for index, name in enumerate(("x", "y", "z", "reynolds_number"))
                },
                "validation_metrics": model_metadata["metrics"],
                "validation_scope": model_metadata["split"]["validation_kind"],
                "explanation_method": method,
                "computed_explanation": explanation,
            }
            interpretation = interpret_surrogate_explanation(settings, context)
        except (MlRequestError, SurrogateError, GptOssServiceError) as error:
            if isinstance(error, GptOssServiceError):
                current_app.logger.warning(
                    "GPT-OSS interpretation failed: status=%s diagnostic=%s",
                    error.status_code,
                    error.diagnostic,
                )
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(
            {
                "explanation_method": method,
                "interpretation": interpretation,
            }
        )

    return blueprint

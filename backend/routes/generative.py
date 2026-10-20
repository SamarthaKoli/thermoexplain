"""VAE reconstruction and GAN field-generation experiment endpoints."""

from typing import Any

from flask import Blueprint, current_app, jsonify, request

from backend.config import Settings
from backend.ml.generative import (
    DEFAULT_EPOCHS,
    GenerativeError,
    generate_synthetic_sample,
    list_generative_models,
    load_model_metadata,
    reconstruct_validation_sample,
    require_torch,
    torch_runtime_status,
    train_generative_model,
    validate_training_config,
)
from backend.services.dataset_inspector import load_manifest
from backend.services.generative_data import (
    GenerativeDataError,
    prepare_case_patches,
    prepared_cache_path,
)
from backend.ml.surrogate import SurrogateError, manifest_digest


def _dataset(settings: Settings) -> tuple[dict[str, Any], str]:
    try:
        manifest = load_manifest(settings.manifest_path)
        current_digest = manifest_digest(settings.manifest_path)
    except (OSError, ValueError, SurrogateError) as error:
        raise GenerativeError("The scanned dataset manifest is unavailable or invalid.", 503) from error
    if manifest.get("status") != "scanned":
        raise GenerativeError("Scan the real dataset before using the generative experiment.", 503)
    return manifest, current_digest


def _selected_case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    case = next(
        (item for item in manifest.get("cases", []) if item.get("case_id") == case_id),
        None,
    )
    if case is None:
        raise GenerativeError("The selected case is not present in the current dataset manifest.")
    metadata = case.get("metadata", {})
    fields = case.get("fields", [])
    if not isinstance(metadata, dict) or not isinstance(fields, list):
        raise GenerativeError("The selected case has malformed scanned metadata.")
    if not any(
        field.get("region") == "coolant"
        and field.get("name") == "T"
        and field.get("association") == "cell"
        and field.get("field_class") == "volScalarField"
        and field.get("units") == "K"
        for field in fields
    ):
        raise GenerativeError("The selected case does not contain the verified coolant/T field in kelvin.")
    if not isinstance(metadata.get("reynolds_number"), int):
        raise GenerativeError("The selected case has no verified Reynolds-number metadata.")
    return case


def _prepared_for_model(
    settings: Settings, model_id: str, manifest: dict[str, Any], current_digest: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = load_model_metadata(settings.models_path, model_id)
    source = metadata["source"]
    if not isinstance(source.get("case_id"), str):
        raise GenerativeError("The saved model is missing verified case metadata.", 409)
    if source.get("manifest_sha256") != current_digest:
        raise GenerativeError("The dataset manifest changed after training; retrain this model.", 409)
    case_id = source.get("case_id")
    _selected_case(manifest, case_id)
    cache_path = prepared_cache_path(
        settings.models_path / "generative" / "prepared", case_id, current_digest
    )
    if not cache_path.is_file():
        try:
            dataset, _digest = prepare_case_patches(
                settings.dataset_path,
                settings.manifest_path,
                settings.models_path / "generative" / "prepared",
                case_id=case_id,
            )
        except GenerativeDataError as error:
            raise GenerativeError(str(error), 503) from error
    else:
        from backend.services.generative_data import load_prepared_dataset

        try:
            dataset = load_prepared_dataset(cache_path)
        except GenerativeDataError as error:
            raise GenerativeError(str(error), 500) from error
    if dataset.get("manifest_sha256") != current_digest:
        raise GenerativeError("Prepared patches do not match the current dataset manifest.", 409)
    return dataset, metadata


def create_generative_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("generative", __name__)

    @blueprint.get("/status")
    def status():
        torch_runtime = torch_runtime_status()
        try:
            manifest, digest = _dataset(settings)
            eligible_cases = []
            for case in manifest.get("cases", []):
                try:
                    _selected_case(manifest, case.get("case_id"))
                except GenerativeError:
                    continue
                metadata = case["metadata"]
                eligible_cases.append(
                    {
                        "case_id": case["case_id"],
                        "reynolds_number": metadata["reynolds_number"],
                        "mesh_resolution": metadata.get("mesh_resolution"),
                        "solution_time": metadata.get("solution_time"),
                        "target": {
                            "region": "coolant",
                            "field": "T",
                            "units": "K",
                            "field_class": "volScalarField",
                        },
                    }
                )
            models = list_generative_models(settings.models_path, digest)
        except GenerativeError as error:
            return jsonify(
                {
                    "dataset_status": "unavailable",
                    "error": str(error),
                    "torch_available": torch_runtime["available"],
                    "torch_runtime": torch_runtime,
                    "models": [],
                    "cases": [],
                }
            ), error.status_code
        return jsonify(
            {
                "dataset_status": "available" if eligible_cases else "unavailable",
                "cases": eligible_cases,
                "models": models,
                "torch_available": torch_runtime["available"],
                "torch_runtime": torch_runtime,
                "training_device": "CPU (bounded pilot)",
                "manifest_sha256": digest,
                "status": "available" if eligible_cases else "unavailable",
                "empty_reason": None if eligible_cases else "No scanned case has the verified coolant/T scalar field in kelvin.",
            }
        )

    @blueprint.post("/train")
    def train():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        model_type = payload.get("model_type")
        case_id = payload.get("case_id")
        seed = payload.get("seed", 42)
        epochs = payload.get("epochs", DEFAULT_EPOCHS)
        if not isinstance(case_id, str) or not case_id:
            return jsonify({"error": "case_id is required."}), 400
        try:
            model_type, seed, epochs = validate_training_config(model_type, seed, epochs)
            require_torch()
            manifest, _digest = _dataset(settings)
            _selected_case(manifest, case_id)
            dataset, _digest = prepare_case_patches(
                settings.dataset_path,
                settings.manifest_path,
                settings.models_path / "generative" / "prepared",
                case_id=case_id,
            )
            metadata = train_generative_model(
                dataset,
                model_type=model_type,
                seed=seed,
                epochs=epochs,
                models_root=settings.models_path,
            )
        except (GenerativeError, GenerativeDataError) as error:
            current_app.logger.warning("Generative training was not completed: %s", error)
            return jsonify({"error": str(error)}), getattr(error, "status_code", 400)
        metadata["available"] = True
        return jsonify({"model": metadata}), 201

    @blueprint.get("/models/<model_id>/samples")
    def samples(model_id: str):
        try:
            manifest, digest = _dataset(settings)
            dataset, metadata = _prepared_for_model(settings, model_id, manifest, digest)
        except (GenerativeError, GenerativeDataError) as error:
            return jsonify({"error": str(error)}), getattr(error, "status_code", 400)
        return jsonify(
            {
                "model_id": model_id,
                "model_type": metadata["model_type"],
                "samples": [sample for sample in dataset["samples"] if sample["split"] == "validation"],
            }
        )

    def selected_sample_operation(operation: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        model_id = payload.get("model_id")
        sample_id = payload.get("sample_id")
        if not isinstance(model_id, str) or not model_id:
            return jsonify({"error": "model_id is required."}), 400
        if not isinstance(sample_id, str) or not sample_id:
            return jsonify({"error": "sample_id is required."}), 400
        try:
            manifest, digest = _dataset(settings)
            dataset, metadata = _prepared_for_model(settings, model_id, manifest, digest)
            if operation == "reconstruct":
                if metadata.get("model_type") != "vae":
                    raise GenerativeError("Select a VAE model to reconstruct a held-out patch.")
                result = reconstruct_validation_sample(
                    dataset,
                    models_root=settings.models_path,
                    model_id=model_id,
                    sample_id=sample_id,
                    manifest_sha256=digest,
                )
            else:
                if metadata.get("model_type") != "gan":
                    raise GenerativeError("Select a GAN model to generate a synthetic field patch.")
                seed = payload.get("seed", 42)
                result = generate_synthetic_sample(
                    dataset,
                    models_root=settings.models_path,
                    model_id=model_id,
                    sample_id=sample_id,
                    manifest_sha256=digest,
                    seed=seed,
                )
        except (GenerativeError, GenerativeDataError) as error:
            return jsonify({"error": str(error)}), getattr(error, "status_code", 400)
        return jsonify(result)

    @blueprint.post("/reconstruct")
    def reconstruct():
        return selected_sample_operation("reconstruct")

    @blueprint.post("/generate")
    def generate():
        return selected_sample_operation("generate")

    return blueprint

"""Aggregate recorded validation evidence without inferring scientific success."""

from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, current_app, jsonify

from backend.config import Settings
from backend.ml.generative import (
    GenerativeError,
    list_generative_models,
    torch_runtime_status,
)
from backend.ml.surrogate import (
    SurrogateError,
    list_saved_models,
    manifest_digest,
)
from backend.services.dataset_inspector import load_manifest
from backend.services.knowledge_corpus import KnowledgeCorpus, KnowledgeError

VALIDATION_STATUSES = (
    "passed",
    "failed",
    "partial",
    "not_evaluated",
    "unavailable",
    "not_reported",
)


def _module(
    module_id: str,
    title: str,
    status: str,
    *,
    metrics: dict[str, Any] | None = None,
    recorded_at: str | None = None,
    limitations: list[str] | None = None,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    return {
        "id": module_id,
        "title": title,
        "status": status,
        "recorded_at": recorded_at,
        "metrics": metrics or {},
        "limitations": limitations or [],
        "diagnostic": diagnostic,
    }


def _dataset_summary(settings: Settings) -> tuple[dict[str, Any], str | None]:
    try:
        manifest = load_manifest(settings.manifest_path)
        digest = manifest_digest(settings.manifest_path)
    except (OSError, ValueError, SurrogateError) as error:
        current_app.logger.warning("Validation dataset manifest is unavailable.")
        return (
            {
                "status": "unavailable",
                "generated_at": None,
                "case_count": None,
                "warnings": [],
                "diagnostic": "The dataset manifest could not be read.",
            },
            None,
        )
    cases = manifest.get("cases", [])
    return (
        {
            "status": manifest.get("status", "not_reported"),
            "generated_at": manifest.get("generated_at"),
            "case_count": len(cases) if isinstance(cases, list) else 0,
            "warnings": manifest.get("warnings", []),
            "diagnostic": None,
        },
        digest,
    )


def _surrogate_modules(settings: Settings, digest: str | None) -> list[dict[str, Any]]:
    if digest is None:
        diagnostic = "A readable scanned dataset manifest is required."
        return [
            _module(
                "surrogate",
                "Surrogate model evaluation",
                "unavailable",
                diagnostic=diagnostic,
            ),
            _module(
                "interpretability",
                "SHAP and LIME interpretability",
                "unavailable",
                diagnostic=diagnostic,
            ),
        ]
    try:
        models = list_saved_models(settings.models_path)
    except (OSError, SurrogateError) as error:
        current_app.logger.warning("Validation surrogate inventory is unavailable.")
        diagnostic = "The saved surrogate model inventory could not be read."
        return [
            _module(
                "surrogate",
                "Surrogate model evaluation",
                "unavailable",
                diagnostic=diagnostic,
            ),
            _module(
                "interpretability",
                "SHAP and LIME interpretability",
                "unavailable",
                diagnostic=diagnostic,
            ),
        ]
    model = next(
        (
            item
            for item in models
            if isinstance(item.get("dataset"), dict)
            and item["dataset"].get("manifest_sha256") == digest
        ),
        None,
    )
    if model is None:
        reason = (
            "No saved surrogate model matches the current dataset manifest."
            if models
            else "No saved surrogate model is available."
        )
        status = "not_evaluated" if models else "unavailable"
        return [
            _module("surrogate", "Surrogate model evaluation", status, diagnostic=reason),
            _module(
                "interpretability",
                "SHAP and LIME interpretability",
                status,
                diagnostic=reason,
            ),
        ]

    recorded_metrics = model.get("metrics", {})
    validation_metrics = (
        recorded_metrics.get("extra_trees_spatial_validation")
        if isinstance(recorded_metrics, dict)
        else None
    )
    limitations = model.get("limitations", [])
    if not isinstance(limitations, list):
        limitations = []
    if isinstance(validation_metrics, dict):
        surrogate_status = "partial"
        surrogate_diagnostic = (
            "Spatial-holdout metrics are recorded; independent-case performance has not been established."
        )
    else:
        surrogate_status = "not_reported"
        surrogate_diagnostic = "No spatial-holdout metric record is present in the saved model metadata."

    explainability = model.get("explainability", {})
    shap = explainability.get("shap_global") if isinstance(explainability, dict) else None
    if isinstance(shap, dict) and isinstance(shap.get("features"), list):
        interpretability_status = "partial"
        interpretability_metrics = {
            "shap_global": shap,
            "lime": explainability.get("lime", {}),
        }
        interpretability_diagnostic = (
            "Recorded global SHAP metadata is available; local LIME quality and SHAP/LIME agreement are not reported."
        )
    else:
        interpretability_status = "not_evaluated"
        interpretability_metrics = {}
        interpretability_diagnostic = "No global SHAP result is recorded."

    return [
        _module(
            "surrogate",
            "Surrogate model evaluation",
            surrogate_status,
            metrics={
                "spatial_holdout": validation_metrics,
                "baseline_mean_predictor": (
                    recorded_metrics.get("baseline_mean_predictor", {})
                    if isinstance(recorded_metrics, dict)
                    else {}
                ),
            },
            recorded_at=model.get("created_at"),
            limitations=limitations,
            diagnostic=surrogate_diagnostic,
        ),
        _module(
            "interpretability",
            "SHAP and LIME interpretability",
            interpretability_status,
            metrics=interpretability_metrics,
            recorded_at=model.get("created_at"),
            limitations=[
                "Feature attribution describes model behavior, not physical causation.",
                "No LIME quality score or SHAP/LIME agreement result is reported.",
            ],
            diagnostic=interpretability_diagnostic,
        ),
    ]


def _generative_module(settings: Settings, digest: str | None) -> dict[str, Any]:
    runtime = torch_runtime_status()
    if digest is None:
        return _module(
            "generative",
            "Generative model fidelity and physical constraints",
            "unavailable",
            metrics={"torch_runtime": runtime},
            diagnostic="A readable scanned dataset manifest is required.",
        )
    try:
        models = list_generative_models(settings.models_path, digest)
    except (OSError, GenerativeError):
        current_app.logger.warning("Validation generative model inventory is unavailable.")
        return _module(
            "generative",
            "Generative model fidelity and physical constraints",
            "unavailable",
            metrics={"torch_runtime": runtime},
            diagnostic="The saved generative model inventory could not be read.",
        )
    compatible_models = [item for item in models if item.get("available")]
    if not compatible_models:
        return _module(
            "generative",
            "Generative model fidelity and physical constraints",
            "unavailable" if not runtime["available"] else "not_evaluated",
            metrics={"torch_runtime": runtime, "saved_model_count": len(models)},
            limitations=(
                ["PyTorch runtime is unavailable."]
                if not runtime["available"]
                else []
            ),
            diagnostic=(
                runtime["diagnostic"]
                if not runtime["available"]
                else "No saved generative model matches the current dataset manifest."
            ),
        )
    limitations = [
        "Recorded statistical metrics and positivity checks do not establish physical validity.",
        "GAN outputs are synthetic and are not CFD solutions.",
    ]
    if not runtime["available"]:
        limitations.append(runtime["diagnostic"])
    return _module(
        "generative",
        "Generative model fidelity and physical constraints",
        "partial",
        metrics={
            "torch_runtime": runtime,
            "models": [
                {
                    "model_id": item.get("model_id"),
                    "model_type": item.get("model_type"),
                    "created_at": item.get("created_at"),
                    "recorded_metrics": item.get("metrics", {}),
                }
                for item in compatible_models
            ],
        },
        recorded_at=max(
            (
                item["created_at"]
                for item in compatible_models
                if isinstance(item.get("created_at"), str)
            ),
            default=None,
        ),
        limitations=limitations,
        diagnostic=(
            "Recorded pilot metrics are available; physical constraints and engineering validity were not established."
        ),
    )


def _knowledge_module(settings: Settings) -> dict[str, Any]:
    try:
        corpus = KnowledgeCorpus(
            settings.knowledge_documents_path, settings.knowledge_index_path
        )
        corpus_status = corpus.status()
        retrieval = corpus.retrieval_status()
    except (OSError, ValueError, KnowledgeError):
        current_app.logger.warning("Validation knowledge corpus status is unavailable.")
        return _module(
            "knowledge_grounding",
            "NLP intent and citation grounding",
            "unavailable",
            diagnostic="The configured knowledge corpus could not be inspected.",
        )
    if retrieval["status"] == "unavailable":
        status = "unavailable"
    else:
        status = "not_evaluated"
    return _module(
        "knowledge_grounding",
        "NLP intent and citation grounding",
        status,
        metrics={
            "source_document_count": corpus_status["source_document_count"],
            "indexed_document_count": corpus_status["document_count"],
            "passage_count": corpus_status["passage_count"],
            "retrieval_status": retrieval["status"],
        },
        recorded_at=corpus_status["indexed_at"],
        limitations=[
            "No citation-grounding or intent-classification benchmark result is recorded."
        ],
        diagnostic=(
            retrieval["diagnostic"]
            if status == "unavailable"
            else "The index is available, but answer quality and citation grounding have not been evaluated."
        ),
    )


def _status_counts(modules: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in VALIDATION_STATUSES}
    for module in modules:
        counts[module["status"]] += 1
    return counts


def create_validation_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("validation", __name__)

    @blueprint.get("/status")
    def status():
        dataset, digest = _dataset_summary(settings)
        modules = [
            _module(
                "mesh_independence",
                "CFD numerical mesh independence",
                "not_evaluated",
                limitations=[
                    "No mesh-convergence, GCI, or cross-mesh field-evaluation result is recorded."
                ],
                diagnostic="The available dataset inventory is not a numerical convergence evaluation.",
            ),
            *_surrogate_modules(settings, digest),
            _generative_module(settings, digest),
            _knowledge_module(settings),
        ]
        counts = _status_counts(modules)
        evidence_bearing = sum(
            counts[status] for status in ("passed", "failed", "partial")
        )
        configured_model = settings.gpt_oss_model in {
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        }
        provider_configured = bool(settings.gpt_oss_api_key and configured_model)
        limitations = [
            "This endpoint consolidates recorded module evidence; it does not run a full engineering validation suite.",
            "No physical experiment, independent-case validation, safety assessment, or regulatory compliance result is established.",
            "GPT-OSS configuration or reachability is operational information and is not counted as validation evidence.",
        ]
        return jsonify(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "full_suite_status": "not_evaluated",
                    "module_count": len(modules),
                    "modules_with_recorded_evidence": evidence_bearing,
                    "status_counts": counts,
                    "limitations": limitations,
                },
                "dataset": dataset,
                "modules": modules,
                "provider": {
                    "provider": "GPT-OSS",
                    "model": settings.gpt_oss_model if configured_model else None,
                    "configured": provider_configured,
                    "status": "configured" if provider_configured else "not_configured",
                    "counts_as_validation": False,
                },
            }
        )

    return blueprint

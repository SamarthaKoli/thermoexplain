import json
import hashlib

from backend.app import create_app
from backend.config import Settings


def make_client(tmp_path, *, api_key=None):
    documents_path = tmp_path / "knowledge"
    documents_path.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "scanned",
                "generated_at": "2026-10-20T00:00:00Z",
                "cases": [{"case_id": "case-1"}],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        manifest_path=manifest_path,
        models_path=tmp_path / "models",
        knowledge_documents_path=documents_path,
        knowledge_index_path=tmp_path / "processed" / "knowledge-index.json",
        gpt_oss_api_key=api_key,
        gpt_oss_model="openai/gpt-oss-120b",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    return settings, app.test_client()


def test_validation_status_reports_unassessed_suite_and_missing_evidence(tmp_path):
    _settings, client = make_client(tmp_path)

    response = client.get("/api/validation/status")

    assert response.status_code == 200
    result = response.get_json()
    assert result["summary"]["full_suite_status"] == "not_evaluated"
    assert result["summary"]["module_count"] == 5
    assert result["summary"]["modules_with_recorded_evidence"] == 0
    modules = {module["id"]: module for module in result["modules"]}
    assert modules["mesh_independence"]["status"] == "not_evaluated"
    assert modules["knowledge_grounding"]["status"] == "unavailable"
    assert modules["knowledge_grounding"]["metrics"]["source_document_count"] == 0
    assert result["dataset"]["case_count"] == 1


def test_validation_status_reports_real_surrogate_metrics_and_keeps_provider_separate(
    tmp_path,
):
    settings, client = make_client(tmp_path, api_key="server-only-test-key")
    manifest_digest = hashlib.sha256(settings.manifest_path.read_bytes()).hexdigest()
    model_id = "qcf_battery_coolant_t_spatial_test"
    (settings.models_path).mkdir()
    (settings.models_path / f"{model_id}.joblib").write_bytes(b"metadata inventory fixture")
    (settings.models_path / f"{model_id}.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "created_at": "2026-10-19T00:00:00Z",
                "dataset": {"manifest_sha256": manifest_digest},
                "metrics": {
                    "extra_trees_spatial_validation": {
                        "mae": 1.7,
                        "r2": -0.14,
                        "rmse": 2.2,
                    },
                    "baseline_mean_predictor": {"r2": -1.1},
                },
                "explainability": {
                    "shap_global": {
                        "method": "TreeSHAP",
                        "features": [{"feature": "x", "mean_absolute_shap": 0.5}],
                    },
                    "lime": {"method": "LIME", "approximation_quality_metric": None},
                },
                "limitations": ["Spatial holdout only."],
            }
        ),
        encoding="utf-8",
    )

    result = client.get("/api/validation/status").get_json()
    modules = {module["id"]: module for module in result["modules"]}

    assert modules["surrogate"]["status"] == "partial"
    assert modules["surrogate"]["metrics"]["spatial_holdout"]["r2"] == -0.14
    assert modules["interpretability"]["status"] == "partial"
    assert result["provider"]["configured"] is True
    assert result["provider"]["counts_as_validation"] is False
    assert "server-only-test-key" not in json.dumps(result)
    assert result["summary"]["modules_with_recorded_evidence"] == 2

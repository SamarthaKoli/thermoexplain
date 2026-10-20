import json
import types

import httpx
import numpy as np
import pytest
from groq import APIStatusError

from backend.app import create_app
from backend.config import Settings
from backend.ml.surrogate import (
    SurrogateError,
    create_spatial_holdout_split,
    explain_lime_local,
    explain_shap_local,
    fit_surrogate_from_samples,
    load_model_bundle,
    predict_one,
    validate_feature_matrix,
)
from backend.services import gpt_oss_service


def make_samples():
    count = 640
    x = np.linspace(0.001, 0.999, count)
    y = (np.arange(count) % 17) / 16
    z = (np.arange(count) % 23) / 22
    reynolds = np.where(np.arange(count) % 2, 560, 11240)
    features = np.column_stack((x, y, z, reynolds))
    target = 290 + 5 * x + 2 * y - z + (reynolds == 11240) * 0.8
    case_ids = np.where(reynolds == 560, "Re560/coarse", "Re11240/coarse")
    case_metadata = [
        {
            "case_id": case_id,
            "reynolds_number": reynolds_number,
            "mesh_resolution": "coarse",
            "solution_time": "200",
            "cell_count": count // 2,
            "sample_count": count // 2,
            "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        }
        for case_id, reynolds_number in (
            ("Re560/coarse", 560),
            ("Re11240/coarse", 11240),
        )
    ]
    return {
        "features": features,
        "target_values": target,
        "normalized_coordinates": features[:, :3],
        "case_ids": case_ids,
        "case_metadata": case_metadata,
        "target": {"region": "coolant", "field": "T", "units": "K"},
        "coordinate_units": None,
        "coordinate_method": "synthetic test coordinates",
        "manifest_generated_at": "test fixture",
        "sampling": {"method": "synthetic test fixture", "seed": 1},
    }


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("surrogate-model")
    metadata = fit_surrogate_from_samples(
        make_samples(),
        model_directory=directory,
        model_id="synthetic_extra_trees",
    )
    bundle, loaded_metadata = load_model_bundle(directory, metadata["model_id"])
    return directory, metadata, bundle, loaded_metadata


def test_feature_validation_rejects_bad_shape_and_non_finite_values():
    with pytest.raises(SurrogateError, match="Expected 4"):
        validate_feature_matrix([1, 2, 3])
    with pytest.raises(SurrogateError, match="finite"):
        validate_feature_matrix([1, 2, float("nan"), 560])
    with pytest.raises(SurrogateError, match="numeric"):
        validate_feature_matrix([1, 2, "unknown", 560])


def test_spatial_split_keeps_blocks_separate_and_leaves_neighbor_buffer():
    coordinates = np.column_stack(
        (
            np.linspace(0, 1, 800),
            np.tile([0.1, 0.5, 0.9, 0.5], 200),
            np.tile([0.1, 0.5, 0.9, 0.5], 200),
        )
    )
    train, validation, split = create_spatial_holdout_split(coordinates)

    assert split["method"] == "spatial_block_holdout_with_two_bin_buffer"
    assert split["axis"] == "x"
    assert max(split["training_bins"]) <= 3
    assert min(split["validation_bins"]) >= 6
    assert set(split["training_bins"]).isdisjoint(split["validation_bins"])
    assert len(train) + len(validation) < len(coordinates)


def test_saved_model_prediction_is_repeatable_and_metadata_reports_heldout_only(
    trained_model,
):
    _directory, metadata, bundle, loaded_metadata = trained_model
    features = [0.82, 0.4, 0.6, 11240]

    first = predict_one(bundle, features)
    second = predict_one(bundle, features)

    assert first == pytest.approx(second, rel=1e-12, abs=1e-12)
    assert loaded_metadata["model_id"] == metadata["model_id"]
    assert metadata["metrics"]["metric_scope"].startswith("held-out spatial validation")
    assert metadata["dataset"]["exploratory"] is True
    assert metadata["coordinate_units"] is None


def test_model_loader_rejects_unsafe_and_missing_model_ids(trained_model):
    directory, _metadata, _bundle, _loaded_metadata = trained_model

    with pytest.raises(SurrogateError, match="invalid"):
        load_model_bundle(directory, "../outside")
    with pytest.raises(SurrogateError, match="unavailable"):
        load_model_bundle(directory, "missing_model")


def test_shap_returns_actual_signed_contributions_that_reconstruct_prediction(
    trained_model,
):
    _directory, _metadata, bundle, _loaded_metadata = trained_model

    explanation = explain_shap_local(bundle, [0.82, 0.4, 0.6, 11240])

    assert explanation["method"] == "SHAP TreeExplainer"
    assert explanation["additivity_checked"] is True
    assert explanation["additivity_passed"] is True
    assert explanation["additivity_error"] <= 1e-3
    assert explanation["prediction"] == pytest.approx(
        explanation["baseline"]
        + sum(item["contribution"] for item in explanation["contributions"]),
        abs=1e-3,
    )
    assert {item["feature"] for item in explanation["contributions"]} == {
        "x",
        "y",
        "z",
        "reynolds_number",
    }


def test_lime_returns_local_approximation_and_only_verified_reynolds_categories(
    trained_model,
):
    _directory, _metadata, bundle, _loaded_metadata = trained_model

    explanation = explain_lime_local(bundle, [0.82, 0.4, 0.6, 11240], num_samples=400)

    assert explanation["method"] == "LIME Tabular local surrogate"
    assert explanation["sample"]["reynolds_number"] == 11240
    assert np.isfinite(explanation["local_r2"])
    assert np.isfinite(explanation["local_approximation_absolute_error"])
    assert explanation["caveat"].startswith("LIME is a local approximation")
    with pytest.raises(SurrogateError, match="verified Reynolds"):
        explain_lime_local(bundle, [0.82, 0.4, 0.6, 3000], num_samples=100)


def test_prediction_api_validates_coordinates_and_returns_real_model_output(
    trained_model, tmp_path
):
    directory, metadata, _bundle, _loaded_metadata = trained_model
    raw = tmp_path / "raw"
    for case in metadata["cases"]:
        case_path = raw.joinpath(*case["case_id"].split("/"))
        case_path.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "scanned",
                "generated_at": "test fixture",
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "metadata": {
                            "reynolds_number": case["reynolds_number"],
                            "mesh_resolution": case["mesh_resolution"],
                        },
                    }
                    for case in metadata["cases"]
                ],
            }
        ),
        encoding="utf-8",
    )
    # Bind this test artifact to the exact manifest bytes served by the API.
    from backend.ml.surrogate import manifest_digest

    updated = fit_surrogate_from_samples(
        make_samples(),
        model_directory=directory,
        model_id="synthetic_extra_trees_api",
        manifest_sha256=manifest_digest(manifest_path),
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=manifest_path,
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        models_path=directory,
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()
    payload = {
        "model_id": updated["model_id"],
        "case_id": "Re560/coarse",
        "coordinates": {"x": 0.4, "y": 0.4, "z": 0.6},
    }

    assert client.post("/api/ml/predict", json={**payload, "coordinates": {"x": "bad", "y": 0, "z": 0}}).status_code == 400
    assert client.post(
        "/api/ml/predict",
        json={**payload, "coordinates": {"x": 2.0, "y": 0.4, "z": 0.6}},
    ).status_code == 400
    response = client.post("/api/ml/predict", json=payload)

    assert response.status_code == 200
    assert response.get_json()["prediction"] == pytest.approx(
        predict_one(
            load_model_bundle(directory, updated["model_id"])[0],
            [0.4, 0.4, 0.6, 560],
        )
    )
    assert client.get("/api/ml/models").get_json()["models"]
    assert client.get(
        "/api/ml/shap/global", query_string={"model_id": updated["model_id"]}
    ).status_code == 200
    shap_response = client.post("/api/ml/shap/local", json=payload)
    lime_response = client.post("/api/ml/lime/local", json=payload)
    assert shap_response.status_code == 200
    assert shap_response.get_json()["explanation"]["additivity_passed"] is True
    assert lime_response.status_code == 200
    assert np.isfinite(lime_response.get_json()["explanation"]["local_r2"])


def test_gpt_oss_interpretation_receives_actual_prediction_and_shap_lime_results_mocked_provider(
    trained_model, tmp_path, monkeypatch
):
    directory, metadata, _bundle, _loaded_metadata = trained_model
    raw = tmp_path / "raw"
    for case in metadata["cases"]:
        raw.joinpath(*case["case_id"].split("/")).mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "metadata": {
                            "reynolds_number": case["reynolds_number"],
                            "mesh_resolution": case["mesh_resolution"],
                        },
                    }
                    for case in metadata["cases"]
                ],
            }
        ),
        encoding="utf-8",
    )
    from backend.ml.surrogate import manifest_digest

    model = fit_surrogate_from_samples(
        make_samples(),
        model_directory=directory,
        model_id="synthetic_groq_interpretation",
        manifest_sha256=manifest_digest(manifest_path),
    )
    captured_requests = []

    class FakeMessage:
        content = "The computed explanation describes this model output only."

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured_requests.append(kwargs)
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=manifest_path,
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        models_path=directory,
        gpt_oss_api_key="server-only-groq-test-key",
        gpt_oss_model="openai/gpt-oss-120b",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()
    request_payload = {
        "model_id": model["model_id"],
        "case_id": "Re11240/coarse",
        "coordinates": {"x": 0.82, "y": 0.4, "z": 0.6},
    }

    for method in ("shap", "lime"):
        response = client.post(
            "/api/ml/gpt-oss/interpretation",
            json={**request_payload, "method": method},
        )
        assert response.status_code == 200
        result = response.get_json()
        assert result["explanation_method"] == method
        assert result["interpretation"]["provider"] == "GPT-OSS"
        assert result["interpretation"]["model"] == "openai/gpt-oss-120b"
        assert "AI-generated interpretation" == result["interpretation"]["label"]

    assert len(captured_requests) == 2
    assert all(
        request["model"] == "openai/gpt-oss-120b"
        for request in captured_requests
    )
    for method, request in zip(("shap", "lime"), captured_requests):
        context = json.loads(
            request["messages"][0]["content"].split(
                "Server-computed context (JSON):\n", 1
            )[1]
        )
        assert context["explanation_method"] == method
        assert context["computed_explanation"]["prediction"] > 0
        assert context["case"]["case_id"] == "Re11240/coarse"
        assert context["features"]["reynolds_number"] == 11240


def test_gpt_oss_failure_is_explicit_and_never_exposes_credentials(monkeypatch, tmp_path):
    class FakeCompletions:
        def create(self, **_kwargs):
            import httpx

            raise httpx.ConnectError("offline")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        gpt_oss_api_key="server-only-groq-test-key",
        gpt_oss_model="openai/gpt-oss-20b",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/api/llm/connection-test",
        json={},
    )

    assert response.status_code == 502
    assert response.get_json()["provider"] == "GPT-OSS"
    assert response.get_json()["status"] == "failed"
    assert response.get_json()["tested_at"]
    assert response.get_json()["diagnostic"] == (
        "Could not reach the configured GPT-OSS service."
    )
    assert "server-only-groq-test-key" not in response.get_data(as_text=True)


def test_gpt_oss_connection_test_uses_live_completions_route_mocked_provider(
    monkeypatch,
):
    captured = {}

    class FakeMessage:
        content = "ready"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_options"] = kwargs

        chat = FakeChat()

    monkeypatch.setattr(gpt_oss_service, "Groq", FakeClient)
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        gpt_oss_api_key="server-only-groq-test-key",
        gpt_oss_model="openai/gpt-oss-120b",
    )
    result = gpt_oss_service.test_gpt_oss_connection(settings)

    assert result["status"] == "connected"
    assert result["provider"] == "GPT-OSS"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["tested_at"]
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["max_completion_tokens"] == 128
    assert captured["client_options"]["timeout"] == 30
    assert captured["client_options"]["max_retries"] == 0


@pytest.mark.parametrize(
    "completion",
    [
        types.SimpleNamespace(choices=[]),
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="  "))]
        ),
    ],
)
def test_gpt_oss_connection_test_rejects_empty_provider_responses(
    monkeypatch, completion
):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            return completion

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        gpt_oss_api_key="server-only-groq-test-key",
        gpt_oss_model="openai/gpt-oss-120b",
    )

    with pytest.raises(gpt_oss_service.GptOssServiceError) as error:
        gpt_oss_service.test_gpt_oss_connection(settings)
    assert error.value.status_code == 502
    assert "GPT-OSS service returned" in error.value.diagnostic


@pytest.mark.parametrize(
    ("status_code", "expected_status", "expected_diagnostic"),
    [
        (401, 401, "Provider rejected the configured API key."),
        (403, 403, "Provider denied access to this model or API."),
        (404, 404, "Provider could not find the configured model ID."),
        (429, 429, "Provider rate limit or quota was exceeded."),
    ],
)
def test_gpt_oss_provider_error_status_is_safe_and_actionable(
    monkeypatch, status_code, expected_status, expected_diagnostic
):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    provider_error = APIStatusError(
        "Secret-bearing provider details", response=response, body={}
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        gpt_oss_api_key="server-only-groq-test-key",
        gpt_oss_model="openai/gpt-oss-20b",
    )

    error = gpt_oss_service._gpt_oss_error(settings, provider_error)
    payload = error.to_payload()

    assert error.status_code == expected_status
    assert payload["diagnostic"] == expected_diagnostic
    assert payload["tested_at"]
    assert "Secret-bearing" not in str(payload)
    assert "server-only-groq-test-key" not in str(payload)


def test_gpt_oss_rejects_unsupported_model_ids():
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        gpt_oss_api_key="test-key",
        gpt_oss_model="my-custom-gpt-oss-model",
    )

    with pytest.raises(gpt_oss_service.GptOssServiceError, match="GPT_OSS_MODEL"):
        gpt_oss_service.require_gpt_oss_configuration(settings)


def test_gpt_oss_missing_configuration_has_an_explicit_status_and_error():
    settings = Settings(cors_origins=("http://localhost:5173",))
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    status = client.get("/api/llm/status").get_json()
    assert status["configured"] is False
    assert status["status"] == "not_verified"
    assert status["provider"] == "GPT-OSS"
    response = client.post("/api/llm/connection-test", json={})
    interpretation = client.post(
        "/api/ml/gpt-oss/interpretation",
        json={"method": "shap"},
    )

    assert response.status_code == 503
    assert "GPT_OSS_API_KEY" in response.get_json()["error"]
    assert interpretation.status_code == 503

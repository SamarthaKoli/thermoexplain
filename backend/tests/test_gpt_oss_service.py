from types import SimpleNamespace

import httpx
import pytest

from backend.app import create_app
from backend.config import Settings
from backend.services import gpt_oss_service
from backend.services.gpt_oss_service import GptOssServiceError


def make_settings(**overrides):
    options = {
        "cors_origins": ("http://localhost:5173",),
        "gpt_oss_api_key": "server-only-test-key",
        "gpt_oss_model": "openai/gpt-oss-120b",
    }
    options.update(overrides)
    return Settings(**options)


def test_lightweight_status_verifies_model_reachability(monkeypatch):
    captured = {}

    class FakeModels:
        @staticmethod
        def list():
            return SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])

    class FakeClient:
        models = FakeModels()

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gpt_oss_service, "Groq", FakeClient)
    result = gpt_oss_service.check_gpt_oss_service(make_settings())

    assert result["status"] == "reachable"
    assert result["provider"] == "GPT-OSS"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["checked_at"]
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 30
    assert "server-only-test-key" not in str(result)


def test_inference_probe_reports_measured_latency_and_uses_configured_model(monkeypatch):
    captured = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Model is available.")
                    )
                ],
                model="openai/gpt-oss-120b",
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    result = gpt_oss_service.test_gpt_oss_connection(make_settings())

    assert result["status"] == "connected"
    assert result["provider"] == "GPT-OSS"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["tested_at"]
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["max_completion_tokens"] == 128
    assert captured["reasoning_effort"] == "low"


def test_inference_probe_rejects_a_different_returned_model(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Model is available.")
                    )
                ],
                model="openai/gpt-oss-20b",
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    with pytest.raises(GptOssServiceError, match="different model") as error:
        gpt_oss_service.test_gpt_oss_connection(make_settings())

    assert error.value.model == "openai/gpt-oss-20b"


def test_generation_uses_supported_completion_budget_and_low_reasoning(monkeypatch):
    captured = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="Complete answer."),
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())

    result = gpt_oss_service.generate_text(make_settings(), "Explain this result.")

    assert result == "Complete answer."
    assert captured["max_completion_tokens"] == 2_048
    assert captured["reasoning_effort"] == "low"


def test_feature_interpretation_uses_only_supplied_evidence(monkeypatch):
    captured = {}

    def fake_completion(_settings, prompt, **_kwargs):
        captured["prompt"] = prompt
        return "The supplied validation evidence is limited."

    monkeypatch.setattr(gpt_oss_service, "_request_completion", fake_completion)
    result = gpt_oss_service.interpret_feature_context(
        make_settings(),
        "validation",
        {"recorded_metric": 1.25, "status": "exploratory"},
    )

    assert "1.25" in captured["prompt"]
    assert "Do not invent or alter numbers" in captured["prompt"]
    assert "regulatory compliance" in captured["prompt"]
    assert result["interpretation"] == "The supplied validation evidence is limited."
    assert result["model"] == "openai/gpt-oss-120b"


def test_feature_interpretation_rejects_unsupported_feature():
    with pytest.raises(ValueError, match="Unsupported"):
        gpt_oss_service.interpret_feature_context(
            make_settings(), "design", {"verified": True}
        )


def test_truncated_completion_is_not_returned_as_a_success(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="Incomplete partial answer"),
                    )
                ]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())

    with pytest.raises(GptOssServiceError) as error:
        gpt_oss_service.generate_text(make_settings(), "Explain this result.")

    assert error.value.status_code == 502
    assert error.value.diagnostic == (
        "GPT-OSS response was truncated before completion; no partial answer was returned."
    )


def test_status_route_returns_not_verified_when_credentials_are_missing():
    app = create_app(make_settings(gpt_oss_api_key=None))
    app.config.update(TESTING=True)

    response = app.test_client().get("/api/llm/status")

    assert response.status_code == 200
    assert response.get_json()["status"] == "not_verified"
    assert response.get_json()["configured"] is False
    assert response.get_json()["provider"] == "GPT-OSS"


def test_connection_test_sanitizes_upstream_errors_and_credentials(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            raise httpx.ConnectError("sensitive connection detail")

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())
    app = create_app(make_settings())
    app.config.update(TESTING=True)

    response = app.test_client().post("/api/llm/connection-test")

    assert response.status_code == 502
    assert response.get_json()["status"] == "failed"
    assert response.get_json()["diagnostic"] == (
        "Could not reach the configured GPT-OSS service."
    )
    assert "sensitive connection detail" not in response.get_data(as_text=True)
    assert "server-only-test-key" not in response.get_data(as_text=True)


def test_missing_model_from_service_is_explicit(monkeypatch):
    class FakeModels:
        @staticmethod
        def list():
            return SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-20b")])

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gpt_oss_service, "Groq", lambda **_kwargs: FakeClient())

    with pytest.raises(GptOssServiceError) as error:
        gpt_oss_service.check_gpt_oss_service(make_settings())

    assert error.value.status_code == 404
    assert "configured GPT-OSS model is not available" in error.value.diagnostic


def test_unexpected_provider_is_rejected_without_initializing_client(monkeypatch):
    def unexpected_client(**_kwargs):
        pytest.fail("A non-GPT-OSS provider must not initialize a client.")

    monkeypatch.setattr(gpt_oss_service, "Groq", unexpected_client)

    with pytest.raises(GptOssServiceError, match="only supports GPT-OSS"):
        gpt_oss_service.require_gpt_oss_configuration(
            make_settings(llm_provider="gemini")
        )

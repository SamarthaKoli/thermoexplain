"""Optional real provider smoke tests; skipped without backend-only credentials."""

import pytest

from backend.config import Settings
from backend.services.gpt_oss_service import (
    SUPPORTED_GPT_OSS_MODELS,
    test_gpt_oss_connection as run_gpt_oss_connection_test,
)

settings = Settings.from_environment()

gpt_oss_configured = bool(
    settings.gpt_oss_api_key and settings.gpt_oss_model in SUPPORTED_GPT_OSS_MODELS
)


@pytest.mark.skipif(
    not gpt_oss_configured,
    reason="Optional live GPT-OSS test requires a backend API key and supported model.",
)
def test_live_gpt_oss_provider_inference():
    result = run_gpt_oss_connection_test(settings)
    assert result["status"] == "connected"
    assert result["provider"] == "GPT-OSS"
    assert result["model"] == settings.gpt_oss_model
    assert result["tested_at"]
    assert result["latency_ms"] >= 0

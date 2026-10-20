"""Central GPT-OSS generation and provider verification through Groq."""

import json
import time
from typing import Any

import httpx
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    Groq,
)

from backend.config import Settings
from backend.services.provider_test_result import (
    ProviderTestError,
    provider_http_diagnostic,
    utc_timestamp,
)

PROVIDER_NAME = "GPT-OSS"
SUPPORTED_GPT_OSS_MODELS = frozenset(
    {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
)
REQUEST_TIMEOUT_SECONDS = 30


class GptOssServiceError(ProviderTestError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model: str | None = None,
    ):
        super().__init__(PROVIDER_NAME, model, message, status_code)


def _gpt_oss_error(settings: Settings, error: Exception) -> GptOssServiceError:
    status = getattr(error, "status_code", None)
    status_code, diagnostic = provider_http_diagnostic(
        status if isinstance(status, int) else None
    )
    if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
        status_code, diagnostic = 504, "GPT-OSS request timed out."
    elif isinstance(error, (APIConnectionError, httpx.HTTPError)):
        status_code, diagnostic = 502, "Could not reach the configured GPT-OSS service."
    return GptOssServiceError(diagnostic, status_code, settings.gpt_oss_model)


def require_gpt_oss_configuration(settings: Settings) -> None:
    if settings.llm_provider != "gpt_oss":
        raise GptOssServiceError(
            "This application only supports GPT-OSS as its language-model provider.",
            503,
            settings.gpt_oss_model,
        )
    if not settings.gpt_oss_api_key or not settings.gpt_oss_model:
        raise GptOssServiceError(
            "GPT-OSS is not configured. Set GPT_OSS_API_KEY and GPT_OSS_MODEL in the backend environment.",
            503,
            settings.gpt_oss_model,
        )
    if settings.gpt_oss_model not in SUPPORTED_GPT_OSS_MODELS:
        raise GptOssServiceError(
            "GPT_OSS_MODEL must be openai/gpt-oss-20b or openai/gpt-oss-120b.",
            503,
            settings.gpt_oss_model,
        )


def _configured_client(settings: Settings) -> Groq:
    require_gpt_oss_configuration(settings)
    client_options: dict[str, Any] = {
        "api_key": settings.gpt_oss_api_key,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "max_retries": 0,
    }
    if settings.gpt_oss_base_url:
        client_options["base_url"] = settings.gpt_oss_base_url
    return Groq(**client_options)


def _request_completion(
    settings: Settings, prompt: str, *, max_completion_tokens: int = 800
) -> str:
    try:
        client = _configured_client(settings)
        response = client.chat.completions.create(
            model=settings.gpt_oss_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_completion_tokens,
        )
    except (APIError, httpx.HTTPError) as error:
        raise _gpt_oss_error(settings, error) from error
    if not response.choices:
        raise GptOssServiceError(
            "The configured GPT-OSS service returned no completion.",
            model=settings.gpt_oss_model,
        )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise GptOssServiceError(
            "The configured GPT-OSS service returned empty text.",
            model=settings.gpt_oss_model,
        )
    return content.strip()


def generate_text(
    settings: Settings, prompt: str, *, max_completion_tokens: int = 800
) -> str:
    return _request_completion(
        settings, prompt, max_completion_tokens=max_completion_tokens
    )


def check_gpt_oss_service(settings: Settings) -> dict[str, Any]:
    client = _configured_client(settings)
    try:
        available_models = client.models.list()
    except (APIError, httpx.HTTPError) as error:
        raise _gpt_oss_error(settings, error) from error
    matching_model = any(
        getattr(model, "id", None) == settings.gpt_oss_model
        for model in available_models.data
    )
    if not matching_model:
        raise GptOssServiceError(
            "The configured GPT-OSS model is not available to this account.",
            404,
            settings.gpt_oss_model,
        )
    return {
        "provider": PROVIDER_NAME,
        "configured": True,
        "model": settings.gpt_oss_model,
        "status": "reachable",
        "checked_at": utc_timestamp(),
        "diagnostic": "The GPT-OSS service is reachable and the configured model is available.",
    }


def test_gpt_oss_connection(settings: Settings) -> dict[str, Any]:
    started_at = time.perf_counter()
    _request_completion(
        settings,
        "Connection check. Reply with a brief confirmation that the model is available. "
        "Do not provide engineering advice.",
        max_completion_tokens=128,
    )
    latency_ms = round((time.perf_counter() - started_at) * 1_000, 1)
    return {
        "provider": PROVIDER_NAME,
        "status": "connected",
        "model": settings.gpt_oss_model or "",
        "tested_at": utc_timestamp(),
        "latency_ms": latency_ms,
        "diagnostic": "GPT-OSS returned a non-empty inference-probe completion.",
    }


def interpret_surrogate_explanation(
    settings: Settings, computed_context: dict[str, Any]
) -> dict[str, str]:
    prompt = (
        "Explain this verified exploratory CFD surrogate prediction and the supplied "
        "computed explanation to a beginner. Use only the supplied information. Do not "
        "calculate, alter, round, or invent numerical predictions, contributions, "
        "metrics, units, physical causes, causal claims, or design recommendations. "
        "Explain signed SHAP contributions relative to their given baseline, or explain "
        "LIME as a local approximation. State that this is not a new CFD simulation, "
        "not physical causality, and not validation on an independent case or design. "
        "If evidence is insufficient, say so. The frontend displays authoritative "
        "numbers separately; focus on interpretation.\n\n"
        "Server-computed context (JSON):\n"
        f"{json.dumps(computed_context, allow_nan=False, sort_keys=True)}"
    )
    try:
        response = _request_completion(settings, prompt)
    except GptOssServiceError:
        raise
    return {
        "label": "AI-generated interpretation",
        "provider": PROVIDER_NAME,
        "model": settings.gpt_oss_model or "",
        "interpretation": response,
        "generated_at": utc_timestamp(),
    }


def interpret_cfd_statistics(
    settings: Settings, statistics: dict[str, object]
) -> dict[str, str]:
    prompt = (
        "You are interpreting verified CFD measurements for an engineering researcher. "
        "Explain the supplied statistics in simple, cautious language. Do not invent "
        "values, units, physical causes, design recommendations, or simulation results. "
        "State when the supplied information is insufficient. This is an AI-generated "
        "interpretation, not a new CFD simulation or validated engineering conclusion.\n\n"
        "Verified selected-case metadata and computed statistics (JSON):\n"
        f"{json.dumps(statistics, allow_nan=False, sort_keys=True)}"
    )
    interpretation = _request_completion(settings, prompt)
    return {
        "label": "AI-generated interpretation",
        "provider": PROVIDER_NAME,
        "model": settings.gpt_oss_model or "",
        "interpretation": interpretation,
        "generated_at": utc_timestamp(),
    }

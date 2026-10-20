"""Safe, structured provider connection-test diagnostics."""

from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderTestError(Exception):
    def __init__(
        self,
        provider: str,
        model: str | None,
        diagnostic: str,
        status_code: int = 502,
    ):
        super().__init__(diagnostic)
        self.provider = provider
        self.model = model
        self.diagnostic = diagnostic
        self.status_code = status_code
        self.tested_at = utc_timestamp()

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": self.diagnostic,
            "provider": self.provider,
            "model": self.model,
            "status": "failed",
            "tested_at": self.tested_at,
            "diagnostic": self.diagnostic,
        }


def provider_http_diagnostic(
    status_code: int | None, error_body: object | None = None
) -> tuple[int, str]:
    error_details = (
        error_body.get("error")
        if isinstance(error_body, dict)
        else None
    )
    if isinstance(error_details, dict):
        category_parts = (
            error_details.get("type"),
            error_details.get("code"),
            error_details.get("message"),
        )
    else:
        category_parts = (error_details,)
    provider_category = " ".join(
        value.casefold()
        for value in category_parts
        if isinstance(value, str)
    )
    if status_code == 402 or any(
        marker in provider_category
        for marker in ("insufficient_quota", "billing", "quota_exceeded")
    ):
        return 402, "Provider quota or billing limit was reached."
    if status_code == 401:
        return 401, "Provider rejected the configured API key."
    if status_code == 403:
        return 403, "Provider denied access to this model or API."
    if status_code == 404:
        return 404, "Provider could not find the configured model ID."
    if status_code == 408:
        return 504, "Provider request timed out."
    if status_code == 429:
        return 429, "Provider rate limit was exceeded; wait before retrying."
    if status_code == 413:
        return 413, "Provider rejected the request because its payload was too large."
    if status_code == 400:
        return 400, "Provider rejected the request or configured model."
    if status_code == 422:
        return 400, "Provider could not process the request; check its inputs and format."
    if status_code is not None and status_code >= 500:
        return 502, "Provider is temporarily unavailable."
    return 502, "Provider request failed; verify model access and network connectivity."

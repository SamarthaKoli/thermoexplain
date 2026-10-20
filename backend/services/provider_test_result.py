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


def provider_http_diagnostic(status_code: int | None) -> tuple[int, str]:
    if status_code == 401:
        return 401, "Provider rejected the configured API key."
    if status_code == 403:
        return 403, "Provider denied access to this model or API."
    if status_code == 404:
        return 404, "Provider could not find the configured model ID."
    if status_code == 408:
        return 504, "Provider request timed out."
    if status_code == 429:
        return 429, "Provider rate limit or quota was exceeded."
    if status_code == 400:
        return 400, "Provider rejected the request or configured model."
    if status_code is not None and status_code >= 500:
        return 502, "Provider is temporarily unavailable."
    return 502, "Provider request failed; verify model access and network connectivity."

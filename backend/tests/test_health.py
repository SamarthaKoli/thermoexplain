from pathlib import Path

import pytest

from backend.app import create_app
from backend.config import Settings


def create_test_client():
    app = create_app(Settings(cors_origins=("http://localhost:5173",)))
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint_reports_backend_process_status():
    response = create_test_client().get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_health_endpoint_allows_the_127_0_0_1_frontend_origin():
    app = create_app(
        Settings(
            cors_origins=(
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            )
        )
    )

    response = app.test_client().get(
        "/api/health",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:5173"


def test_readiness_endpoint_reports_application_readiness():
    response = create_test_client().get("/api/ready")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ready"}


def test_unknown_endpoint_returns_safe_json_error():
    response = create_test_client().get("/api/unknown")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Not found"}


def test_empty_cors_configuration_fails_with_helpful_message(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "")

    with pytest.raises(ValueError, match="CORS_ORIGINS must contain"):
        Settings.from_environment()


def test_environment_loading_uses_project_root_env_independent_of_working_directory(
    monkeypatch, tmp_path
):
    from backend import config

    captured = {}

    def capture_dotenv_path(*, dotenv_path):
        captured["dotenv_path"] = dotenv_path
        return True

    monkeypatch.setattr(config, "load_dotenv", capture_dotenv_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    Settings.from_environment()

    assert captured["dotenv_path"] == (
        Path(__file__).resolve().parents[2] / ".env"
    )

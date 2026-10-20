"""HTTP routing integration tests; provider calls are deliberately mocked elsewhere."""

from backend.app import create_app
from backend.config import Settings


def make_settings(tmp_path):
    return Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.json",
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        knowledge_documents_path=tmp_path / "knowledge",
        knowledge_index_path=tmp_path / "processed" / "knowledge-index.json",
    )


def test_provider_and_knowledge_routes_are_registered_with_expected_methods(tmp_path):
    app = create_app(make_settings(tmp_path))
    rules = {
        (rule.rule, method)
        for rule in app.url_map.iter_rules()
        for method in rule.methods
        if method not in {"HEAD", "OPTIONS"}
    }

    assert {
        ("/api/llm/status", "GET"),
        ("/api/llm/connection-test", "POST"),
        ("/api/llm/analysis", "POST"),
        ("/api/ml/gpt-oss/interpretation", "POST"),
        ("/api/knowledge/status", "GET"),
        ("/api/knowledge/retrieve", "POST"),
        ("/api/knowledge/answer", "POST"),
        ("/api/knowledge/connection-test", "POST"),
    }.issubset(rules)


def test_wrong_methods_missing_routes_and_trailing_slashes_return_json(tmp_path):
    app = create_app(make_settings(tmp_path))
    app.config.update(TESTING=True)
    client = app.test_client()

    for path in (
        "/api/llm/connection-test",
        "/api/ml/gpt-oss/interpretation",
        "/api/knowledge/retrieve",
    ):
        response = client.get(path)
        assert response.status_code == 405
        assert response.is_json
        assert "Method not allowed" in response.get_json()["error"]

    for path in (
        "/api/llm/missing",
        "/api/ml/gpt-oss/missing",
        "/api/knowledge/missing",
        "/api/llm/status/",
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.is_json
        assert response.get_json()["error"] == "Not found"


def test_backend_cors_allows_configured_vite_origin_and_expected_port(tmp_path):
    app = create_app(make_settings(tmp_path))
    response = app.test_client().options(
        "/api/llm/status",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"

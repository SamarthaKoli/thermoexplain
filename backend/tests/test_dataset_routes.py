import json

from backend.app import create_app
from backend.config import Settings
from backend.services.dataset_inspector import (
    scan_dataset,
    write_audit_report,
    write_manifest,
)


def test_dataset_endpoints_report_empty_dataset_without_fabricated_cases(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=tmp_path / "manifest.json",
        audit_report_path=tmp_path / "reports" / "audit.md",
        processed_path=tmp_path / "processed",
    )
    manifest = scan_dataset(raw)
    write_manifest(manifest, settings.manifest_path)
    write_audit_report(manifest, settings.audit_report_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/api/dataset/scan").get_json()["status"] == "no_dataset_files"
    assert client.get("/api/dataset/cases").get_json() == {"cases": []}
    assert client.get("/api/dataset/fields").get_json() == {"fields": []}
    assert client.get("/api/dataset/preprocessing").get_json()["status"] == "not_run"
    assert client.get("/api/dataset/audit").get_json()["status"] == "available"
    assert client.get("/api/dataset/case-metadata?case_id=missing").status_code == 404
    assert client.get("/api/dataset/mesh-info").status_code == 400


def test_scan_endpoint_refreshes_manifest_and_audit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "points.csv").write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=tmp_path / "manifest.json",
        audit_report_path=tmp_path / "reports" / "audit.md",
        processed_path=tmp_path / "processed",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post("/api/dataset/scan")

    assert response.status_code == 200
    assert response.get_json()["status"] == "scanned"
    assert json.loads(settings.manifest_path.read_text(encoding="utf-8"))["cases"]
    assert settings.audit_report_path.is_file()


def test_preprocess_endpoint_accepts_only_scanned_supported_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "points.csv").write_text(
        "x,y,z,temperature\n1,2,3,300\n4,5,6,301\n", encoding="utf-8"
    )
    settings = Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=raw,
        manifest_path=tmp_path / "manifest.json",
        audit_report_path=tmp_path / "reports" / "audit.md",
        processed_path=tmp_path / "processed",
    )
    manifest = scan_dataset(raw)
    write_manifest(manifest, settings.manifest_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/api/dataset/preprocess",
        json={"file": "points.csv", "sample_size": 1, "seed": 5},
    )

    assert response.status_code == 201
    assert response.get_json()["output_rows"] == 1
    assert client.get("/api/dataset/preprocessing").get_json()["status"] == "completed"
    assert client.post(
        "/api/dataset/preprocess", json={"file": "..\\outside.csv"}
    ).status_code == 400

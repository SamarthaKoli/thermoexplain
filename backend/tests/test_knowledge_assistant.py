import json
import io
import sys
import types

import pytest

from backend.app import create_app
from backend.config import Settings
from backend.services import bert_classifier, knowledge_answer
from backend.services.bert_classifier import (
    BertClassifier,
    BertError,
    _validate_test_labels,
    _verified_label_set,
)
from backend.services.knowledge_corpus import (
    KnowledgeCorpus,
    KnowledgeError,
    _extract_document,
)
from backend.services.gpt_oss_service import GptOssServiceError


def make_settings(tmp_path, *, api_key=None, model=None, bert_model=None, bert_test=None):
    documents = tmp_path / "knowledge"
    documents.mkdir(parents=True, exist_ok=True)
    return Settings(
        cors_origins=("http://localhost:5173",),
        dataset_path=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.json",
        audit_report_path=tmp_path / "audit.md",
        processed_path=tmp_path / "processed",
        gpt_oss_api_key=api_key,
        gpt_oss_model=model,
        knowledge_documents_path=documents,
        knowledge_index_path=tmp_path / "processed" / "knowledge-index.json",
        bert_model_path=bert_model,
        bert_test_data_path=bert_test,
    )


def make_indexed_app(tmp_path, *, api_key=None, model=None):
    settings = make_settings(tmp_path, api_key=api_key, model=model)
    source = settings.knowledge_documents_path / "cooling.md"
    source.write_text(
        "---\ntitle: Cooling research note\nsource_url: https://example.org/cooling\n---\n"
        "# Liquid cooling\n"
        "Coolant flow removes heat from battery cells. Pumping power and flow distribution "
        "affect the thermal management design. CFD can compare temperature fields.\n"
        "# Evidence limits\n"
        "A steady-state result does not establish transient safety.",
        encoding="utf-8",
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()
    indexed = client.post("/api/knowledge/index")
    assert indexed.status_code == 200
    return settings, app, client, indexed.get_json()


def make_pdf_with_text(text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode("ascii"))
        document.extend(item)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(document)


def test_parses_actual_pdf_metadata_and_page_number(tmp_path):
    pdf_path = tmp_path / "thermal.pdf"
    pdf_path.write_bytes(make_pdf_with_text("Battery cooling transfers heat"))

    document, passages = _extract_document(pdf_path)

    assert document["page_count"] == 1
    assert document["title"] == "thermal"
    assert passages[0]["page"] == 1
    assert "Battery cooling transfers heat" in passages[0]["text"]


def test_markdown_metadata_sections_chunking_and_stable_ids(tmp_path):
    source = tmp_path / "manual.md"
    source.write_text(
        "---\ntitle: Flow guide\nsource_url: https://example.org/guide\n---\n"
        "## Pressure drop\n"
        + ("Pressure drop depends on fluid viscosity and channel geometry. " * 55),
        encoding="utf-8",
    )

    document, passages = _extract_document(source)
    _, repeated_passages = _extract_document(source)

    assert document["title"] == "Flow guide"
    assert document["source_url"] == "https://example.org/guide"
    assert len(passages) > 1
    assert {passage["section"] for passage in passages} == {"Pressure drop"}
    assert passages[0]["passage_id"] == repeated_passages[0]["passage_id"]
    assert passages[0]["end_character"] - passages[0]["start_character"] <= 1_200


def test_rejects_malformed_pdf_and_non_utf8_text(tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")
    with pytest.raises(KnowledgeError, match="valid PDF header"):
        _extract_document(pdf_path)

    text_path = tmp_path / "broken.txt"
    text_path.write_bytes(b"\xff\xfe")
    with pytest.raises(KnowledgeError, match="UTF-8"):
        _extract_document(text_path)


def test_tfidf_retrieval_scores_actual_passages_and_index_persists(tmp_path):
    settings, _app, client, indexed = make_indexed_app(tmp_path)

    result = client.post(
        "/api/knowledge/retrieve",
        json={"question": "How does coolant flow remove battery heat?"},
    )

    assert indexed["indexing_status"] == "indexed"
    assert indexed["document_count"] == 1
    assert indexed["passage_count"] >= 2
    assert result.status_code == 200
    payload = result.get_json()
    assert payload["passages"]
    assert payload["passages"][0]["relevance_score"] > 0.05
    assert payload["passages"][0]["page"] is None
    assert payload["passages"][0]["source_url"] == "https://example.org/cooling"

    restored = KnowledgeCorpus(
        settings.knowledge_documents_path, settings.knowledge_index_path
    )
    assert restored.status()["corpus_version"] == indexed["corpus_version"]
    assert restored.retrieve("coolant flow removes heat")["passages"]


def test_empty_corpus_and_no_useful_evidence_are_honest(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()
    status = client.get("/api/knowledge/status").get_json()
    assert status["corpus"]["indexing_status"] == "not_indexed"
    assert status["bert"]["status"] == "not_configured"
    assert status["bert"]["label_set"] == []
    assert client.post("/api/knowledge/retrieve", json={"question": "thermal"}).status_code == 409

    indexed = client.post("/api/knowledge/index")
    assert indexed.get_json()["indexing_status"] == "empty"
    (settings.knowledge_documents_path / "source.md").write_text(
        "# Battery cooling\nCoolant flow removes heat from battery cells.",
        encoding="utf-8",
    )
    assert client.post("/api/knowledge/index").get_json()["indexing_status"] == "indexed"
    no_match = client.post(
        "/api/knowledge/retrieve",
        json={"question": "qzxv unrelated words"},
    )
    assert no_match.status_code == 200
    assert no_match.get_json()["passages"] == []
    assert no_match.get_json()["empty_reason"] in {
        "The query has no terms in the indexed vocabulary.",
        "No passage met the minimum TF-IDF cosine-similarity threshold (0.05).",
    }


def test_rejects_invalid_question_and_retrieval_limits(tmp_path):
    _settings, _app, client, _indexed = make_indexed_app(tmp_path)
    for body in ({}, {"question": " "}, {"question": "x" * 1001}):
        assert client.post("/api/knowledge/retrieve", json=body).status_code in {400, 413}
    for top_k in (True, 0, 9, "5"):
        assert (
            client.post(
                "/api/knowledge/retrieve",
                json={"question": "coolant battery heat", "top_k": top_k},
            ).status_code
            == 400
        )


def test_upload_validation_size_and_no_overwrite(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    unsupported = client.post(
        "/api/knowledge/upload",
        data={"file": (io.BytesIO(b"unsafe"), "notes.docx")},
        content_type="multipart/form-data",
    )
    assert unsupported.status_code == 415

    invalid_pdf = client.post(
        "/api/knowledge/upload",
        data={"file": (io.BytesIO(b"not pdf"), "notes.pdf")},
        content_type="multipart/form-data",
    )
    assert invalid_pdf.status_code == 400

    uploaded = client.post(
        "/api/knowledge/upload",
        data={"file": (io.BytesIO(b"Thermal source text."), "source.txt")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    assert uploaded.get_json()["indexing_required"] is True
    assert client.post(
        "/api/knowledge/upload",
        data={"file": (io.BytesIO(b"Other text."), "source.txt")},
        content_type="multipart/form-data",
    ).status_code == 409


def test_upload_rejects_source_urls_with_embedded_credentials(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/api/knowledge/upload",
        data={
            "file": (
                io.BytesIO(
                    b"---\ntitle: Private source\n"
                    b"source_url: https://user:password@example.org/doc\n---\n"
                    b"Thermal source."
                ),
                "private.md",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "without credentials" in response.get_json()["error"]


def test_answer_endpoint_maps_only_valid_retrieved_citations_mocked_provider(
    monkeypatch, tmp_path
):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-120b"
    )
    retrieval = client.post(
        "/api/knowledge/retrieve",
        json={"question": "How does coolant flow remove heat?"},
    ).get_json()
    passage = retrieval["passages"][0]
    captured = {}

    def mocked_provider(_settings, prompt):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "answer": "The indexed note says coolant flow removes heat from cells.",
                "citations": [passage["passage_id"]],
                "insufficient_evidence": False,
            }
        )

    monkeypatch.setattr(knowledge_answer, "generate_text", mocked_provider)
    response = client.post(
        "/api/knowledge/answer",
        json={
            "question": "How does coolant flow remove heat?",
            "passage_ids": [passage["passage_id"]],
        },
    )

    assert response.status_code == 200
    assert response.get_json()["citations"] == [passage["passage_id"]]
    assert "AI-generated answer grounded in retrieved passages" in response.get_json()["label"]
    assert "https://example.org/cooling" in captured["prompt"]
    assert passage["passage_id"] in captured["prompt"]


def test_answer_rejects_unknown_citations_and_unretrieved_passages_mocked_provider(
    monkeypatch, tmp_path
):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-120b"
    )
    retrieval = client.post(
        "/api/knowledge/retrieve",
        json={"question": "How does coolant flow remove heat?"},
    ).get_json()
    passage_id = retrieval["passages"][0]["passage_id"]
    monkeypatch.setattr(
        knowledge_answer,
        "generate_text",
        lambda *_args: json.dumps(
            {
                "answer": "Unsupported citation.",
                "citations": ["invented-passage-id"],
                "insufficient_evidence": False,
            }
        ),
    )

    invalid = client.post(
        "/api/knowledge/answer",
        json={
            "question": "How does coolant flow remove heat?",
            "passage_ids": [passage_id],
        },
    )
    not_retrieved = client.post(
        "/api/knowledge/answer",
        json={"question": "How does coolant flow remove heat?", "passage_ids": ["fake"]},
    )
    assert invalid.status_code == 502
    assert "invalid source references" in invalid.get_json()["error"]
    assert not_retrieved.status_code == 400


def test_empty_evidence_never_calls_mocked_provider(monkeypatch, tmp_path):
    settings = make_settings(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-120b"
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()
    client.post("/api/knowledge/index")
    called = False

    def unexpected_provider(*_args):
        nonlocal called
        called = True
        raise AssertionError("The provider must not be called without evidence.")

    monkeypatch.setattr(knowledge_answer, "generate_text", unexpected_provider)
    response = client.post(
        "/api/knowledge/answer",
        json={"question": "thermal", "passage_ids": []},
    )
    assert response.status_code == 400
    assert called is False


def test_groq_provider_failure_is_explicit_and_key_stays_private(
    monkeypatch, tmp_path
):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="backend-secret-test-key", model="openai/gpt-oss-120b"
    )
    passage_id = client.post(
        "/api/knowledge/retrieve",
        json={"question": "coolant flow removes heat from battery cells"},
    ).get_json()["passages"][0]["passage_id"]

    def failing_provider(*_args):
        raise GptOssServiceError("GPT-OSS request failed. No interpretation was generated.")

    monkeypatch.setattr(knowledge_answer, "generate_text", failing_provider)
    response = client.post(
        "/api/knowledge/answer",
        json={
            "question": "How does coolant remove heat?",
            "passage_ids": [passage_id],
        },
    )
    assert response.status_code == 502
    assert "No interpretation was generated" in response.get_json()["error"]
    assert "backend-secret-test-key" not in response.get_data(as_text=True)


def test_supported_gpt_oss_models_are_accepted_for_knowledge_assistant(tmp_path):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-20b"
    )
    status = client.get("/api/knowledge/status").get_json()
    assert status["gpt_oss"]["configured"] is True


def test_gpt_oss_connection_test_uses_mocked_provider_only_in_test(
    monkeypatch, tmp_path
):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-120b"
    )
    captured = {}

    def mocked_provider(settings):
        captured["model"] = settings.gpt_oss_model
        return {
            "provider": "GPT-OSS",
            "status": "connected",
            "model": settings.gpt_oss_model,
            "tested_at": "2026-10-18T00:00:00Z",
            "diagnostic": "GPT-OSS returned a non-empty inference-probe completion.",
        }

    monkeypatch.setattr(knowledge_answer, "test_gpt_oss_connection", mocked_provider)
    response = client.post("/api/knowledge/connection-test")

    assert response.status_code == 200
    result = response.get_json()
    assert result["provider"] == "GPT-OSS"
    assert result["status"] == "connected"
    assert result["tested_at"]
    assert result["diagnostic"]
    assert captured["model"] == "openai/gpt-oss-120b"


def test_gpt_oss_connection_test_returns_sanitized_provider_failure(
    monkeypatch, tmp_path
):
    _settings, _app, client, _indexed = make_indexed_app(
        tmp_path, api_key="test-only-key", model="openai/gpt-oss-120b"
    )

    def failing_provider(_settings):
        raise GptOssServiceError(
            "Provider rejected the configured API key.",
            401,
            "openai/gpt-oss-120b",
        )

    monkeypatch.setattr(knowledge_answer, "test_gpt_oss_connection", failing_provider)
    response = client.post("/api/knowledge/connection-test")

    assert response.status_code == 401
    result = response.get_json()
    assert result["provider"] == "GPT-OSS"
    assert result["model"] == "openai/gpt-oss-120b"
    assert result["status"] == "failed"
    assert result["tested_at"]
    assert result["diagnostic"] == "Provider rejected the configured API key."
    assert "test-only-key" not in response.get_data(as_text=True)


def test_bert_input_validation_runs_without_configured_model(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.post("/api/knowledge/bert/classify", json={"text": ""}).status_code == 400
    assert (
        client.post(
            "/api/knowledge/bert/classify", json={"text": "x" * 4_001}
        ).status_code
        == 413
    )
    unavailable = client.post(
        "/api/knowledge/bert/classify", json={"text": "Battery cooling"}
    )
    assert unavailable.status_code == 503
    assert "not configured" in unavailable.get_json()["error"]


def test_bert_is_not_configured_without_local_model_and_heldout_labels(tmp_path):
    settings = make_settings(tmp_path)
    classifier = BertClassifier(settings)
    assert classifier.status() == {
        "status": "not_configured",
        "label_set": [],
        "held_out_test_size": 0,
        "reason": (
            "A locally fine-tuned BERT model and its separate labeled held-out "
            "JSONL test file have not both been configured."
        ),
    }
    with pytest.raises(BertError, match="not configured"):
        classifier.classify("Cooling design question")


def test_bert_uses_configured_labels_and_heldout_metrics_with_test_model(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    model_path = tmp_path / "local-bert"
    model_path.mkdir()
    test_path = tmp_path / "heldout.jsonl"
    test_path.write_text(
        json.dumps({"text": "first category", "label": "thermal"}) + "\n"
        + json.dumps({"text": "second category", "label": "fluid"}) + "\n",
        encoding="utf-8",
    )

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()

        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": torch.tensor(
                    [[0 if "first" in text else 1] for text in texts]
                )
            }

    class FakeModel:
        config = types.SimpleNamespace(
            model_type="bert",
            num_labels=2,
            id2label={0: "thermal", 1: "fluid"},
        )

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

        @staticmethod
        def eval():
            return None

        @staticmethod
        def __call__(input_ids):
            values = input_ids[:, 0]
            logits = torch.zeros((len(values), 2))
            for index, value in enumerate(values.tolist()):
                logits[index, value] = 5
            return types.SimpleNamespace(logits=logits)

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForSequenceClassification=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    settings = make_settings(
        tmp_path,
        bert_model=model_path,
        bert_test=test_path,
    )
    classifier = BertClassifier(settings)

    assert classifier.status()["status"] == "configured"
    assert classifier.classify("first category question")["predicted_label"] == "thermal"
    evaluation = classifier.evaluate()
    assert evaluation["evaluation_split"] == "configured held-out test set"
    assert evaluation["test_example_count"] == 2
    assert evaluation["accuracy"] == 1
    assert evaluation["macro_f1"] == 1


def test_bert_rejects_generic_or_unmatched_label_sets():
    generic_model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            model_type="bert",
            num_labels=2,
            id2label={0: "LABEL_0", 1: "LABEL_1"},
        )
    )
    with pytest.raises(BertError, match="missing or generic"):
        _verified_label_set(generic_model)

    labels = ["thermal-risk", "fluid-flow"]
    with pytest.raises(BertError, match="must match"):
        _validate_test_labels(
            labels,
            [{"text": "category", "label": "thermal-risk"}],
        )

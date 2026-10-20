"""Knowledge document, retrieval, BERT, and GPT-OSS endpoints."""

from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from backend.config import Settings
from backend.services.bert_classifier import BertClassifier, BertError
from backend.services.knowledge_answer import (
    answer_from_evidence,
    test_knowledge_gpt_oss_connection,
)
from backend.services.knowledge_corpus import (
    MAX_CORPUS_BYTES,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENTS,
    SUPPORTED_SUFFIXES,
    KnowledgeCorpus,
    KnowledgeError,
)
from backend.services.gpt_oss_service import GptOssServiceError
from backend.services.gpt_oss_service import (
    PROVIDER_NAME as GPT_OSS_PROVIDER_NAME,
)

MAX_QUESTION_CHARACTERS = 1_000
MAX_RETRIEVAL_PASSAGES = 8


def _question(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise KnowledgeError("A JSON request body is required.")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise KnowledgeError("A non-empty question is required.")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise KnowledgeError(
            f"Question must be at most {MAX_QUESTION_CHARACTERS:,} characters.", 413
        )
    if any(ord(character) < 32 and character not in "\n\r\t" for character in question):
        raise KnowledgeError("Question contains unsupported control characters.")
    return question


def _safe_source_url(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeError("source_url must be a valid HTTP or HTTPS URL.")
    parsed = urlparse(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise KnowledgeError(
            "source_url must be a valid HTTP or HTTPS URL without credentials."
        )
    return value.strip()


def create_knowledge_blueprint(settings: Settings) -> Blueprint:
    blueprint = Blueprint("knowledge", __name__)
    corpus = KnowledgeCorpus(
        settings.knowledge_documents_path, settings.knowledge_index_path
    )
    bert = BertClassifier(settings)

    @blueprint.get("/status")
    def status():
        configured_model = settings.gpt_oss_model in {
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        }
        return jsonify(
            {
                "corpus": corpus.status(),
                "retrieval": {
                    "method": "TF-IDF with unigram/bigram cosine similarity",
                    "minimum_cosine_similarity": 0.05,
                    "chunk_characters": 1_200,
                    "chunk_overlap_characters": 200,
                },
                "bert": bert.status(),
                "gpt_oss": {
                    "provider": GPT_OSS_PROVIDER_NAME,
                    "model": settings.gpt_oss_model if configured_model else None,
                    "configured": bool(settings.gpt_oss_api_key and configured_model),
                    "status": (
                        "configured"
                        if settings.gpt_oss_api_key and configured_model
                        else "not_configured"
                    ),
                },
            }
        )

    @blueprint.post("/index")
    def build_index():
        try:
            result = corpus.build_index()
        except KnowledgeError as error:
            return jsonify({"error": str(error)}), error.status_code
        except OSError:
            current_app.logger.exception("Knowledge corpus indexing failed.")
            return jsonify({"error": "The knowledge index could not be saved."}), 500
        return jsonify(result), 200

    @blueprint.post("/upload")
    def upload_document():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose one PDF, TXT, or Markdown document."}), 400
        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.casefold()
        if not filename or suffix not in SUPPORTED_SUFFIXES:
            return jsonify({"error": "Only PDF, TXT, and Markdown files are supported."}), 415
        content = uploaded.stream.read(MAX_DOCUMENT_BYTES + 1)
        if len(content) > MAX_DOCUMENT_BYTES:
            return jsonify({"error": "Documents must not exceed 10 MiB."}), 413
        if not content:
            return jsonify({"error": "The uploaded document is empty."}), 400
        if suffix == ".pdf" and not content.startswith(b"%PDF-"):
            return jsonify({"error": "The uploaded file does not have a valid PDF header."}), 400
        if suffix != ".pdf":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                return jsonify({"error": "Text and Markdown uploads must use UTF-8."}), 400
            if suffix in {".md", ".markdown"}:
                metadata_line = next(
                    (
                        line.split(":", 1)[1].strip().strip("'\"")
                        for line in text.splitlines()[:20]
                        if line.strip().casefold().startswith("source_url:")
                    ),
                    None,
                )
                try:
                    _safe_source_url(metadata_line)
                except KnowledgeError as error:
                    return jsonify({"error": str(error)}), error.status_code
        documents_path = settings.knowledge_documents_path.expanduser().resolve()
        try:
            documents_path.mkdir(parents=True, exist_ok=True)
            existing_files = [
                path
                for path in documents_path.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.casefold() in SUPPORTED_SUFFIXES
            ]
            if len(existing_files) >= MAX_DOCUMENTS:
                return jsonify({"error": f"The knowledge corpus is limited to {MAX_DOCUMENTS} documents."}), 413
            total_bytes = sum(path.stat().st_size for path in existing_files)
            if total_bytes + len(content) > MAX_CORPUS_BYTES:
                return jsonify({"error": "The knowledge corpus is limited to 50 MiB."}), 413
            destination = (documents_path / filename).resolve()
            destination.relative_to(documents_path)
            with destination.open("xb") as destination_file:
                destination_file.write(content)
        except FileExistsError:
            return jsonify({"error": "A document with that filename already exists."}), 409
        except (OSError, ValueError):
            current_app.logger.exception("Knowledge document upload failed.")
            return jsonify({"error": "The document could not be saved."}), 500
        return jsonify(
            {
                "file_name": filename,
                "size_bytes": len(content),
                "indexing_required": True,
            }
        ), 201

    @blueprint.post("/retrieve")
    def retrieve():
        try:
            payload = request.get_json(silent=True)
            question = _question(payload)
            top_k = payload.get("top_k", 5)
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise KnowledgeError("top_k must be an integer from 1 to 8.")
            if not 1 <= top_k <= MAX_RETRIEVAL_PASSAGES:
                raise KnowledgeError("top_k must be an integer from 1 to 8.")
            result = corpus.retrieve(question, top_k)
        except KnowledgeError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.post("/answer")
    def answer():
        payload = request.get_json(silent=True)
        try:
            question = _question(payload)
            passage_ids = payload.get("passage_ids")
            if (
                not isinstance(passage_ids, list)
                or not 1 <= len(passage_ids) <= MAX_RETRIEVAL_PASSAGES
                or not all(isinstance(item, str) for item in passage_ids)
                or len(set(passage_ids)) != len(passage_ids)
            ):
                raise KnowledgeError("Select between 1 and 8 retrieved passage IDs.")
            passages = corpus.passages_for_answer(question, passage_ids)
            result = answer_from_evidence(settings, question, passages)
        except KnowledgeError as error:
            return jsonify({"error": str(error)}), error.status_code
        except GptOssServiceError as error:
            current_app.logger.warning("Knowledge answer request did not succeed.")
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.post("/connection-test")
    def connection_test():
        try:
            result = test_knowledge_gpt_oss_connection(settings)
        except GptOssServiceError as error:
            current_app.logger.warning("Knowledge GPT-OSS connection test did not succeed.")
            return jsonify(error.to_payload()), error.status_code
        return jsonify(result)

    @blueprint.get("/bert/status")
    def bert_status():
        return jsonify(bert.status())

    @blueprint.post("/bert/classify")
    def classify():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A JSON request body is required."}), 400
        text = payload.get("text")
        try:
            result = bert.classify(text)
        except BertError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    @blueprint.post("/bert/evaluate")
    def evaluate_bert():
        try:
            result = bert.evaluate()
        except BertError as error:
            return jsonify({"error": str(error)}), error.status_code
        return jsonify(result)

    return blueprint

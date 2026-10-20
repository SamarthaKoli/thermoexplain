"""Document extraction, deterministic chunking, and local TF-IDF retrieval."""

import hashlib
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_CORPUS_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_CHARACTERS = 2_000_000
MAX_CORPUS_CHARACTERS = 20_000_000
MAX_DOCUMENTS = 500
MAX_PASSAGES = 30_000
MAX_INDEX_BYTES = 100 * 1024 * 1024
CHUNK_CHARACTERS = 1_200
CHUNK_OVERLAP = 200
INDEX_SCHEMA_VERSION = 1


class KnowledgeError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _markdown_metadata(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        item = re.match(r"^\s*(title|source_url)\s*:\s*['\"]?(.+?)['\"]?\s*$", line, re.I)
        if item:
            metadata[item.group(1).casefold()] = item.group(2).strip()
    return metadata, text[match.end() :]


def _heading_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        underline_heading = (
            line.strip()
            if line_index + 1 < len(lines)
            and re.fullmatch(r"\s{0,3}(?:={3,}|-{3,})\s*", lines[line_index + 1])
            else None
        )
        if heading is None and underline_heading:
            heading_title = underline_heading
            line_index += 1
        else:
            heading_title = heading.group(1).strip() if heading else None
        if heading_title:
            if "\n".join(current_lines).strip():
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading_title
            current_lines = []
        else:
            current_lines.append(line)
        line_index += 1
    if "\n".join(current_lines).strip():
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections or [(None, text.strip())]


def _extract_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = path.suffix.casefold()
    content = path.read_bytes()
    if not content:
        raise KnowledgeError(f"{path.name}: document is empty.")
    digest = hashlib.sha256(content).hexdigest()
    source_url: str | None = None
    document_title = path.stem
    sections_by_page: list[tuple[int | None, str | None, str]] = []

    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise KnowledgeError(f"{path.name}: file does not contain a valid PDF header.")
        try:
            reader = PdfReader(io.BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise KnowledgeError(f"{path.name}: encrypted PDFs are not supported.")
            if reader.metadata and reader.metadata.title:
                document_title = str(reader.metadata.title).strip() or document_title
            extracted_characters = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                extracted_characters += len(text)
                if extracted_characters > MAX_DOCUMENT_CHARACTERS:
                    raise KnowledgeError(
                        f"{path.name}: extracted text exceeds the "
                        f"{MAX_DOCUMENT_CHARACTERS:,}-character limit."
                    )
                for section, body in _heading_sections(text):
                    sections_by_page.append((page_number, section, body))
        except KnowledgeError:
            raise
        except (
            PdfReadError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ) as error:
            raise KnowledgeError(f"{path.name}: PDF text extraction failed.") from error
        page_count = len(reader.pages)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise KnowledgeError(f"{path.name}: text must be UTF-8 encoded.") from error
        metadata, text = _markdown_metadata(text) if suffix in {".md", ".markdown"} else ({}, text)
        document_title = metadata.get("title", document_title)
        source_url = metadata.get("source_url")
        if source_url:
            parsed_url = urlparse(source_url)
            if (
                parsed_url.scheme not in {"http", "https"}
                or not parsed_url.netloc
                or parsed_url.username
                or parsed_url.password
            ):
                raise KnowledgeError(
                    f"{path.name}: source_url must be an HTTP or HTTPS URL without credentials."
                )
        sections_by_page.extend(
            (None, section, body) for section, body in _heading_sections(text)
        )
        page_count = None

    total_characters = sum(len(body) for _, _, body in sections_by_page)
    if total_characters > MAX_DOCUMENT_CHARACTERS:
        raise KnowledgeError(
            f"{path.name}: extracted text exceeds the {MAX_DOCUMENT_CHARACTERS:,}-character limit."
        )
    passages = _chunk_sections(
        sections_by_page,
        source_hash=digest,
        document_title=document_title,
        file_name=path.name,
        source_url=source_url,
    )
    if not passages:
        raise KnowledgeError(f"{path.name}: no extractable text was found.")
    document = {
        "document_id": digest[:16],
        "title": document_title,
        "file_name": path.name,
        "source_url": source_url,
        "page_count": page_count,
        "sha256": digest,
        "chunk_count": len(passages),
    }
    return document, passages


def _chunk_sections(
    sections: list[tuple[int | None, str | None, str]],
    *,
    source_hash: str,
    document_title: str,
    file_name: str,
    source_url: str | None,
) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for page_number, section, text in sections:
        start = 0
        while start < len(text):
            end = min(start + CHUNK_CHARACTERS, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start + CHUNK_CHARACTERS // 2, end),
                               text.rfind(" ", start + CHUNK_CHARACTERS // 2, end))
                if boundary > start:
                    end = boundary
            passage_text = text[start:end].strip()
            if passage_text:
                passage_digest = hashlib.sha256(
                    f"{source_hash}:{page_number}:{section}:{start}:{end}".encode("utf-8")
                    + passage_text.encode("utf-8")
                ).hexdigest()[:20]
                passages.append(
                    {
                        "passage_id": f"p-{passage_digest}",
                        "document_id": source_hash[:16],
                        "title": document_title,
                        "file_name": file_name,
                        "source_url": source_url,
                        "page": page_number,
                        "section": section,
                        "start_character": start,
                        "end_character": end,
                        "text": passage_text,
                    }
                )
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
            while start < len(text) and text[start].isspace():
                start += 1
    return passages


class KnowledgeCorpus:
    def __init__(self, documents_path: Path, index_path: Path):
        self.documents_path = documents_path.expanduser()
        self.index_path = index_path.expanduser()
        self._snapshot: dict[str, Any] | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load_error: str | None = None
        if self.index_path.is_file():
            try:
                self._restore()
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._load_error = "The saved knowledge index is invalid; rebuild it."
                logger.warning("Saved knowledge index could not be loaded: %s", error)

    def _fit(self, snapshot: dict[str, Any]) -> None:
        passages = snapshot["passages"]
        self._vectorizer = None
        self._matrix = None
        if passages:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=100_000,
                sublinear_tf=True,
                dtype=np.float32,
            )
            try:
                matrix = vectorizer.fit_transform(
                    [passage["text"] for passage in passages]
                )
            except ValueError as error:
                if "empty vocabulary" not in str(error):
                    raise
            else:
                self._vectorizer = vectorizer
                self._matrix = matrix
        self._snapshot = snapshot

    def _restore(self) -> None:
        if self.index_path.stat().st_size > MAX_INDEX_BYTES:
            raise ValueError("Knowledge index exceeds the 100 MiB limit.")
        snapshot = json.loads(self.index_path.read_text(encoding="utf-8"))
        if snapshot.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported knowledge index schema.")
        self._fit(snapshot)
        self._load_error = None

    def _source_document_count(self) -> int:
        if not self.documents_path.is_dir():
            return 0
        try:
            return sum(
                1
                for path in self.documents_path.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.casefold() in SUPPORTED_SUFFIXES
            )
        except OSError as error:
            raise KnowledgeError(
                "The configured knowledge-document directory could not be read.", 503
            ) from error

    def build_index(self) -> dict[str, Any]:
        if not self.documents_path.exists():
            self.documents_path.mkdir(parents=True, exist_ok=True)
        if not self.documents_path.is_dir():
            raise KnowledgeError("The configured knowledge-document path is not a directory.", 500)
        all_files = sorted(
            (
                path
                for path in self.documents_path.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.relative_to(self.documents_path).as_posix().casefold(),
        )
        candidates = [
            path for path in all_files if path.suffix.casefold() in SUPPORTED_SUFFIXES
        ]
        if len(candidates) > MAX_DOCUMENTS:
            raise KnowledgeError(f"The corpus exceeds the {MAX_DOCUMENTS}-document limit.", 413)

        documents: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []
        unsupported_count = sum(
            path.name != ".gitkeep" and path.suffix.casefold() not in SUPPORTED_SUFFIXES
            for path in all_files
        )
        warnings = (
            [f"{unsupported_count} files with unsupported formats were ignored."]
            if unsupported_count
            else []
        )
        corpus_bytes = 0
        corpus_characters = 0
        version_parts: list[str] = []
        for path in candidates:
            relative_name = path.relative_to(self.documents_path).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                warnings.append(f"{relative_name}: file metadata could not be read.")
                continue
            corpus_bytes += size
            if corpus_bytes > MAX_CORPUS_BYTES:
                raise KnowledgeError("The corpus exceeds the 50 MiB source-document limit.", 413)
            if size > MAX_DOCUMENT_BYTES:
                warnings.append(f"{relative_name}: exceeds the 10 MiB document limit.")
                continue
            try:
                document, extracted = _extract_document(path)
            except (OSError, KnowledgeError) as error:
                warnings.append(str(error))
                continue
            corpus_characters += sum(len(item["text"]) for item in extracted)
            if corpus_characters > MAX_CORPUS_CHARACTERS:
                raise KnowledgeError(
                    "The extracted corpus exceeds the 20 million character limit.", 413
                )
            document["file_name"] = relative_name
            document["document_id"] = hashlib.sha256(
                f"{relative_name}:{document['sha256']}".encode("utf-8")
            ).hexdigest()[:16]
            for passage in extracted:
                passage["file_name"] = relative_name
                passage["document_id"] = document["document_id"]
                passage["passage_id"] = "p-" + hashlib.sha256(
                    f"{relative_name}:{passage['passage_id']}".encode("utf-8")
                ).hexdigest()[:20]
            version_parts.append(f"{relative_name}:{document['sha256']}")
            documents.append(document)
            passages.extend(extracted)
            if len(passages) > MAX_PASSAGES:
                raise KnowledgeError(f"The corpus exceeds the {MAX_PASSAGES:,}-passage limit.", 413)

        version = hashlib.sha256(
            (
                f"knowledge-index-v{INDEX_SCHEMA_VERSION}:"
                f"chunk-{CHUNK_CHARACTERS}-overlap-{CHUNK_OVERLAP}:tfidf-v1\n"
                + "\n".join(version_parts)
            ).encode("utf-8")
        ).hexdigest()
        snapshot = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "corpus_version": version,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "documents": documents,
            "passages": passages,
            "warnings": warnings,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        try:
            temporary_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.index_path)
        except OSError:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
        self._fit(snapshot)
        self._load_error = None
        return self.status()

    def status(self) -> dict[str, Any]:
        source_document_count = self._source_document_count()
        if self._snapshot is None:
            return {
                "indexing_status": "error" if self._load_error else "not_indexed",
                "corpus_version": None,
                "indexed_at": None,
                "document_count": 0,
                "passage_count": 0,
                "source_document_count": source_document_count,
                "documents": [],
                "warnings": [self._load_error] if self._load_error else [],
            }
        snapshot = self._snapshot
        count = len(snapshot["passages"])
        return {
            "indexing_status": "indexed" if count else "empty",
            "corpus_version": snapshot["corpus_version"],
            "indexed_at": snapshot["indexed_at"],
            "document_count": len(snapshot["documents"]),
            "passage_count": count,
            "source_document_count": source_document_count,
            "documents": snapshot["documents"],
            "warnings": snapshot.get("warnings", []),
        }

    def retrieval_status(self) -> dict[str, str]:
        if self._load_error:
            return {
                "status": "unavailable",
                "diagnostic": "The saved knowledge index is invalid. Rebuild the index from supported source documents.",
            }
        if self._snapshot is None:
            if self._source_document_count() == 0:
                return {
                    "status": "unavailable",
                    "diagnostic": (
                        "No PDF, TXT, or Markdown source documents were found. Add genuine "
                        "sources under data/knowledge (or THERMOEXPLAIN_KNOWLEDGE_PATH), "
                        "then build the index."
                    ),
                }
            return {
                "status": "unavailable",
                "diagnostic": "Source documents are present, but no knowledge index has been built yet.",
            }
        if not self._snapshot["passages"]:
            return {
                "status": "unavailable",
                "diagnostic": "The knowledge index contains no searchable passages.",
            }
        if self._vectorizer is None or self._matrix is None:
            return {
                "status": "unavailable",
                "diagnostic": "The indexed documents produced no searchable text.",
            }
        return {
            "status": "available",
            "diagnostic": "The TF-IDF index is loaded and can retrieve passages.",
        }

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if self._snapshot is None:
            if self._load_error:
                raise KnowledgeError(self._load_error, 503)
            raise KnowledgeError("The knowledge index has not been built yet.", 409)
        passages = self._snapshot["passages"]
        if not passages or self._vectorizer is None or self._matrix is None:
            return {
                "query": query,
                "corpus_version": self._snapshot["corpus_version"],
                "passages": [],
                "empty_reason": "The indexed corpus contains no searchable text.",
            }
        query_vector = self._vectorizer.transform([query])
        if query_vector.nnz == 0:
            return {
                "query": query,
                "corpus_version": self._snapshot["corpus_version"],
                "passages": [],
                "empty_reason": "The query has no terms in the indexed vocabulary.",
            }
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), passages[index]["passage_id"]),
        )
        matches = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < 0.05:
                break
            if score <= 0:
                continue
            matches.append({**passages[index], "relevance_score": score})
            if len(matches) == top_k:
                break
        return {
            "query": query,
            "corpus_version": self._snapshot["corpus_version"],
            "passages": matches,
            "empty_reason": (
                None
                if matches
                else "No passage met the minimum TF-IDF cosine-similarity threshold (0.05)."
            ),
        }

    def passages_for_answer(
        self, query: str, passage_ids: list[str]
    ) -> list[dict[str, Any]]:
        retrieved = self.retrieve(query, top_k=10)["passages"]
        available = {passage["passage_id"]: passage for passage in retrieved}
        if not passage_ids or any(passage_id not in available for passage_id in passage_ids):
            raise KnowledgeError(
                "Select one or more passages returned by retrieval for this question."
            )
        return [available[passage_id] for passage_id in passage_ids]

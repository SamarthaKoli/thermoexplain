"""Load and evaluate a locally configured, fine-tuned BERT classifier."""

import json
import re
from pathlib import Path
from typing import Any

from backend.config import Settings

MAX_TEST_FILE_BYTES = 5 * 1024 * 1024
MAX_TEST_EXAMPLES = 2_000
MAX_CLASSIFICATION_TEXT = 4_000


class BertError(Exception):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _verified_label_set(model: Any) -> list[str]:
    if getattr(model.config, "model_type", None) != "bert":
        raise BertError("The configured sequence-classification model is not a BERT architecture.")
    id_to_label = getattr(model.config, "id2label", {})
    try:
        labels = [
            str(id_to_label[index] if index in id_to_label else id_to_label[str(index)]).strip()
            for index in range(int(model.config.num_labels))
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise BertError("The BERT model does not define a complete label set.") from error
    if (
        len(labels) < 2
        or len({label.casefold() for label in labels}) != len(labels)
        or any(
            not label.strip()
            or re.fullmatch(
                r"(?:label|class|category)[ _-]?\d+|\d+", label, re.I
            )
            for label in labels
        )
    ):
        raise BertError(
            "BERT labels are missing or generic. Configure a fine-tuned model with "
            "an explicit, verified category label for every output."
        )
    return labels


def _validate_test_labels(
    labels: list[str], test_examples: list[dict[str, str]]
) -> None:
    observed_labels = {item["label"] for item in test_examples}
    if observed_labels != set(labels):
        raise BertError(
            "The held-out test labels must match the BERT model's configured label set "
            "exactly and include every category."
        )


class BertClassifier:
    def __init__(self, settings: Settings):
        self.model_path = settings.bert_model_path
        self.test_data_path = settings.bert_test_data_path
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._labels: list[str] = []
        self._test_examples: list[dict[str, str]] = []

    def status(self) -> dict[str, Any]:
        if self.model_path is None or self.test_data_path is None:
            return {
                "status": "not_configured",
                "label_set": [],
                "held_out_test_size": 0,
                "reason": (
                    "A locally fine-tuned BERT model and its separate labeled held-out "
                    "JSONL test file have not both been configured."
                ),
            }
        try:
            self._load()
        except BertError as error:
            return {
                "status": "not_configured",
                "label_set": [],
                "held_out_test_size": 0,
                "reason": str(error),
            }
        return {
            "status": "configured",
            "label_set": self._labels,
            "held_out_test_size": len(self._test_examples),
            "reason": None,
        }

    def classify(self, text: str) -> dict[str, Any]:
        self._validate_text(text)
        self._load()
        probabilities = self._predict([text])[0]
        predicted_index = int(probabilities.argmax().item())
        return {
            "classifier": "BERT",
            "predicted_label": self._labels[predicted_index],
            "confidence": float(probabilities[predicted_index].item()),
            "confidence_method": "softmax score; not calibrated",
            "label_set": self._labels,
            "status": "inference",
        }

    def evaluate(self) -> dict[str, Any]:
        self._load()
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

        texts = [item["text"] for item in self._test_examples]
        expected = [self._labels.index(item["label"]) for item in self._test_examples]
        probabilities = self._predict(texts)
        predicted = probabilities.argmax(dim=1).tolist()
        return {
            "evaluation_split": "configured held-out test set",
            "test_example_count": len(expected),
            "label_set": self._labels,
            "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(
                expected, predicted, labels=list(range(len(self._labels)))
            ).tolist(),
        }

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise BertError("Text to classify is required.", 400)
        if len(text) > MAX_CLASSIFICATION_TEXT:
            raise BertError(
                f"Text to classify must be at most {MAX_CLASSIFICATION_TEXT:,} characters.",
                413,
            )

    def _read_test_examples(self) -> list[dict[str, str]]:
        if self.test_data_path is None:
            raise BertError("The held-out BERT test file is not configured.")
        path = self.test_data_path.expanduser()
        try:
            if not path.is_file():
                raise BertError("The configured held-out BERT test file does not exist.")
            if path.stat().st_size > MAX_TEST_FILE_BYTES:
                raise BertError("The held-out BERT test file exceeds the 5 MiB limit.", 413)
            examples: list[dict[str, str]] = []
            with path.open(encoding="utf-8") as test_file:
                for line_number, line in enumerate(test_file, start=1):
                    if not line.strip():
                        continue
                    if len(examples) >= MAX_TEST_EXAMPLES:
                        raise BertError(
                            f"The held-out test set exceeds {MAX_TEST_EXAMPLES:,} examples.",
                            413,
                        )
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise BertError(
                            f"Held-out test file line {line_number} is not valid JSON."
                        ) from error
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("text"), str)
                        or not row["text"].strip()
                        or len(row["text"]) > MAX_CLASSIFICATION_TEXT
                        or not isinstance(row.get("label"), str)
                        or not row["label"].strip()
                    ):
                        raise BertError(
                            f"Held-out test file line {line_number} must contain non-empty "
                            "'text' and 'label' strings."
                        )
                    examples.append({"text": row["text"], "label": row["label"]})
        except (OSError, UnicodeDecodeError) as error:
            raise BertError("The held-out BERT test file could not be read.") from error
        if not examples:
            raise BertError("The held-out BERT test file contains no labeled examples.")
        return examples

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None or self.test_data_path is None:
            raise BertError(
                "BERT is not configured. Provide a local fine-tuned BERT model and a "
                "separate held-out JSONL test set."
            )
        model_path = self.model_path.expanduser()
        if not model_path.is_dir():
            raise BertError("The configured local BERT model directory does not exist.")
        test_path = self.test_data_path.expanduser()
        try:
            test_path.resolve().relative_to(model_path.resolve())
        except ValueError:
            pass
        else:
            raise BertError(
                "Keep the held-out BERT test file outside the model directory."
            )
        test_examples = self._read_test_examples()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise BertError(
                "BERT inference requires the optional dependencies; install "
                "requirements-bert.txt in the backend environment."
            ) from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise BertError("The configured local BERT model could not be loaded.") from error
        labels = _verified_label_set(model)
        _validate_test_labels(labels, test_examples)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._labels = labels
        self._test_examples = test_examples

    def _predict(self, texts: list[str]):
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise BertError("The configured BERT model is not loaded.")
        batches = []
        with self._torch.no_grad():
            for offset in range(0, len(texts), 16):
                encoded = self._tokenizer(
                    texts[offset : offset + 16],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = self._model(**encoded).logits
                batches.append(self._torch.softmax(logits, dim=1))
        return self._torch.cat(batches, dim=0)

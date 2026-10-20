"""Exploratory, spatially validated surrogate training and explanation."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.services.surrogate_data import FEATURE_NAMES, SurrogateDataError


MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,95}$")
SPATIAL_GRID_BINS = 8
SPATIAL_TRAIN_MAX_BIN = 3
SPATIAL_VALIDATION_MIN_BIN = 6
MAX_SHAP_GLOBAL_SAMPLES = 200
MAX_EXPLANATION_BACKGROUND_SAMPLES = 2_000
MODEL_CONFIG = {
    "n_estimators": 120,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "n_jobs": -1,
    "random_state": 42,
}


class SurrogateError(Exception):
    """A model artifact, feature matrix, prediction, or explanation is invalid."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_feature_matrix(
    features: Any, *, expected_feature_count: int = len(FEATURE_NAMES)
) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SurrogateError("Prediction features must be numeric values.") from error
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != expected_feature_count:
        raise SurrogateError(
            f"Expected {expected_feature_count} numeric prediction features."
        )
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise SurrogateError("Prediction features must be finite numeric values.")
    return matrix


def create_spatial_holdout_split(
    normalized_coordinates: Any,
    *,
    bins: int = SPATIAL_GRID_BINS,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    """Hold out a spatial slab, with two bins excluded as a neighbor buffer."""
    normalized = np.asarray(normalized_coordinates, dtype=np.float64)
    if (
        normalized.ndim != 2
        or normalized.shape[1] != 3
        or normalized.shape[0] < 10
        or not np.isfinite(normalized).all()
        or np.any(normalized < 0)
        or np.any(normalized > 1)
    ):
        raise SurrogateError("Normalized cell-center coordinates are invalid.")
    if bins < 4:
        raise SurrogateError("At least four spatial bins are required for holdout.")

    spatial_blocks = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    occupied_by_axis = [
        int(np.unique(spatial_blocks[:, axis]).size) for axis in range(3)
    ]
    split_axis = max(range(3), key=lambda axis: (occupied_by_axis[axis], -axis))
    selected_blocks = spatial_blocks[:, split_axis]
    train_indexes = np.flatnonzero(selected_blocks <= SPATIAL_TRAIN_MAX_BIN)
    validation_indexes = np.flatnonzero(
        selected_blocks >= SPATIAL_VALIDATION_MIN_BIN
    )
    if train_indexes.size < 10 or validation_indexes.size < 10:
        raise SurrogateError(
            "The sampled cases do not populate both spatial holdout regions sufficiently."
        )
    train_bins = set(selected_blocks[train_indexes].tolist())
    validation_bins = set(selected_blocks[validation_indexes].tolist())
    if train_bins.intersection(validation_bins):
        raise SurrogateError("Spatial blocks overlap between training and validation.")
    if min(validation_bins) - max(train_bins) < 2:
        raise SurrogateError("The spatial validation buffer is not wide enough.")
    return (
        train_indexes,
        validation_indexes,
        {
            "method": "spatial_block_holdout_with_two_bin_buffer",
            "axis": ("x", "y", "z")[split_axis],
            "grid_bins": bins,
            "training_bins": sorted(train_bins),
            "validation_bins": sorted(validation_bins),
            "buffer_bins": [
                index
                for index in range(bins)
                if index not in train_bins and index not in validation_bins
            ],
            "minimum_bin_gap": min(validation_bins) - max(train_bins),
            "occupied_bins_by_axis": occupied_by_axis,
        },
    )


def _metrics(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> dict[str, float | None]:
    variance = float(np.var(actual))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if variance > 0 else None,
    }


def _shap_matrix(explainer: Any, features: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(explainer.shap_values(features), dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.shape != features.shape or not np.isfinite(values).all():
        raise SurrogateError("SHAP returned an invalid feature-contribution matrix.")
    return values


def _explainer(model: Any, background: NDArray[np.float64]) -> Any:
    try:
        import shap
    except ImportError as error:
        raise SurrogateError("Install the declared SHAP dependency to explain this model.") from error
    return shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _store_artifact(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".joblib", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        joblib.dump(bundle, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def fit_surrogate_from_samples(
    samples: dict[str, Any],
    *,
    model_directory: Path,
    model_id: str | None = None,
    manifest_sha256: str = "synthetic-test-fixture",
) -> dict[str, Any]:
    """Fit a baseline and Extra Trees model, evaluate only on buffered spatial holdout."""
    if model_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_id = f"coolant_t_extratrees_{timestamp}".lower()
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise SurrogateError("Model ID must contain only lowercase letters, digits, '_' or '-'.")
    X = validate_feature_matrix(samples.get("features"))
    try:
        y = np.asarray(samples.get("target_values"), dtype=np.float64)
        normalized = np.asarray(samples.get("normalized_coordinates"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SurrogateError("Training target values or coordinates are invalid.") from error
    case_ids = np.asarray(samples.get("case_ids"), dtype=str)
    if (
        y.shape != (X.shape[0],)
        or case_ids.shape != (X.shape[0],)
        or normalized.shape != (X.shape[0], 3)
    ):
        raise SurrogateError(
            "Target values, normalized coordinates, and case IDs must match the feature row count."
        )
    if not np.isfinite(y).all() or len(np.unique(case_ids)) < 2:
        raise SurrogateError("Training requires finite targets from at least two cases.")
    train_indexes, validation_indexes, split = create_spatial_holdout_split(normalized)
    X_train, y_train = X[train_indexes], y[train_indexes]
    X_validation, y_validation = X[validation_indexes], y[validation_indexes]
    training_cases = sorted(set(case_ids[train_indexes].tolist()))
    validation_cases = sorted(set(case_ids[validation_indexes].tolist()))
    if not training_cases or not validation_cases:
        raise SurrogateError("The spatial split must include cases in both data partitions.")

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_metrics = _metrics(y_validation, baseline.predict(X_validation))

    model = ExtraTreesRegressor(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    model_metrics = _metrics(y_validation, model.predict(X_validation))

    rng = np.random.default_rng(MODEL_CONFIG["random_state"])
    background_indexes = rng.choice(
        len(X_train),
        size=min(MAX_EXPLANATION_BACKGROUND_SAMPLES, len(X_train)),
        replace=False,
    )
    background = X_train[background_indexes]
    shap_indexes = np.linspace(
        0,
        len(X_validation) - 1,
        num=min(MAX_SHAP_GLOBAL_SAMPLES, len(X_validation)),
        dtype=np.int64,
    )
    shap_validation = X_validation[shap_indexes]
    explainer = _explainer(model, background)
    shap_values = _shap_matrix(explainer, shap_validation)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    shap_predictions = model.predict(shap_validation)
    reconstructed = base_value + shap_values.sum(axis=1)
    errors = np.abs(shap_predictions - reconstructed)
    tolerance = np.maximum(1e-5, np.abs(shap_predictions) * 1e-6)
    additivity_passed = bool(np.all(errors <= tolerance))
    if not additivity_passed:
        raise SurrogateError(
            "SHAP contributions did not reconstruct model predictions within tolerance."
        )
    global_importance = [
        {
            "feature": feature,
            "mean_absolute_shap": float(np.mean(np.abs(shap_values[:, index]))),
            "mean_shap": float(np.mean(shap_values[:, index])),
        }
        for index, feature in enumerate(FEATURE_NAMES)
    ]
    global_importance.sort(
        key=lambda item: item["mean_absolute_shap"], reverse=True
    )

    case_metadata = samples["case_metadata"]
    model_metadata = {
        "schema_version": 1,
        "model_id": model_id,
        "model_name": "Extra Trees spatial surrogate",
        "target": {
            "region": samples["target"]["region"],
            "field": samples["target"]["field"],
            "units": samples["target"]["units"],
            "field_class": "volScalarField",
        },
        "features": [
            {"name": name, "units": None if name in ("x", "y", "z") else "dimensionless"}
            for name in FEATURE_NAMES
        ],
        "coordinate_units": samples["coordinate_units"],
        "coordinate_method": samples["coordinate_method"],
        "cases": case_metadata,
        "dataset": {
            "manifest_sha256": manifest_sha256,
            "manifest_generated_at": samples.get("manifest_generated_at"),
            "case_count": len(case_metadata),
            "reynolds_numbers": sorted(
                {case["reynolds_number"] for case in case_metadata}
            ),
            "mesh_resolutions": sorted(
                {case["mesh_resolution"] for case in case_metadata}
            ),
            "single_geometry": True,
            "exploratory": True,
        },
        "sampling": samples["sampling"],
        "split": {
            **split,
            "training_sample_count": int(len(train_indexes)),
            "validation_sample_count": int(len(validation_indexes)),
            "excluded_buffer_sample_count": int(
                X.shape[0] - len(train_indexes) - len(validation_indexes)
            ),
            "training_cases": training_cases,
            "validation_cases": validation_cases,
            "validation_kind": "spatial holdout; not independent-case or new-design validation",
        },
        "metrics": {
            "baseline_mean_predictor": baseline_metrics,
            "extra_trees_spatial_validation": model_metrics,
            "metric_scope": "held-out spatial validation samples only; not training performance",
            "validation_sample_count": int(len(validation_indexes)),
        },
        "model_config": MODEL_CONFIG,
        "explainability": {
            "shap_global": {
                "method": "TreeExplainer with interventional training background",
                "baseline_definition": "Mean model output over the stored training background sample.",
                "sample_count": int(len(shap_validation)),
                "additivity_checked": True,
                "max_absolute_additivity_error": float(errors.max(initial=0.0)),
                "features": global_importance,
            },
            "lime": {
                "method": "LimeTabularExplainer with verified Reynolds values treated as categories",
                "approximation_quality_metric": "local R-squared",
            },
        },
        "limitations": [
            "Exploratory surrogate for one battery-pack geometry and the scanned QCFD cases only.",
            "Only the two observed Reynolds numbers and three observed mesh resolutions are represented.",
            "Spatial validation is held out by coordinate blocks with a buffer; it does not establish generalization to new cases or designs.",
            "Coordinate values have no verified physical length unit in the scanned metadata.",
            "The selected numeric solution times are the latest common outputs; solver convergence was not independently confirmed.",
            "Predictions approximate stored CFD fields and are not new simulations or validated engineering conclusions.",
        ],
    }
    artifact_path = model_directory / f"{model_id}.joblib"
    metadata_path = model_directory / f"{model_id}.json"
    if artifact_path.exists() or metadata_path.exists():
        raise SurrogateError(f"Model ID '{model_id}' already exists; choose a new ID.")
    _store_artifact(
        artifact_path,
        {
            "model": model,
            "lime_background": background,
            "shap_background": background,
            "feature_names": list(FEATURE_NAMES),
            "reynolds_values": sorted(
                {int(case["reynolds_number"]) for case in case_metadata}
            ),
        },
    )
    try:
        _atomic_json(metadata_path, model_metadata)
    except OSError:
        artifact_path.unlink(missing_ok=True)
        raise
    return model_metadata


def list_saved_models(model_directory: Path) -> list[dict[str, Any]]:
    if not model_directory.exists():
        return []
    models = []
    for path in sorted(model_directory.glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SurrogateError("A saved model metadata file cannot be read.") from error
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("model_id"), str)
            or not MODEL_ID_PATTERN.fullmatch(metadata["model_id"])
            or not (model_directory / f"{metadata['model_id']}.joblib").is_file()
        ):
            continue
        models.append(metadata)
    return models


def load_model_bundle(model_directory: Path, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
        raise SurrogateError("The selected model ID is invalid.")
    base_directory = model_directory.expanduser().resolve()
    metadata_path = (base_directory / f"{model_id}.json").resolve()
    artifact_path = (base_directory / f"{model_id}.joblib").resolve()
    if metadata_path.parent != base_directory or artifact_path.parent != base_directory:
        raise SurrogateError("The selected model is outside the configured model directory.")
    if not metadata_path.is_file() or not artifact_path.is_file():
        raise SurrogateError("The selected trained model is unavailable.", 404)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bundle = joblib.load(artifact_path)
    except (OSError, json.JSONDecodeError, ValueError, EOFError) as error:
        raise SurrogateError("The selected trained model could not be loaded.") from error
    if (
        not isinstance(bundle, dict)
        or bundle.get("feature_names") != list(FEATURE_NAMES)
        or not hasattr(bundle.get("model"), "predict")
        or getattr(bundle["model"], "n_features_in_", None) != len(FEATURE_NAMES)
        or metadata.get("model_id") != model_id
    ):
        raise SurrogateError("The saved model artifact has an unsupported format.")
    return bundle, metadata


def predict_one(bundle: dict[str, Any], features: Any) -> float:
    matrix = validate_feature_matrix(features)
    if matrix.shape[0] != 1:
        raise SurrogateError("Exactly one prediction input is required.")
    prediction = float(bundle["model"].predict(matrix)[0])
    if not np.isfinite(prediction):
        raise SurrogateError("The trained model returned a non-finite prediction.")
    return prediction


def explain_shap_local(
    bundle: dict[str, Any], features: Any
) -> dict[str, Any]:
    matrix = validate_feature_matrix(features)
    if matrix.shape[0] != 1:
        raise SurrogateError("Exactly one local explanation input is required.")
    explainer = _explainer(bundle["model"], bundle["shap_background"])
    values = _shap_matrix(explainer, matrix)[0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    prediction = predict_one(bundle, matrix)
    reconstruction = base_value + float(values.sum())
    additivity_error = abs(prediction - reconstruction)
    tolerance = max(1e-5, abs(prediction) * 1e-6)
    return {
        "method": "SHAP TreeExplainer",
        "prediction": prediction,
        "baseline": base_value,
        "baseline_definition": "Mean model output over the stored training background sample.",
        "contributions": [
            {
                "feature": name,
                "value": float(matrix[0, index]),
                "contribution": float(values[index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        ],
        "reconstructed_prediction": reconstruction,
        "additivity_error": additivity_error,
        "additivity_checked": True,
        "additivity_passed": additivity_error <= tolerance,
    }


def explain_lime_local(
    bundle: dict[str, Any],
    features: Any,
    *,
    seed: int = 42,
    num_samples: int = 2_000,
) -> dict[str, Any]:
    matrix = validate_feature_matrix(features)
    if matrix.shape[0] != 1:
        raise SurrogateError("Exactly one local explanation input is required.")
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as error:
        raise SurrogateError("Install the declared LIME dependency to explain this prediction.") from error

    reynolds_values = np.asarray(bundle["reynolds_values"], dtype=np.float64)
    if not np.any(np.isclose(reynolds_values, matrix[0, 3])):
        raise SurrogateError("LIME supports only verified Reynolds numbers in this model.")
    background = np.asarray(bundle["lime_background"], dtype=np.float64)
    category_codes = np.searchsorted(reynolds_values, background[:, 3])
    encoded_background = background.copy()
    encoded_background[:, 3] = category_codes
    encoded_instance = matrix[0].copy()
    encoded_instance[3] = np.searchsorted(reynolds_values, matrix[0, 3])

    def predict_encoded(encoded: NDArray[np.float64]) -> NDArray[np.float64]:
        decoded = np.asarray(encoded, dtype=np.float64).copy()
        codes = np.rint(decoded[:, 3]).astype(int)
        if np.any(codes < 0) or np.any(codes >= len(reynolds_values)):
            raise SurrogateError("LIME generated an unsupported Reynolds category.")
        decoded[:, 3] = reynolds_values[codes]
        return bundle["model"].predict(decoded)

    explainer = LimeTabularExplainer(
        training_data=encoded_background,
        feature_names=list(FEATURE_NAMES),
        categorical_features=[3],
        categorical_names={3: [str(int(value)) for value in reynolds_values]},
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        encoded_instance,
        predict_encoded,
        num_features=len(FEATURE_NAMES),
        num_samples=num_samples,
        labels=(1,),
    )
    weights = explanation.local_exp.get(1)
    if not weights:
        raise SurrogateError("LIME did not produce local feature contributions.")
    model_prediction = predict_one(bundle, matrix)
    local_prediction = float(np.asarray(explanation.local_pred).reshape(-1)[0])
    return {
        "method": "LIME Tabular local surrogate",
        "sample": {
            name: float(matrix[0, index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "prediction": model_prediction,
        "local_prediction": local_prediction,
        "intercept": float(explanation.intercept[1]),
        "local_r2": float(explanation.score),
        "local_approximation_absolute_error": abs(local_prediction - model_prediction),
        "num_samples": num_samples,
        "contributions": [
            {
                "feature": FEATURE_NAMES[index],
                "value": float(matrix[0, index]),
                "contribution": float(weight),
            }
            for index, weight in weights
        ],
        "caveat": "LIME is a local approximation and does not establish physical causality.",
    }


def manifest_digest(manifest_path: Path) -> str:
    try:
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as error:
        raise SurrogateError("The scanned dataset manifest could not be read.") from error

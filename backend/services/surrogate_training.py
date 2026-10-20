"""Assemble bounded, verified case samples and train the exploratory surrogate."""

from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.surrogate import SurrogateError, fit_surrogate_from_samples, manifest_digest
from backend.services.surrogate_data import (
    DEFAULT_FIELD,
    DEFAULT_MAX_SAMPLES_PER_CASE,
    DEFAULT_REGION,
    SurrogateDataError,
    collect_verified_samples,
)


def train_from_dataset(
    *,
    dataset_path: Path,
    manifest_path: Path,
    model_directory: Path,
    max_samples_per_case: int = DEFAULT_MAX_SAMPLES_PER_CASE,
    seed: int = 42,
    region: str = DEFAULT_REGION,
    field_name: str = DEFAULT_FIELD,
    model_id: str | None = None,
) -> dict[str, Any]:
    collected = collect_verified_samples(
        dataset_path,
        manifest_path,
        max_samples_per_case=max_samples_per_case,
        seed=seed,
        region=region,
        field_name=field_name,
    )
    features = []
    target_values = []
    normalized_coordinates = []
    case_ids = []
    case_metadata = []
    for case in collected["cases"]:
        coordinates = case["coordinates"]
        reynolds = np.full((len(coordinates), 1), case["reynolds_number"])
        features.append(np.column_stack((coordinates, reynolds)))
        target_values.append(case["values"])
        bounds = np.asarray(case["bounds"], dtype=np.float64)
        widths = bounds[:, 1] - bounds[:, 0]
        if np.any(widths <= 0):
            raise SurrogateDataError(
                f"Case {case['case_id']} does not have three-dimensional coordinate extent."
            )
        normalized_coordinates.append(
            np.clip((coordinates - bounds[:, 0]) / widths, 0.0, 1.0)
        )
        case_ids.extend([case["case_id"]] * len(coordinates))
        case_metadata.append(
            {
                "case_id": case["case_id"],
                "reynolds_number": case["reynolds_number"],
                "mesh_resolution": case["mesh_resolution"],
                "solution_time": case["solution_time"],
                "cell_count": case["cell_count"],
                "sample_count": case["sample_count"],
                "bounds": case["bounds"],
            }
        )
    samples = {
        "features": np.concatenate(features),
        "target_values": np.concatenate(target_values),
        "normalized_coordinates": np.concatenate(normalized_coordinates),
        "case_ids": np.asarray(case_ids),
        "case_metadata": case_metadata,
        "target": {
            "region": collected["region"],
            "field": collected["field"],
            "units": collected["target_units"],
        },
        "coordinate_units": collected["coordinate_units"],
        "coordinate_method": collected["coordinate_method"],
        "manifest_generated_at": collected["dataset_version"]["manifest_generated_at"],
        "sampling": {
            "method": "seeded random sampling without replacement within each case",
            "seed": seed,
            "maximum_per_case": max_samples_per_case,
            "sample_count_by_case": {
                case["case_id"]: case["sample_count"] for case in collected["cases"]
            },
            "source_cell_count_by_case": {
                case["case_id"]: case["cell_count"] for case in collected["cases"]
            },
            "target_field_values_are_cell_aligned": True,
        },
    }
    return fit_surrogate_from_samples(
        samples,
        model_directory=model_directory,
        model_id=model_id,
        manifest_sha256=manifest_digest(manifest_path),
    )

"""Train an exploratory surrogate from the currently scanned CFD dataset."""

import argparse
import json
import sys
from pathlib import Path

from backend.config import Settings
from backend.ml.surrogate import SurrogateError
from backend.services.surrogate_data import (
    DEFAULT_FIELD,
    DEFAULT_MAX_SAMPLES_PER_CASE,
    DEFAULT_REGION,
    SurrogateDataError,
)
from backend.services.surrogate_training import train_from_dataset


def parse_args() -> argparse.Namespace:
    settings = Settings.from_environment()
    parser = argparse.ArgumentParser(
        description="Train a spatially validated exploratory CFD field surrogate."
    )
    parser.add_argument("--source", type=Path, default=settings.dataset_path)
    parser.add_argument("--manifest", type=Path, default=settings.manifest_path)
    parser.add_argument("--model-directory", type=Path, default=settings.models_path)
    parser.add_argument(
        "--max-samples-per-case",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_CASE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--model-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = train_from_dataset(
            dataset_path=args.source,
            manifest_path=args.manifest,
            model_directory=args.model_directory,
            max_samples_per_case=args.max_samples_per_case,
            seed=args.seed,
            region=args.region,
            field_name=args.field,
            model_id=args.model_id,
        )
    except (SurrogateDataError, SurrogateError, OSError, ValueError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

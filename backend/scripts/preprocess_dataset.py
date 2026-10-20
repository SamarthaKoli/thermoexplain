"""Preprocess one supported CSV point table without changing the source."""

import argparse
from pathlib import Path

from backend.services.dataset_preprocessor import preprocess_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-id")
    parser.add_argument("--mesh-id")
    args = parser.parse_args()

    result = preprocess_csv(
        args.source,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        case_id=args.case_id,
        mesh_id=args.mesh_id,
    )
    print(
        f"Preprocessing {result['status']}: "
        f"{result['output_rows']} of {result['source_rows']} rows."
    )
    print(f"Output: {result['output_file']}")


if __name__ == "__main__":
    main()

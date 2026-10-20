"""Scan a dataset without extracting archives or modifying source files."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.services.dataset_inspector import (
    scan_dataset,
    write_audit_report,
    write_manifest,
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("THERMOEXPLAIN_DATASET_PATH", "data/raw")).expanduser(),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/reports/data-audit.md")
    )
    args = parser.parse_args()

    manifest = scan_dataset(args.source)
    write_manifest(manifest, args.manifest)
    write_audit_report(manifest, args.report)
    summary = manifest["summary"]
    print(
        f"Scan status: {manifest['status']}; "
        f"{summary['file_count']} file(s), "
        f"{summary['candidate_count']} candidate data file(s), "
        f"{len(manifest['cases'])} case(s)."
    )
    print(f"Manifest: {args.manifest}")
    print(f"Audit report: {args.report}")


if __name__ == "__main__":
    main()

"""Load the FD001 train and test trajectories into PostgreSQL."""

import json
from pathlib import Path

from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import ingest_fd001_readings
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.ingestion.fd001 import (
    load_trajectory_file,
    validate_trajectory_structure,
)
from turbofan_copilot.ingestion.raw_files import validate_expected_files
from turbofan_copilot.ingestion.source_manifest import load_source_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "nasa_cmapss_fd001.json"
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "cmapss" / "FD001"


def main() -> None:
    """Validate the raw FD001 files, then upsert train and test rows idempotently."""
    manifest = load_source_manifest(MANIFEST_PATH)
    validate_expected_files(manifest, RAW_DIRECTORY)

    train_frame = load_trajectory_file(RAW_DIRECTORY / "train_FD001.txt")
    test_frame = load_trajectory_file(RAW_DIRECTORY / "test_FD001.txt")
    validate_trajectory_structure(train_frame)
    validate_trajectory_structure(test_frame)

    with Session(get_engine()) as session:
        train_rows = ingest_fd001_readings(session, train_frame, split="train")
        test_rows = ingest_fd001_readings(session, test_frame, split="test")
        session.commit()

    report = {
        "train_rows": train_rows,
        "train_engines": int(train_frame["unit_id"].nunique()),
        "test_rows": test_rows,
        "test_engines": int(test_frame["unit_id"].nunique()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

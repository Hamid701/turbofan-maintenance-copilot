"""Presence checks for raw files declared by a source manifest."""

from pathlib import Path

from turbofan_copilot.ingestion.source_manifest import SourceManifest


def validate_expected_files(
    manifest: SourceManifest,
    raw_directory: str | Path,
) -> tuple[Path, ...]:
    """Return expected file paths or raise with all missing filenames."""
    directory = Path(raw_directory)
    expected_paths = tuple(directory / name for name in manifest.expected_files)
    missing_names = tuple(path.name for path in expected_paths if not path.is_file())

    if missing_names:
        missing_list = ", ".join(missing_names)
        raise FileNotFoundError(f"Missing expected files in {directory}: {missing_list}")

    return expected_paths

"""Tests for FD001 parsing and structural validation."""

from pathlib import Path

import pytest

from turbofan_copilot.ingestion.fd001 import (
    TRAJECTORY_COLUMNS,
    load_rul_file,
    load_trajectory_file,
    validate_rul_structure,
    validate_trajectory_structure,
)


def trajectory_row(unit_id: int, cycle: int) -> str:
    """Build one synthetic row with the FD001 26-value structure."""
    values = [
        str(unit_id),
        str(cycle),
        "0.1",
        "0.2",
        "100.0",
        *(["1.0"] * 21),
    ]
    return "  ".join(values)


def write_trajectory(path: Path, rows: list[str]) -> None:
    """Write synthetic rows with the trailing whitespace seen in NASA files."""
    path.write_text("  \n".join(rows) + "  \n", encoding="utf-8")


def test_load_trajectory_file_assigns_columns_and_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.txt"
    write_trajectory(path, [trajectory_row(1, 1), trajectory_row(1, 2)])

    frame = load_trajectory_file(path)

    assert frame.shape == (2, 26)
    assert tuple(frame.columns) == TRAJECTORY_COLUMNS
    assert str(frame["unit_id"].dtype) == "int64"
    assert str(frame["sensor_1"].dtype) == "float64"


def test_load_rul_file_returns_integer_targets(tmp_path: Path) -> None:
    path = tmp_path / "rul.txt"
    path.write_text("10 \n20 \n", encoding="utf-8")

    rul = load_rul_file(path)

    assert rul.tolist() == [10, 20]
    assert str(rul.dtype) == "int64"


def test_validate_trajectory_structure_accepts_contiguous_engines(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.txt"
    write_trajectory(
        path,
        [trajectory_row(1, 1), trajectory_row(1, 2), trajectory_row(2, 1)],
    )
    frame = load_trajectory_file(path)

    validate_trajectory_structure(frame)


def test_validate_trajectory_structure_rejects_cycle_gap(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.txt"
    write_trajectory(path, [trajectory_row(1, 1), trajectory_row(1, 3)])
    frame = load_trajectory_file(path)

    with pytest.raises(ValueError, match="Cycles for engine 1"):
        validate_trajectory_structure(frame)


def test_validate_rul_structure_rejects_target_count_mismatch(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.txt"
    write_trajectory(
        trajectory_path,
        [trajectory_row(1, 1), trajectory_row(2, 1)],
    )
    rul_path = tmp_path / "rul.txt"
    rul_path.write_text("10\n", encoding="utf-8")

    test_frame = load_trajectory_file(trajectory_path)
    rul = load_rul_file(rul_path)

    with pytest.raises(ValueError, match="Expected 2 RUL targets"):
        validate_rul_structure(test_frame, rul)

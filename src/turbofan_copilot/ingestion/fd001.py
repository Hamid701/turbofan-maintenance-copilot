"""Parsing for NASA C-MAPSS FD001 source files."""

from pathlib import Path

import pandas as pd

SETTING_COLUMNS = tuple(f"setting_{number}" for number in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{number}" for number in range(1, 22))
TRAJECTORY_COLUMNS = ("unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS)

TRAJECTORY_DTYPES = {
    "unit_id": "int64",
    "cycle": "int64",
    **{column: "float64" for column in (*SETTING_COLUMNS, *SENSOR_COLUMNS)},
}


def load_trajectory_file(path: str | Path) -> pd.DataFrame:
    """Load an FD001 train or test trajectory file without transforming values."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=TRAJECTORY_COLUMNS,
        dtype=TRAJECTORY_DTYPES,
    )


def load_rul_file(path: str | Path) -> pd.Series:
    """Load the one-column remaining-useful-life target file."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["rul"],
        dtype={"rul": "int64"},
    )
    return frame["rul"]


def validate_trajectory_structure(frame: pd.DataFrame) -> None:
    """Validate the table-level structure shared by FD001 train and test data."""
    if tuple(frame.columns) != TRAJECTORY_COLUMNS:
        raise ValueError("Trajectory columns do not match the FD001 contract")
    if frame.empty:
        raise ValueError("Trajectory data must not be empty")
    if frame.isna().any().any():
        raise ValueError("Trajectory data contains missing values")

    engine_ids = frame["unit_id"].drop_duplicates().tolist()
    expected_engine_ids = list(range(1, len(engine_ids) + 1))
    if engine_ids != expected_engine_ids:
        raise ValueError("Engine IDs must be contiguous and ordered from 1")

    for engine_id, engine_rows in frame.groupby("unit_id", sort=False):
        cycles = engine_rows["cycle"].tolist()
        expected_cycles = list(range(1, len(cycles) + 1))
        if cycles != expected_cycles:
            raise ValueError(f"Cycles for engine {engine_id} must be contiguous from 1")


def validate_rul_structure(test_frame: pd.DataFrame, rul: pd.Series) -> None:
    """Validate the relationship between test engines and their RUL targets."""
    expected_targets = test_frame["unit_id"].nunique()
    if len(rul) != expected_targets:
        raise ValueError(
            f"Expected {expected_targets} RUL targets for test engines, found {len(rul)}"
        )
    if rul.isna().any():
        raise ValueError("RUL targets contain missing values")
    if (rul < 0).any():
        raise ValueError("RUL targets must not be negative")

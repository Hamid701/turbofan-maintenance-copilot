"""Tests for the RUL evaluation protocol: grouped split, target, cases, metrics."""

import pandas as pd
import pytest

from turbofan_copilot.evaluation.rul_evaluation import (
    MIN_HISTORY_CYCLES,
    MIN_TRUE_RUL,
    evaluate_predictor,
    nasa_score,
    score_rul,
    split_engine_ids,
    validation_cases,
)
from turbofan_copilot.health.degradation import RUL_CAP
from turbofan_copilot.health.rul_features import label_train_readings, piecewise_rul


def run_to_failure(*, engines: int, cycles: int) -> pd.DataFrame:
    """Train-shaped frame: every engine runs from cycle 1 to its failure."""
    rows = [
        {"split": "train", "unit_id": unit, "cycle": cycle, "sensor_1": float(cycle)}
        for unit in range(1, engines + 1)
        for cycle in range(1, cycles + 1)
    ]
    return pd.DataFrame(rows)


def test_split_is_deterministic_disjoint_and_covers_every_engine() -> None:
    fit, validation = split_engine_ids(range(1, 101), validation_fraction=0.2, seed=7)

    assert len(fit) == 80
    assert len(validation) == 20
    assert set(fit) & set(validation) == set()
    assert sorted([*fit, *validation]) == list(range(1, 101))
    assert split_engine_ids(range(1, 101), validation_fraction=0.2, seed=7) == (fit, validation)
    assert split_engine_ids(range(1, 101), validation_fraction=0.2, seed=8) != (fit, validation)


def test_split_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError, match="validation_fraction must be between"):
        split_engine_ids(range(1, 11), validation_fraction=1.0)
    with pytest.raises(ValueError, match="at least two engines"):
        split_engine_ids([1])


def test_piecewise_target_holds_flat_at_the_cap() -> None:
    assert piecewise_rul(10) == 10
    assert piecewise_rul(RUL_CAP) == RUL_CAP
    assert piecewise_rul(400) == RUL_CAP
    with pytest.raises(ValueError, match="must not be negative"):
        piecewise_rul(-1)


def test_labels_count_down_to_zero_at_failure_and_cap_early_life() -> None:
    labelled = label_train_readings(run_to_failure(engines=1, cycles=200))

    assert labelled.loc[labelled["cycle"] == 200, "rul"].item() == 0
    assert labelled.loc[labelled["cycle"] == 199, "rul"].item() == 1
    assert labelled.loc[labelled["cycle"] == 75, "rul"].item() == RUL_CAP
    assert labelled.loc[labelled["cycle"] == 1, "rul"].item() == RUL_CAP


def test_labelling_refuses_to_overwrite_an_existing_column() -> None:
    frame = label_train_readings(run_to_failure(engines=1, cycles=10))
    with pytest.raises(ValueError, match="already carries a rul column"):
        label_train_readings(frame)


def test_validation_cases_truncate_before_failure_with_the_true_remaining_life() -> None:
    frame = run_to_failure(engines=2, cycles=200)

    cases = validation_cases(frame, [1, 2], draws_per_engine=3, seed=1)

    assert len(cases) == 6
    for history, truth in cases:
        assert MIN_TRUE_RUL <= truth <= RUL_CAP
        assert len(history) >= MIN_HISTORY_CYCLES
        # The history stops exactly `truth` cycles short of the engine's failure.
        assert int(history["cycle"].max()) == 200 - truth


def test_validation_cases_are_seeded_and_skip_engines_that_are_too_short() -> None:
    frame = run_to_failure(engines=1, cycles=200)
    first = validation_cases(frame, [1], draws_per_engine=2, seed=3)
    assert [truth for _, truth in first] == [
        truth for _, truth in validation_cases(frame, [1], draws_per_engine=2, seed=3)
    ]

    short = run_to_failure(engines=1, cycles=MIN_HISTORY_CYCLES + MIN_TRUE_RUL - 1)
    with pytest.raises(ValueError, match="no engine was long enough"):
        validation_cases(short, [1])


def test_nasa_score_punishes_late_predictions_harder_than_early_ones() -> None:
    late = nasa_score([20.0])
    early = nasa_score([-20.0])

    assert late > early
    assert nasa_score([0.0]) == 0.0


def test_score_reports_error_size_and_direction() -> None:
    score = score_rul([10.0, 10.0], [0, 20])

    assert score.cases == 2
    assert score.mae == 10.0
    assert score.rmse == 10.0
    # Equal and opposite errors cancel, so a mean error of zero means no bias.
    assert score.mean_error == 0.0

    biased = score_rul([30.0, 30.0], [20, 20])
    assert biased.mean_error == 10.0


def test_score_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_rul([1.0], [1, 2])
    with pytest.raises(ValueError, match="at least one case"):
        score_rul([], [])


def test_evaluate_predictor_scores_a_perfect_predictor_as_zero_error() -> None:
    frame = run_to_failure(engines=2, cycles=200)
    cases = validation_cases(frame, [1, 2], draws_per_engine=2, seed=5)
    truths = {int(history["cycle"].max()): truth for history, truth in cases}

    score = evaluate_predictor(cases, lambda history: float(truths[int(history["cycle"].max())]))

    assert score.mae == 0.0
    assert score.rmse == 0.0
    assert score.nasa_score == 0.0

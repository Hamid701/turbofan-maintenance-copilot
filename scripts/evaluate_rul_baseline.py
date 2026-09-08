"""Score the deterministic degradation RUL baseline against FD001 ground truth."""

import json
from math import sqrt
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import load_split_readings
from turbofan_copilot.db.models import EngineRul
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.health.degradation import (
    RUL_CAP,
    estimate_engine_rul,
    fit_degradation_model,
)

TRAIN_CUT_FRACTIONS = (0.5, 0.7, 0.9)


def error_stats(errors: list[float]) -> dict[str, float]:
    """Return count, MAE, and RMSE for a list of (estimate - truth) errors."""
    return {
        "n": len(errors),
        "mae": round(fmean(abs(error) for error in errors), 2),
        "rmse": round(sqrt(fmean(error * error for error in errors)), 2),
    }


def main() -> None:
    """Fit on train, score held-out cut points and the test targets, print MAE/RMSE."""
    with Session(get_engine()) as session:
        train = load_split_readings(session, "train")
        test = load_split_readings(session, "test")
        targets = {
            int(unit): int(rul)
            for unit, rul in session.execute(select(EngineRul.unit_id, EngineRul.rul)).all()
        }

    model = fit_degradation_model(train)

    train_errors: list[float] = []
    for unit in sorted(int(value) for value in train["unit_id"].unique()):
        engine = train[train["unit_id"] == unit]
        max_cycle = int(engine["cycle"].max())
        for fraction in TRAIN_CUT_FRACTIONS:
            cut = int(max_cycle * fraction)
            if cut < 2:
                continue
            estimate = estimate_engine_rul(engine[engine["cycle"] <= cut], model)
            true_rul = min(max_cycle - cut, RUL_CAP)
            train_errors.append(estimate.estimated_rul - true_rul)

    test_errors: list[float] = []
    clipped_targets: list[int] = []
    for unit in sorted(int(value) for value in test["unit_id"].unique()):
        estimate = estimate_engine_rul(test[test["unit_id"] == unit], model)
        true_rul = min(targets[unit], RUL_CAP)
        clipped_targets.append(true_rul)
        test_errors.append(estimate.estimated_rul - true_rul)

    mean_target = fmean(clipped_targets)
    naive_errors = [mean_target - target for target in clipped_targets]

    report = {
        "rul_cap": RUL_CAP,
        "informative_sensors": [norm.sensor for norm in model.sensors],
        "failure_index": round(model.failure_index, 4),
        "train_cut_evaluation": error_stats(train_errors),
        "test_target_evaluation": error_stats(test_errors),
        "naive_constant_mean_baseline": error_stats(naive_errors),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

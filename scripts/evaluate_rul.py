"""Score RUL predictors under one protocol: fit split, held-out validation, opt-in test.

Model selection happens on the validation split. The FD001 test targets are only
scored with ``--test``, which should be run once per finalist so the published
number stays an honest estimate of unseen-engine performance.
"""

import argparse
import json
from collections.abc import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from turbofan_copilot.db.fd001_store import load_split_readings
from turbofan_copilot.db.models import EngineRul
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.evaluation.rul_evaluation import (
    DEFAULT_DRAWS_PER_ENGINE,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    RulCase,
    RulScore,
    evaluate_predictor,
    label_train_readings,
    piecewise_rul,
    split_engine_ids,
    validation_cases,
)
from turbofan_copilot.health.degradation import (
    RUL_CAP,
    DegradationModel,
    estimate_engine_rul,
    fit_degradation_model,
)
from turbofan_copilot.health.rul_regressor import fit_boosted_rul_regressor, fit_rul_regressor

DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1000.0)

# (learning_rate, max_leaf_nodes, max_iter). A deliberately small grid: enough to
# see whether depth or the number of trees is what moves the score, few enough that
# selecting on 100 validation cases stays meaningful.
BOOSTING_GRID = (
    (0.05, 31, 400),
    (0.1, 31, 200),
    (0.05, 15, 400),
    (0.1, 63, 200),
)


def boosting_label(learning_rate: float, max_leaf_nodes: int, max_iter: int) -> str:
    """Name one boosting configuration so its score is identifiable in the report."""
    return f"boosted_lr{learning_rate:g}_leaves{max_leaf_nodes}_iter{max_iter}"


def baseline_predictor(model: DegradationModel) -> Callable[[pd.DataFrame], float]:
    """Wrap the deterministic degradation baseline as a scoreable predictor."""

    def predict(history: pd.DataFrame) -> float:
        return float(estimate_engine_rul(history, model).estimated_rul)

    return predict


def constant_predictor(value: float) -> Callable[[pd.DataFrame], float]:
    """A control that ignores the sensors and always predicts the same number."""

    def predict(history: pd.DataFrame) -> float:
        return value

    return predict


def test_cases(test_readings: pd.DataFrame, targets: dict[int, int]) -> tuple[RulCase, ...]:
    """Pair each test engine's full history with its published remaining life."""
    cases: list[RulCase] = []
    for unit in sorted(int(value) for value in test_readings["unit_id"].unique()):
        history = test_readings.loc[test_readings["unit_id"] == unit]
        cases.append((history, piecewise_rul(targets[unit], cap=RUL_CAP)))
    return tuple(cases)


def parse_args() -> argparse.Namespace:
    """Read the split, seed, and whether the test set may be touched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS_PER_ENGINE)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="ridge penalties to compare on the validation split",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="also score the held-out FD001 test targets (use once per finalist)",
    )
    return parser.parse_args()


def main() -> None:
    """Fit on the training split, score the validation split, optionally score test."""
    args = parse_args()

    with Session(get_engine()) as session:
        train = load_split_readings(session, "train")
        test = load_split_readings(session, "test") if args.test else None
        targets = (
            {
                int(unit): int(rul)
                for unit, rul in session.execute(select(EngineRul.unit_id, EngineRul.rul)).all()
            }
            if args.test
            else {}
        )

    fit_units, validation_units = split_engine_ids(
        train["unit_id"].tolist(),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    fit_readings = train.loc[train["unit_id"].isin(fit_units)]
    labelled_fit = label_train_readings(fit_readings)
    model = fit_degradation_model(fit_readings)
    cases = validation_cases(train, validation_units, draws_per_engine=args.draws, seed=args.seed)
    naive_value = float(labelled_fit["rul"].mean())

    scores: dict[str, RulScore] = {
        "degradation_baseline": evaluate_predictor(cases, baseline_predictor(model)),
        "naive_constant": evaluate_predictor(cases, constant_predictor(naive_value)),
    }
    for alpha in args.alphas:
        regressor = fit_rul_regressor(labelled_fit, alpha=alpha)
        scores[f"ridge_alpha_{alpha:g}"] = evaluate_predictor(cases, regressor.predict)
    for learning_rate, leaves, iterations in BOOSTING_GRID:
        boosted = fit_boosted_rul_regressor(
            labelled_fit,
            learning_rate=learning_rate,
            max_leaf_nodes=leaves,
            max_iter=iterations,
        )
        label = boosting_label(learning_rate, leaves, iterations)
        scores[label] = evaluate_predictor(cases, boosted.predict)

    best_alpha = min(args.alphas, key=lambda alpha: scores[f"ridge_alpha_{alpha:g}"].mae)
    best_boosting = min(BOOSTING_GRID, key=lambda config: scores[boosting_label(*config)].mae)
    report: dict[str, object] = {
        "protocol": {
            "seed": args.seed,
            "fit_engines": len(fit_units),
            "validation_engines": len(validation_units),
            "draws_per_engine": args.draws,
            "rul_cap": RUL_CAP,
            "naive_constant": round(naive_value, 2),
            "best_alpha_on_validation": best_alpha,
            "best_boosting_on_validation": boosting_label(*best_boosting),
        },
        "validation": {name: score.model_dump() for name, score in scores.items()},
    }

    if test is not None:
        labelled_train = label_train_readings(train)
        final_model = fit_degradation_model(train)
        final_regressor = fit_rul_regressor(labelled_train, alpha=best_alpha)
        final_learning_rate, final_leaves, final_iterations = best_boosting
        final_boosted = fit_boosted_rul_regressor(
            labelled_train,
            learning_rate=final_learning_rate,
            max_leaf_nodes=final_leaves,
            max_iter=final_iterations,
        )
        final_cases = test_cases(test, targets)
        test_report: dict[str, object] = {
            "note": "held-out FD001 targets; refitted on all 100 train engines",
            "degradation_baseline": evaluate_predictor(
                final_cases, baseline_predictor(final_model)
            ).model_dump(),
            "naive_constant": evaluate_predictor(
                final_cases, constant_predictor(naive_value)
            ).model_dump(),
            f"ridge_alpha_{best_alpha:g}": evaluate_predictor(
                final_cases, final_regressor.predict
            ).model_dump(),
            boosting_label(*best_boosting): evaluate_predictor(
                final_cases, final_boosted.predict
            ).model_dump(),
        }
        report["test"] = test_report

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

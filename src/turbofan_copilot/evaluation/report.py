"""A versioned, structured evaluation report.

One run of the evaluation produces one :class:`EvaluationReport`: a
:class:`RunContext` (when, which commit, which configuration) plus whatever
sections were scored - :class:`RetrievalScore` and :class:`AbstentionScore` so
far, with token/cost to come as another optional section.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.evaluation.retrieval_metrics import RanksChunks, evaluate_retriever

if TYPE_CHECKING:
    from turbofan_copilot.evaluation.pipeline_cases import PipelineCase
    from turbofan_copilot.llm.answer import GroundedAnswer


class GitRevision(BaseModel):
    """The commit a report was produced from, and whether the tree was dirty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str
    dirty: bool


def git_revision() -> GitRevision | None:
    """Return the current commit, or ``None`` when git is unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return GitRevision(sha=sha, dirty=bool(status.strip()))


class RunContext(BaseModel):
    """Everything needed to know which system produced a report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: datetime
    git: GitRevision | None
    config: dict[str, str]


class RetrievalScore(BaseModel):
    """Page-grounded retrieval metrics over one evaluation set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mean_reciprocal_rank: float


class AbstentionScore(BaseModel):
    """How well the pipeline answered-or-abstained over the labelled cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: int
    accuracy: float
    answered_when_should_abstain: tuple[str, ...]
    abstained_when_should_answer: tuple[str, ...]
    errors: tuple[str, ...] = ()


class TimingScore(BaseModel):
    """Wall-clock seconds spent scoring each section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval_seconds: float
    abstention_seconds: float | None = None


class CostScore(BaseModel):
    """Token totals and a rough dollar estimate for the LLM calls in a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int
    prompt_tokens: int
    completion_tokens: int
    usd_estimate: float | None


class EvaluationReport(BaseModel):
    """One evaluation run: its context and every section that was scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunContext
    retrieval: RetrievalScore | None = None
    abstention: AbstentionScore | None = None
    cost: CostScore | None = None
    timing: TimingScore | None = None


def new_run_context(config: dict[str, str]) -> RunContext:
    """Stamp a run with the current time (UTC) and commit."""
    return RunContext(created_at=datetime.now(UTC), git=git_revision(), config=config)


def score_retrieval(
    cases: Iterable[RetrievalEvaluationCase],
    retriever: RanksChunks,
) -> RetrievalScore:
    """Score any retriever over the page-grounded cases."""
    case_list = tuple(cases)
    metrics = evaluate_retriever(case_list, retriever)
    return RetrievalScore(
        cases=len(case_list),
        hit_at_1=metrics["hit_at_1"],
        hit_at_3=metrics["hit_at_3"],
        hit_at_5=metrics["hit_at_5"],
        mean_reciprocal_rank=metrics["mean_reciprocal_rank"],
    )


def score_abstention(
    cases: Iterable[PipelineCase],
    answer: Callable[[str], GroundedAnswer],
) -> AbstentionScore:
    """Run every case through ``answer`` and score the answer-or-abstain decision.

    ``answer`` is anything that maps a question to a ``GroundedAnswer`` - in
    practice ``build_answer_chain(...).invoke``.
    """
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("abstention evaluation needs at least one case")

    outcomes: list[tuple[str, bool, bool]] = []
    errors: list[str] = []
    for case in case_list:
        try:
            abstained = answer(case.question).abstained
        except Exception:  # an eval run records failures rather than aborting
            errors.append(case.case_id)
            continue
        outcomes.append((case.case_id, case.should_abstain, abstained))

    correct = sum(should == did for _, should, did in outcomes)
    return AbstentionScore(
        cases=len(case_list),
        accuracy=correct / len(outcomes) if outcomes else 0.0,
        answered_when_should_abstain=tuple(
            case_id for case_id, should, did in outcomes if should and not did
        ),
        abstained_when_should_answer=tuple(
            case_id for case_id, should, did in outcomes if not should and did
        ),
        errors=tuple(errors),
    )


def write_report(report: EvaluationReport, directory: str | Path) -> Path:
    """Write the report as pretty JSON named by its run timestamp; return the path."""
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    stamp = report.run.created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target_directory / f"evaluation-{stamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


class MetricDelta(BaseModel):
    """One metric's baseline value, candidate value, and whether it got worse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    regressed: bool


class ReportDiff(BaseModel):
    """Every comparable metric between two reports, and an overall verdict.

    ``warnings`` names anything that makes the two runs less than like-for-like -
    a different retriever, a different evaluation-set size - so a meaningless
    comparison is visible rather than silent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    deltas: tuple[MetricDelta, ...]
    warnings: tuple[str, ...] = ()
    regressed: bool


_SECTIONS = ("retrieval", "abstention", "cost", "timing")

# Fully-qualified metrics where a lower candidate value is a regression. Cost and
# timing are reported but never gate: a prompt change may legitimately cost more
# or take longer, and that is a judgement call, not a failure.
_HIGHER_IS_BETTER = frozenset(
    {
        "retrieval.hit_at_1",
        "retrieval.hit_at_3",
        "retrieval.hit_at_5",
        "retrieval.mean_reciprocal_rank",
        "abstention.accuracy",
    }
)

# Retrieval scoring is deterministic, so any drop is real and gates at zero.
# Abstention accuracy comes from non-deterministic model calls over ~10 cases,
# where one case flipping moves the metric by 0.1 with nothing actually wrong.
DEFAULT_TOLERANCES: dict[str, float] = {"abstention.accuracy": 0.15}


def _comparable_metrics(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> list[tuple[str, float | None, float | None]]:
    """Pull (name, baseline, candidate) for every numeric metric worth comparing."""
    rows: list[tuple[str, float | None, float | None]] = []
    for section in _SECTIONS:
        base_section = getattr(baseline, section)
        cand_section = getattr(candidate, section)
        present = base_section if base_section is not None else cand_section
        if present is None:
            continue
        for field in type(present).model_fields:
            if field == "cases":  # a count of the eval set, not a metric
                continue
            base_value = getattr(base_section, field, None)
            cand_value = getattr(cand_section, field, None)
            sample = base_value if base_value is not None else cand_value
            if not isinstance(sample, int | float):
                continue  # skip the lists of failing case ids
            rows.append((f"{section}.{field}", base_value, cand_value))
    return rows


def comparability_warnings(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> tuple[str, ...]:
    """Name every reason these two runs are not a like-for-like comparison."""
    warnings: list[str] = []
    for section in _SECTIONS:
        base_section = getattr(baseline, section)
        cand_section = getattr(candidate, section)
        if base_section is None or cand_section is None:
            continue
        base_cases = getattr(base_section, "cases", None)
        cand_cases = getattr(cand_section, "cases", None)
        if base_cases != cand_cases:
            warnings.append(
                f"{section}: {base_cases} cases in the baseline vs {cand_cases} in the candidate"
            )

    base_config, cand_config = baseline.run.config, candidate.run.config
    for key in sorted(base_config.keys() & cand_config.keys()):
        if base_config[key] != cand_config[key]:
            warnings.append(f"config {key}: {base_config[key]!r} -> {cand_config[key]!r}")
    return tuple(warnings)


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    tolerances: Mapping[str, float] | None = None,
) -> ReportDiff:
    """Diff two reports metric by metric and flag quality regressions.

    ``tolerances`` maps a fully-qualified metric name to the drop it may absorb
    before counting as a regression; anything unlisted gates at zero. It defaults
    to :data:`DEFAULT_TOLERANCES`, which allows for abstention's non-determinism.
    """
    allowed = DEFAULT_TOLERANCES if tolerances is None else tolerances
    deltas: list[MetricDelta] = []
    for metric, base_value, cand_value in _comparable_metrics(baseline, candidate):
        change = (
            cand_value - base_value if base_value is not None and cand_value is not None else None
        )
        regressed = (
            metric in _HIGHER_IS_BETTER
            and change is not None
            and change < -allowed.get(metric, 0.0)
        )
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=base_value,
                candidate=cand_value,
                delta=change,
                regressed=regressed,
            )
        )
    return ReportDiff(
        deltas=tuple(deltas),
        warnings=comparability_warnings(baseline, candidate),
        regressed=any(delta.regressed for delta in deltas),
    )

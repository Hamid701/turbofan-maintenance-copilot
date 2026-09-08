"""Tests for the versioned evaluation report."""

import json
from pathlib import Path

from turbofan_copilot.evaluation.pipeline_cases import PipelineCase
from turbofan_copilot.evaluation.report import (
    AbstentionScore,
    CostScore,
    EvaluationReport,
    GitRevision,
    MetricDelta,
    ReportDiff,
    RetrievalScore,
    RunContext,
    TimingScore,
    compare_reports,
    git_revision,
    new_run_context,
    score_abstention,
    score_retrieval,
    write_report,
)
from turbofan_copilot.evaluation.retrieval_case import RetrievalEvaluationCase
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import GroundedAnswer, abstention
from turbofan_copilot.retrieval.semantic_retrieval import ScoredChunk


def _case(case_id: str, page: int) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        question=f"question {case_id}",
        expected_source_id="faa-h-8083-32b-chapter-1",
        expected_pdf_pages=(page,),
        relevance_note="hand-built for the test",
    )


def _chunk(page: int) -> DocumentChunk:
    return DocumentChunk(
        source_id="faa-h-8083-32b-chapter-1",
        chapter_number=1,
        pdf_page_number=page,
        printed_page_label=f"1-{page}",
        chunk_index=1,
        text=f"text for page {page}",
    )


class _FixedRetriever:
    """Return page 10 first, then page 20, for any question."""

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        return ((0.9, _chunk(10)), (0.5, _chunk(20)))[:limit]


def test_score_retrieval_computes_hit_and_reciprocal_rank() -> None:
    cases = (_case("hits-at-1", 10), _case("hits-at-2", 20), _case("misses", 99))

    score = score_retrieval(cases, _FixedRetriever())

    assert score.cases == 3
    assert score.hit_at_1 == 1 / 3  # only the page-10 case is first
    assert score.hit_at_3 == 2 / 3  # page-10 and page-20 cases both within top 3
    assert score.mean_reciprocal_rank == (1.0 + 0.5 + 0.0) / 3


def _pipeline_case(case_id: str, *, should_abstain: bool) -> PipelineCase:
    return PipelineCase(
        case_id=case_id,
        question=f"question {case_id}",
        should_abstain=should_abstain,
        note="hand-built for the test",
    )


def test_score_abstention_counts_both_kinds_of_mistake() -> None:
    cases = (
        _pipeline_case("answerable-answered", should_abstain=False),
        _pipeline_case("answerable-refused", should_abstain=False),
        _pipeline_case("unanswerable-refused", should_abstain=True),
        _pipeline_case("unanswerable-answered", should_abstain=True),
    )

    # The fake answers when the case id contains "answered", abstains otherwise.
    def fake_answer(question: str) -> GroundedAnswer:
        if "answered" in question:
            return GroundedAnswer(answer="here you go", citations=())
        return abstention("not covered")

    score = score_abstention(cases, fake_answer)

    assert score.cases == 4
    assert score.accuracy == 0.5
    assert score.answered_when_should_abstain == ("unanswerable-answered",)
    assert score.abstained_when_should_answer == ("answerable-refused",)
    assert score.errors == ()


def test_score_abstention_records_a_raising_case_as_an_error() -> None:
    cases = (
        _pipeline_case("ok", should_abstain=True),
        _pipeline_case("boom", should_abstain=True),
    )

    def flaky_answer(question: str) -> GroundedAnswer:
        if "boom" in question:
            raise RuntimeError("model call failed")
        return abstention("not covered")

    score = score_abstention(cases, flaky_answer)

    assert score.cases == 2
    assert score.errors == ("boom",)
    assert score.accuracy == 1.0  # scored over the one case that ran


def test_git_revision_is_none_or_well_formed() -> None:
    revision = git_revision()

    assert revision is None or (
        isinstance(revision, GitRevision)
        and len(revision.sha) == 40
        and isinstance(revision.dirty, bool)
    )


def test_report_round_trips_through_json() -> None:
    report = EvaluationReport(
        run=new_run_context({"retriever": "hybrid_rrf", "rrf_k": "60"}),
        retrieval=score_retrieval((_case("only", 10),), _FixedRetriever()),
        abstention=score_abstention(
            (_pipeline_case("only", should_abstain=True),),
            lambda _question: abstention("not covered"),
        ),
        cost=CostScore(calls=10, prompt_tokens=8000, completion_tokens=400, usd_estimate=0.0014),
        timing=TimingScore(retrieval_seconds=1.2, abstention_seconds=8.4),
    )

    restored = EvaluationReport.model_validate(json.loads(report.model_dump_json()))

    assert restored == report


def _report(
    *,
    mrr: float,
    accuracy: float | None = None,
    usd: float | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        run=new_run_context({"retriever": "hybrid_rrf"}),
        retrieval=RetrievalScore(
            cases=9, hit_at_1=mrr, hit_at_3=mrr, hit_at_5=mrr, mean_reciprocal_rank=mrr
        ),
        abstention=None
        if accuracy is None
        else AbstentionScore(
            cases=10,
            accuracy=accuracy,
            answered_when_should_abstain=(),
            abstained_when_should_answer=(),
        ),
        cost=None
        if usd is None
        else CostScore(calls=10, prompt_tokens=1, completion_tokens=1, usd_estimate=usd),
    )


def _delta(diff: ReportDiff, metric: str) -> MetricDelta:
    return next(delta for delta in diff.deltas if delta.metric == metric)


def test_compare_reports_flags_a_dropped_quality_metric() -> None:
    diff = compare_reports(_report(mrr=0.78, accuracy=1.0), _report(mrr=0.67, accuracy=1.0))

    assert diff.regressed is True
    mrr = _delta(diff, "retrieval.mean_reciprocal_rank")
    assert mrr.delta is not None and mrr.delta < 0
    assert mrr.regressed is True
    assert _delta(diff, "abstention.accuracy").regressed is False


def test_compare_reports_does_not_flag_an_improvement_or_a_cost_rise() -> None:
    diff = compare_reports(
        _report(mrr=0.70, accuracy=0.9, usd=0.001),
        _report(mrr=0.78, accuracy=1.0, usd=0.004),
    )

    assert diff.regressed is False
    cost = _delta(diff, "cost.usd_estimate")
    assert cost.delta is not None and cost.delta > 0


def test_compare_reports_tolerates_a_missing_section_in_the_candidate() -> None:
    diff = compare_reports(_report(mrr=0.78, accuracy=1.0), _report(mrr=0.78))

    accuracy = _delta(diff, "abstention.accuracy")
    assert accuracy.candidate is None
    assert accuracy.delta is None
    assert diff.regressed is False


def test_compare_reports_respects_tolerances() -> None:
    strict = compare_reports(_report(mrr=0.780), _report(mrr=0.779))
    # The helper moves every retrieval metric together, so all four need the slack.
    loose = compare_reports(
        _report(mrr=0.780),
        _report(mrr=0.779),
        tolerances=dict.fromkeys(
            (
                "retrieval.hit_at_1",
                "retrieval.hit_at_3",
                "retrieval.hit_at_5",
                "retrieval.mean_reciprocal_rank",
            ),
            0.01,
        ),
    )

    assert strict.regressed is True
    assert loose.regressed is False


def test_abstention_wobble_is_tolerated_by_default_but_retrieval_is_not() -> None:
    """One flipped case out of ten is model noise; a retrieval drop never is."""
    noisy = compare_reports(_report(mrr=0.78, accuracy=1.0), _report(mrr=0.78, accuracy=0.9))
    real = compare_reports(_report(mrr=0.78, accuracy=1.0), _report(mrr=0.77, accuracy=1.0))

    assert noisy.regressed is False
    assert real.regressed is True


def test_compare_reports_warns_when_the_runs_are_not_like_for_like() -> None:
    baseline = _report(mrr=0.78)
    candidate = _report(mrr=0.78)
    shifted = candidate.model_copy(
        update={"run": candidate.run.model_copy(update={"config": {"retriever": "lexical_idf"}})}
    )

    diff = compare_reports(baseline, shifted)

    assert any("config retriever" in warning for warning in diff.warnings)


def test_compare_reports_warns_on_a_different_evaluation_set_size() -> None:
    baseline = _report(mrr=0.78)
    candidate = _report(mrr=0.78)
    assert candidate.retrieval is not None
    resized = candidate.model_copy(
        update={"retrieval": candidate.retrieval.model_copy(update={"cases": 20})}
    )

    diff = compare_reports(baseline, resized)

    assert any("20 in the candidate" in warning for warning in diff.warnings)


def test_write_report_names_the_file_by_run_timestamp(tmp_path: Path) -> None:
    context = RunContext.model_validate(
        {
            "created_at": "2026-09-07T08:30:00Z",
            "git": None,
            "config": {"retriever": "hybrid_rrf"},
        }
    )
    report = EvaluationReport(run=context)

    path = write_report(report, tmp_path)

    assert path.name == "evaluation-20260907T083000Z.json"
    assert EvaluationReport.model_validate(json.loads(path.read_text(encoding="utf-8"))) == report

"""Score retrieval quality and, when a key is set, answer-or-abstain accuracy.

One command, one timestamped report under ``data/evaluation/reports/``. Retrieval
scoring runs offline against the persisted corpus; abstention scoring is skipped
without ``TURBOFAN_OPENAI_API_KEY`` so the retrieval half always runs. When
``data/evaluation/baseline.json`` exists the new report is diffed against it and
the command exits non-zero on a regression; ``--set-baseline`` promotes the new
report to the baseline instead.
"""

import argparse
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session

from turbofan_copilot.core.config import get_settings
from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.evaluation.pipeline_cases import load_pipeline_cases
from turbofan_copilot.evaluation.report import (
    CostScore,
    EvaluationReport,
    ReportDiff,
    TimingScore,
    compare_reports,
    new_run_context,
    score_abstention,
    score_retrieval,
    write_report,
)
from turbofan_copilot.evaluation.retrieval_case import (
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.llm.openai_provider import build_openai_provider
from turbofan_copilot.llm.pipeline import build_answer_chain
from turbofan_copilot.llm.usage import estimate_usd
from turbofan_copilot.retrieval.bge_embedder import MODEL_ID, MODEL_REVISION, BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import RRF_K, HybridRetriever
from turbofan_copilot.retrieval.semantic_retrieval import SemanticRetriever, SupportsRank

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORTS_DIR = EVAL_DIR / "reports"
BASELINE_PATH = EVAL_DIR / "baseline.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def _cell(value: float | None, *, signed: bool = False) -> str:
    """Format one table cell: whole numbers as ints, fractions to 4 dp."""
    if value is None:
        return "-"
    if value == int(value):
        return f"{int(value):+d}" if signed else str(int(value))
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _format_diff(diff: ReportDiff) -> str:
    """Render the metric-by-metric comparison as an aligned table."""
    width = max(len(delta.metric) for delta in diff.deltas)
    lines = [f"! not like-for-like - {warning}" for warning in diff.warnings]
    if lines:
        lines.append("")
    lines.append(f"{'metric'.ljust(width)}  {'baseline':>10}  {'candidate':>10}  {'delta':>10}")
    for delta in diff.deltas:
        base = _cell(delta.baseline)
        cand = _cell(delta.candidate)
        change = _cell(delta.delta, signed=True)
        flag = "  REGRESSED" if delta.regressed else ""
        lines.append(f"{delta.metric.ljust(width)}  {base:>10}  {cand:>10}  {change:>10}{flag}")
    return "\n".join(lines)


def main() -> None:
    """Build the hybrid retriever, score every section, write and compare the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=("hybrid", "lexical"),
        default="hybrid",
        help="which retrieval configuration to score (default: hybrid)",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="write this report as data/evaluation/baseline.json and skip the diff",
    )
    args = parser.parse_args()

    settings = get_settings()
    retrieval_cases = load_retrieval_evaluation_cases(EVAL_DIR / "retrieval_cases.json")
    pipeline_cases = load_pipeline_cases(EVAL_DIR / "pipeline_cases.json")

    with Session(get_engine()) as session:
        corpus = load_persisted_corpus(session)

    validate_retrieval_case_pages(retrieval_cases, corpus.chunks)

    lexical = LexicalRetriever(corpus.chunks)
    config = {"corpus_chunks": str(len(corpus.chunks))}
    if args.retriever == "lexical":
        retriever: SupportsRank = lexical
        config["retriever"] = "lexical_idf"
    else:
        embedder = BgeEmbedder(MODEL_CACHE)
        semantic = SemanticRetriever(corpus.chunks, corpus.vectors, embedder)
        retriever = HybridRetriever(lexical, semantic, corpus_size=len(corpus.chunks))
        config["retriever"] = "hybrid_rrf"
        config["embedding_model"] = MODEL_ID
        config["embedding_revision"] = MODEL_REVISION
        config["rrf_k"] = str(RRF_K)

    started = perf_counter()
    retrieval = score_retrieval(retrieval_cases, retriever)
    retrieval_seconds = perf_counter() - started

    abstention = None
    cost = None
    abstention_seconds: float | None = None
    if settings.openai_api_key is None:
        print("TURBOFAN_OPENAI_API_KEY not set - skipping abstention scoring\n")
    else:
        provider = build_openai_provider(settings)
        chain = build_answer_chain(retriever, provider)

        started = perf_counter()
        abstention = score_abstention(pipeline_cases, chain.invoke)
        abstention_seconds = perf_counter() - started
        config["answer_model"] = settings.openai_model

        usage = provider.usage
        cost = CostScore(
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            usd_estimate=estimate_usd(provider.model, usage),
        )

    report = EvaluationReport(
        run=new_run_context(config),
        retrieval=retrieval,
        abstention=abstention,
        cost=cost,
        timing=TimingScore(
            retrieval_seconds=round(retrieval_seconds, 3),
            abstention_seconds=None if abstention_seconds is None else round(abstention_seconds, 3),
        ),
    )
    path = write_report(report, REPORTS_DIR)
    print(report.model_dump_json(indent=2))
    print(f"\nwrote {path.relative_to(PROJECT_ROOT)}")

    if args.set_baseline:
        BASELINE_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"updated {BASELINE_PATH.relative_to(PROJECT_ROOT)}")
        return

    if not BASELINE_PATH.exists():
        print("\nno baseline.json yet - run with --set-baseline to create one")
        return

    baseline = EvaluationReport.model_validate_json(BASELINE_PATH.read_text(encoding="utf-8"))
    diff = compare_reports(baseline, report)
    print(f"\nvs {BASELINE_PATH.relative_to(PROJECT_ROOT)}:\n{_format_diff(diff)}")
    if diff.regressed:
        raise SystemExit("a metric regressed against the baseline")


if __name__ == "__main__":
    main()

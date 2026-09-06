"""Measure the HNSW index: that it is usable, its recall versus exact, and the nine-case metrics."""

import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.models import EMBEDDING_HNSW_INDEX
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.db.vector_search import (
    DbSemanticRetriever,
    ScoredChunk,
    search_by_vector,
)
from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.evaluation.retrieval_case import (
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import (
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import evaluate_semantic_retriever

RECALL_DEPTH = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"

IN_MEMORY_700 = {
    "semantic": {"hit_at_1": 0.6667, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7593},
    "hybrid": {"hit_at_1": 0.7778, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.7778},
}


def rounded(scores: dict[str, float]) -> dict[str, float]:
    """Round to four places and rename mean_reciprocal_rank to mrr."""
    return {
        "hit_at_1": round(scores["hit_at_1"], 4),
        "hit_at_3": round(scores["hit_at_3"], 4),
        "hit_at_5": round(scores["hit_at_5"], 4),
        "mrr": round(scores["mean_reciprocal_rank"], 4),
    }


def keys(ranked: tuple[ScoredChunk, ...]) -> set[tuple[str, int, int]]:
    """Reduce a ranking to a set of chunk citation keys."""
    return {(chunk.source_id, chunk.pdf_page_number, chunk.chunk_index) for _, chunk in ranked}


def main() -> None:
    """Confirm the HNSW index is usable, measure its recall, and re-check the metrics."""
    engine = get_engine()
    embedder = BgeEmbedder(MODEL_CACHE)
    cases = load_retrieval_evaluation_cases(CASES_PATH)

    with Session(engine) as session:
        corpus = load_persisted_corpus(session)
        validate_retrieval_case_pages(cases, corpus.chunks)

        index_definition = session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
            {"name": EMBEDDING_HNSW_INDEX},
        ).scalar_one()

        # At 1,050 rows the planner prefers a sequential scan, so force the index on
        # for the rest of this transaction to actually exercise the HNSW path.
        session.execute(text("SET LOCAL enable_seqscan = off"))

        probe_vector = embedder.embed_question(cases[0].question)
        literal_vector = "[" + ",".join(repr(value) for value in probe_vector) + "]"
        plan = [
            row[0]
            for row in session.execute(
                text(
                    "EXPLAIN SELECT chunk_id FROM chunk_embeddings "
                    f"ORDER BY embedding <=> '{literal_vector}' LIMIT 5"
                )
            ).all()
        ]
        index_used = any(EMBEDDING_HNSW_INDEX in line for line in plan)
        plan = [line if len(line) <= 160 else f"{line[:160]}..." for line in plan]

        recall_per_case = {}
        for case in cases:
            question_vector = embedder.embed_question(case.question)
            approx = keys(
                search_by_vector(session, question_vector, limit=RECALL_DEPTH, exact=False)
            )
            exact = keys(search_by_vector(session, question_vector, limit=RECALL_DEPTH, exact=True))
            recall_per_case[case.case_id] = round(len(approx & exact) / RECALL_DEPTH, 3)

        lexical = LexicalRetriever(corpus.chunks)
        db_semantic = DbSemanticRetriever(session, embedder, exact=True)
        hybrid = HybridRetriever(lexical, db_semantic, corpus_size=len(corpus.chunks))
        db_metrics = {
            "semantic": rounded(evaluate_semantic_retriever(cases, db_semantic)),
            "hybrid": rounded(evaluate_hybrid_retriever(cases, hybrid)),
        }

    report = {
        "index_name": EMBEDDING_HNSW_INDEX,
        "index_definition": index_definition,
        "explain_plan_seqscan_disabled": plan,
        "index_used_when_forced": index_used,
        "recall_at_10_vs_exact": {
            "per_case": recall_per_case,
            "mean": round(sum(recall_per_case.values()) / len(recall_per_case), 3),
        },
        "in_memory_700": IN_MEMORY_700,
        "database_metrics_exact_scan": db_metrics,
        "matches_in_memory": db_metrics == IN_MEMORY_700,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

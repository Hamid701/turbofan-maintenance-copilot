"""Run the three retrievers over the PostgreSQL-persisted corpus and compare to memory."""

import json
from pathlib import Path

from sqlalchemy.orm import Session

from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.evaluation.lexical_retrieval import (
    LexicalRetriever,
    evaluate_lexical_retriever,
)
from turbofan_copilot.evaluation.retrieval_case import (
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import (
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import (
    SemanticRetriever,
    evaluate_semantic_retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"

# In-memory 700-character results from Module 16 (scripts/measure_rechunk_retrieval.py).
IN_MEMORY_700 = {
    "lexical": {"hit_at_1": 0.5556, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7222},
    "semantic": {"hit_at_1": 0.6667, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7593},
    "hybrid": {"hit_at_1": 0.7778, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.7778},
}


def rounded(scores: dict[str, float]) -> dict[str, float]:
    """Round to four places and rename mean_reciprocal_rank to mrr for comparison."""
    return {
        "hit_at_1": round(scores["hit_at_1"], 4),
        "hit_at_3": round(scores["hit_at_3"], 4),
        "hit_at_5": round(scores["hit_at_5"], 4),
        "mrr": round(scores["mean_reciprocal_rank"], 4),
    }


def main() -> None:
    """Load the persisted corpus, score the nine cases, and check it matches memory."""
    with Session(get_engine()) as session:
        corpus = load_persisted_corpus(session)

    cases = load_retrieval_evaluation_cases(CASES_PATH)
    validate_retrieval_case_pages(cases, corpus.chunks)

    embedder = BgeEmbedder(MODEL_CACHE)
    lexical = LexicalRetriever(corpus.chunks)
    semantic = SemanticRetriever(corpus.chunks, corpus.vectors, embedder)
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(corpus.chunks))

    db_metrics = {
        "lexical": rounded(evaluate_lexical_retriever(cases, lexical)),
        "semantic": rounded(evaluate_semantic_retriever(cases, semantic)),
        "hybrid": rounded(evaluate_hybrid_retriever(cases, hybrid)),
    }
    matches_memory = db_metrics == IN_MEMORY_700

    report = {
        "source": "postgresql",
        "chunk_count": len(corpus.chunks),
        "vector_count": len(corpus.vectors),
        "vector_dimension": sorted({len(vector) for vector in corpus.vectors}),
        "model_id": corpus.model_id,
        "model_revision": corpus.model_revision,
        "in_memory_700": IN_MEMORY_700,
        "database_metrics": db_metrics,
        "matches_in_memory_exactly": matches_memory,
    }
    print(json.dumps(report, indent=2))

    if not matches_memory:
        raise RuntimeError("database retrieval metrics do not match the in-memory baseline")


if __name__ == "__main__":
    main()

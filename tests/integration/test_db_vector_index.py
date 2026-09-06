"""Live-database checks for the HNSW embedding index and SQL cosine search."""

from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.models import EMBEDDING_HNSW_INDEX
from turbofan_copilot.db.vector_search import DbSemanticRetriever, search_by_vector
from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.evaluation.retrieval_case import load_retrieval_evaluation_cases
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import (
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import evaluate_semantic_retriever

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
EXPECTED_CHUNKS = 1050

IN_MEMORY_700 = {
    "semantic": {"hit_at_1": 0.6667, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7593},
    "hybrid": {"hit_at_1": 0.7778, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.7778},
}


def rounded(scores: dict[str, float]) -> dict[str, float]:
    return {
        "hit_at_1": round(scores["hit_at_1"], 4),
        "hit_at_3": round(scores["hit_at_3"], 4),
        "hit_at_5": round(scores["hit_at_5"], 4),
        "mrr": round(scores["mean_reciprocal_rank"], 4),
    }


def test_hnsw_index_exists(db_engine: Engine) -> None:
    with Session(db_engine) as session:
        definition = session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
            {"name": EMBEDDING_HNSW_INDEX},
        ).scalar_one_or_none()
    assert definition is not None
    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition


def test_sql_cosine_search_reproduces_the_in_memory_semantic_metrics(
    db_engine: Engine,
    embedder: BgeEmbedder,
) -> None:
    with Session(db_engine) as session:
        corpus = load_persisted_corpus(session)
        if len(corpus.chunks) != EXPECTED_CHUNKS:
            pytest.skip(f"persisted corpus has {len(corpus.chunks)} chunks; run the ingestion")

        cases = load_retrieval_evaluation_cases(CASES_PATH)
        lexical = LexicalRetriever(corpus.chunks)
        db_semantic = DbSemanticRetriever(session, embedder, exact=True)
        hybrid = HybridRetriever(lexical, db_semantic, corpus_size=len(corpus.chunks))

        assert {
            "semantic": rounded(evaluate_semantic_retriever(cases, db_semantic)),
            "hybrid": rounded(evaluate_hybrid_retriever(cases, hybrid)),
        } == IN_MEMORY_700


def test_indexed_search_recall_matches_exact_for_top_results(
    db_engine: Engine,
    embedder: BgeEmbedder,
) -> None:
    with Session(db_engine) as session:
        if len(load_persisted_corpus(session).chunks) != EXPECTED_CHUNKS:
            pytest.skip("persisted corpus not ingested")

        # Force the planner to use the HNSW index; at this corpus size it would seq-scan.
        session.execute(text("SET LOCAL enable_seqscan = off"))

        cases = load_retrieval_evaluation_cases(CASES_PATH)
        for case in cases:
            question_vector = embedder.embed_question(case.question)
            approx = search_by_vector(session, question_vector, limit=5, exact=False)
            exact = search_by_vector(session, question_vector, limit=5, exact=True)
            approx_keys = {(c.source_id, c.pdf_page_number, c.chunk_index) for _, c in approx}
            exact_keys = {(c.source_id, c.pdf_page_number, c.chunk_index) for _, c in exact}
            assert approx_keys == exact_keys, case.case_id

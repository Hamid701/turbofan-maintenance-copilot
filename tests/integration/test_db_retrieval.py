"""Live-database checks that PostgreSQL retrieval reproduces the in-memory baseline."""

from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from turbofan_copilot.db.corpus import PersistedCorpus, load_persisted_corpus
from turbofan_copilot.evaluation.lexical_retrieval import (
    LexicalRetriever,
    evaluate_lexical_retriever,
)
from turbofan_copilot.evaluation.retrieval_case import (
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.retrieval.bge_embedder import (
    EMBEDDING_DIMENSION,
    MODEL_ID,
    BgeEmbedder,
)
from turbofan_copilot.retrieval.hybrid_retrieval import (
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import (
    SemanticRetriever,
    evaluate_semantic_retriever,
)

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"

EXPECTED_CHUNKS = 1050
IN_MEMORY_700 = {
    "lexical": {"hit_at_1": 0.5556, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7222},
    "semantic": {"hit_at_1": 0.6667, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7593},
    "hybrid": {"hit_at_1": 0.7778, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.7778},
}


@pytest.fixture(scope="module")
def corpus(db_engine: Engine) -> PersistedCorpus:
    """Load the persisted corpus, or skip if the 700-char corpus has not been ingested."""
    with Session(db_engine) as session:
        try:
            loaded = load_persisted_corpus(session)
        except RuntimeError as error:
            pytest.skip(str(error))
    if len(loaded.chunks) != EXPECTED_CHUNKS:
        pytest.skip(
            f"persisted corpus has {len(loaded.chunks)} chunks; "
            f"expected {EXPECTED_CHUNKS} from the 700-char ingestion"
        )
    return loaded


def rounded(scores: dict[str, float]) -> dict[str, float]:
    """Round to four places and rename mean_reciprocal_rank to mrr."""
    return {
        "hit_at_1": round(scores["hit_at_1"], 4),
        "hit_at_3": round(scores["hit_at_3"], 4),
        "hit_at_5": round(scores["hit_at_5"], 4),
        "mrr": round(scores["mean_reciprocal_rank"], 4),
    }


def test_persisted_corpus_is_aligned_and_single_model(corpus: PersistedCorpus) -> None:
    assert len(corpus.chunks) == len(corpus.vectors) == EXPECTED_CHUNKS
    assert {len(vector) for vector in corpus.vectors} == {EMBEDDING_DIMENSION}
    assert corpus.model_id == MODEL_ID


def test_database_retrieval_reproduces_the_in_memory_metrics(
    corpus: PersistedCorpus,
    embedder: BgeEmbedder,
) -> None:
    cases = load_retrieval_evaluation_cases(CASES_PATH)
    validate_retrieval_case_pages(cases, corpus.chunks)

    lexical = LexicalRetriever(corpus.chunks)
    semantic = SemanticRetriever(corpus.chunks, corpus.vectors, embedder)
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(corpus.chunks))

    assert {
        "lexical": rounded(evaluate_lexical_retriever(cases, lexical)),
        "semantic": rounded(evaluate_semantic_retriever(cases, semantic)),
        "hybrid": rounded(evaluate_hybrid_retriever(cases, hybrid)),
    } == IN_MEMORY_700

"""Compare lexical, semantic, and RRF-hybrid ranking on the nine frozen retrieval cases."""

import json
from pathlib import Path

from turbofan_copilot.evaluation.lexical_retrieval import (
    LexicalRetriever,
    evaluate_lexical_retriever,
)
from turbofan_copilot.evaluation.retrieval_case import (
    RetrievalEvaluationCase,
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.ingestion.corpus_build import load_document_chunks
from turbofan_copilot.retrieval.bge_embedder import MODEL_ID, MODEL_REVISION, BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import (
    RRF_K,
    HybridRetriever,
    evaluate_hybrid_retriever,
)
from turbofan_copilot.retrieval.semantic_retrieval import (
    ScoredChunk,
    SemanticRetriever,
    evaluate_semantic_retriever,
)

EXPECTED_CHUNK_COUNT = 503
FOCUS_CASE_ID = "compressor_air_path_paraphrase"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks.jsonl"
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"

type Ranker = LexicalRetriever | SemanticRetriever | HybridRetriever


def first_relevant_rank(
    ranking: tuple[ScoredChunk, ...],
    case: RetrievalEvaluationCase,
) -> int | None:
    """Return the 1-based position of the first chunk from an expected page."""
    return next(
        (
            rank
            for rank, (_, chunk) in enumerate(ranking, start=1)
            if chunk.source_id == case.expected_source_id
            and chunk.pdf_page_number in case.expected_pdf_pages
        ),
        None,
    )


def full_order(ranker: Ranker, question: str) -> list[tuple[str, int, int]]:
    """Return the full ranked list of chunk identities for one question."""
    return [
        (chunk.source_id, chunk.pdf_page_number, chunk.chunk_index)
        for _, chunk in ranker.rank(question, limit=EXPECTED_CHUNK_COUNT)
    ]


def main() -> None:
    """Score all three retrievers on the frozen cases and print a comparison report."""
    chunks = load_document_chunks(CORPUS_PATH)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(f"corpus has {len(chunks)} chunks; expected {EXPECTED_CHUNK_COUNT}")

    cases = load_retrieval_evaluation_cases(CASES_PATH)
    validate_retrieval_case_pages(cases, chunks)

    embedder = BgeEmbedder(MODEL_CACHE)
    vectors = embedder.embed_passages(chunk.text for chunk in chunks)

    lexical = LexicalRetriever(chunks)
    semantic = SemanticRetriever(chunks, vectors, embedder)
    hybrid = HybridRetriever(lexical, semantic, corpus_size=len(chunks))

    raw_metrics = {
        "lexical": evaluate_lexical_retriever(cases, lexical),
        "semantic": evaluate_semantic_retriever(cases, semantic),
        "hybrid": evaluate_hybrid_retriever(cases, hybrid),
    }
    metrics = {
        name: {key: round(value, 4) for key, value in scores.items()}
        for name, scores in raw_metrics.items()
    }

    hybrid_first_run = {case.case_id: full_order(hybrid, case.question) for case in cases}
    hybrid_second_run = {case.case_id: full_order(hybrid, case.question) for case in cases}
    unique_keys = {(c.source_id, c.pdf_page_number, c.chunk_index) for c in chunks}

    per_case = []
    for case in cases:
        rankings = {
            "lexical": lexical.rank(case.question, limit=len(chunks)),
            "semantic": semantic.rank(case.question, limit=len(chunks)),
            "hybrid": hybrid.rank(case.question, limit=len(chunks)),
        }
        per_case.append(
            {
                "case_id": case.case_id,
                "expected_page": case.expected_pdf_pages[0],
                "first_relevant_rank_full_corpus": {
                    name: first_relevant_rank(ranking, case) for name, ranking in rankings.items()
                },
            }
        )

    focus_case = next(case for case in cases if case.case_id == FOCUS_CASE_ID)
    focus_hybrid = hybrid.rank(focus_case.question, limit=len(chunks))
    focus_rows = [
        {
            "rank": rank,
            "pdf_page_number": chunk.pdf_page_number,
            "chunk_index": chunk.chunk_index,
            "fused_score": round(score, 6),
        }
        for rank, (score, chunk) in enumerate(focus_hybrid, start=1)
        if chunk.source_id == focus_case.expected_source_id
        and chunk.pdf_page_number in focus_case.expected_pdf_pages
    ]

    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rrf_k": RRF_K,
        "corpus_path": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "cases_path": str(CASES_PATH.relative_to(PROJECT_ROOT)),
        "chunk_count": len(chunks),
        "case_count": len(cases),
        "metrics_note": "Hit@k and MRR use the top-5 budget for all three retrievers.",
        "metrics": metrics,
        "hybrid_determinism": {
            "sort_key_is_total": len(unique_keys) == len(chunks),
            "identical_across_two_runs": hybrid_first_run == hybrid_second_run,
        },
        "per_case_first_relevant_rank": per_case,
        "focus_case": {
            "case_id": focus_case.case_id,
            "expected_page": focus_case.expected_pdf_pages[0],
            "hybrid_expected_page_chunks": focus_rows,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

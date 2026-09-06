"""Rank the nine frozen retrieval cases against in-memory BGE corpus vectors."""

import json
from pathlib import Path

from turbofan_copilot.evaluation.retrieval_case import (
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.ingestion.corpus_build import load_document_chunks
from turbofan_copilot.retrieval.bge_embedder import MODEL_ID, MODEL_REVISION, BgeEmbedder
from turbofan_copilot.retrieval.semantic_retrieval import (
    SemanticRetriever,
    evaluate_semantic_retriever,
)

EXPECTED_CHUNK_COUNT = 503
FOCUS_CASE_ID = "compressor_air_path_paraphrase"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks.jsonl"
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def identity_order(retriever: SemanticRetriever, question: str) -> list[tuple[str, int, int]]:
    """Return the full ranked list of chunk identities for one question."""
    return [
        (chunk.source_id, chunk.pdf_page_number, chunk.chunk_index)
        for _, chunk in retriever.rank(question, limit=EXPECTED_CHUNK_COUNT)
    ]


def main() -> None:
    """Embed the corpus once, score the frozen cases, and print a verification report."""
    chunks = load_document_chunks(CORPUS_PATH)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(f"corpus has {len(chunks)} chunks; expected {EXPECTED_CHUNK_COUNT}")

    cases = load_retrieval_evaluation_cases(CASES_PATH)
    validate_retrieval_case_pages(cases, chunks)

    embedder = BgeEmbedder(MODEL_CACHE)
    vectors = embedder.embed_passages(chunk.text for chunk in chunks)
    retriever = SemanticRetriever(chunks, vectors, embedder)

    metrics = evaluate_semantic_retriever(cases, retriever)

    unique_sort_keys = {
        (chunk.source_id, chunk.pdf_page_number, chunk.chunk_index) for chunk in chunks
    }
    sort_key_is_total = len(unique_sort_keys) == len(chunks)

    first_order = {case.case_id: identity_order(retriever, case.question) for case in cases}
    second_order = {case.case_id: identity_order(retriever, case.question) for case in cases}
    identical_across_two_runs = first_order == second_order

    per_case = []
    for case in cases:
        full_ranking = retriever.rank(case.question, limit=EXPECTED_CHUNK_COUNT)
        top_score, top_chunk = full_ranking[0]
        ranked_chunks = tuple(chunk for _, chunk in full_ranking)
        first_relevant_rank = next(
            (
                rank
                for rank, chunk in enumerate(ranked_chunks, start=1)
                if chunk.source_id == case.expected_source_id
                and chunk.pdf_page_number in case.expected_pdf_pages
            ),
            None,
        )
        per_case.append(
            {
                "case_id": case.case_id,
                "expected_page": case.expected_pdf_pages[0],
                "first_relevant_rank_full_corpus": first_relevant_rank,
                "in_top_5": first_relevant_rank is not None and first_relevant_rank <= 5,
                "top_result": {
                    "source_id": top_chunk.source_id,
                    "pdf_page_number": top_chunk.pdf_page_number,
                    "chunk_index": top_chunk.chunk_index,
                    "score": round(top_score, 6),
                },
            }
        )

    focus_case = next(case for case in cases if case.case_id == FOCUS_CASE_ID)
    focus_ranking = retriever.rank(focus_case.question, limit=EXPECTED_CHUNK_COUNT)
    focus_page_rows = [
        {
            "rank": rank,
            "pdf_page_number": chunk.pdf_page_number,
            "chunk_index": chunk.chunk_index,
            "score": round(score, 6),
        }
        for rank, (score, chunk) in enumerate(focus_ranking, start=1)
        if chunk.source_id == focus_case.expected_source_id
        and chunk.pdf_page_number in focus_case.expected_pdf_pages
    ]

    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "corpus_path": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "cases_path": str(CASES_PATH.relative_to(PROJECT_ROOT)),
        "chunk_count": len(chunks),
        "case_count": len(cases),
        "metrics_note": "Hit@k and MRR use the top-5 budget, matching the lexical baseline.",
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "determinism": {
            "sort_key_is_total": sort_key_is_total,
            "identical_across_two_runs": identical_across_two_runs,
        },
        "per_case": per_case,
        "focus_case": {
            "case_id": focus_case.case_id,
            "expected_source_id": focus_case.expected_source_id,
            "expected_page": focus_case.expected_pdf_pages[0],
            "expected_page_chunks": focus_page_rows,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Re-chunk experiment: rebuild the corpus with a smaller window, re-run all three retrievers."""

import json
from math import fsum
from pathlib import Path
from statistics import median

from turbofan_copilot.evaluation.lexical_retrieval import (
    LexicalRetriever,
    evaluate_lexical_retriever,
)
from turbofan_copilot.evaluation.retrieval_case import (
    RetrievalEvaluationCase,
    load_retrieval_evaluation_cases,
    validate_retrieval_case_pages,
)
from turbofan_copilot.ingestion.corpus_build import (
    build_document_corpus,
    load_document_chunks,
)
from turbofan_copilot.retrieval.bge_embedder import MODEL_ID, BgeEmbedder, EmbeddingVector
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

VARIANT_MAX_CHARACTERS = 700
VARIANT_OVERLAP_CHARACTERS = 100
FOCUS_CASE_ID = "compressor_air_path_paraphrase"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATHS = [
    PROJECT_ROOT / "data" / "manifests" / name
    for name in (
        "faa_h_8083_32b_chapter_01.json",
        "faa_h_8083_32b_chapter_06.json",
        "faa_h_8083_32b_chapter_10.json",
    )
]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "faa"
VARIANT_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks_window700.jsonl"
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"

# Frozen character-window (1,500 / 200) results from Module 15 Steps 4 and 6, for reference.
BASELINE_1500_CHAR = {
    "chunk_count": 503,
    "lexical": {"hit_at_1": 0.6667, "hit_at_3": 0.8889, "hit_at_5": 0.8889, "mrr": 0.7778},
    "semantic": {"hit_at_1": 0.4444, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.5926},
    "hybrid": {"hit_at_1": 0.6667, "hit_at_3": 0.7778, "hit_at_5": 0.7778, "mrr": 0.7037},
}


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


def cosine(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Return the dot product of two unit-length vectors."""
    return fsum(a * b for a, b in zip(left, right, strict=True))


def main() -> None:
    """Rebuild the corpus with a smaller window and print a three-retriever comparison."""
    chunks = build_document_corpus(
        MANIFEST_PATHS,
        RAW_DIRECTORY,
        VARIANT_PATH,
        max_characters=VARIANT_MAX_CHARACTERS,
        overlap_characters=VARIANT_OVERLAP_CHARACTERS,
    )
    if load_document_chunks(VARIANT_PATH) != chunks:
        raise RuntimeError("written variant corpus does not round-trip")

    cases = load_retrieval_evaluation_cases(CASES_PATH)
    validate_retrieval_case_pages(cases, chunks)

    embedder = BgeEmbedder(MODEL_CACHE)
    vectors = embedder.embed_passages(chunk.text for chunk in chunks)
    vector_by_chunk = dict(zip(chunks, vectors, strict=True))

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

    char_lengths = sorted(len(chunk.text) for chunk in chunks)
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
    focus_vector = embedder.embed_question(focus_case.question)
    focus_hybrid = hybrid.rank(focus_case.question, limit=len(chunks))
    focus_rows = [
        {
            "hybrid_rank": rank,
            "pdf_page_number": chunk.pdf_page_number,
            "chunk_index": chunk.chunk_index,
            "cosine": round(cosine(focus_vector, vector_by_chunk[chunk]), 4),
            "text_head": " ".join(chunk.text.split())[:140],
        }
        for rank, (_, chunk) in enumerate(focus_hybrid, start=1)
        if chunk.source_id == focus_case.expected_source_id
        and chunk.pdf_page_number in focus_case.expected_pdf_pages
    ]

    report = {
        "experiment": "smaller character window",
        "model_id": MODEL_ID,
        "rrf_k": RRF_K,
        "variant_path": str(VARIANT_PATH.relative_to(PROJECT_ROOT)),
        "max_characters": VARIANT_MAX_CHARACTERS,
        "overlap_characters": VARIANT_OVERLAP_CHARACTERS,
        "chunk_count": len(chunks),
        "chunk_char_length": {
            "min": char_lengths[0],
            "median": median(char_lengths),
            "max": char_lengths[-1],
        },
        "metrics_note": "Hit@k and MRR use the top-5 budget for all three retrievers.",
        "baseline_1500_char": BASELINE_1500_CHAR,
        "variant_metrics": metrics,
        "per_case_first_relevant_rank": per_case,
        "focus_case": {
            "case_id": focus_case.case_id,
            "expected_page": focus_case.expected_pdf_pages[0],
            "expected_page_chunks_in_hybrid": focus_rows,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

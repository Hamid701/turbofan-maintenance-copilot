"""Audit the FAA corpus against BGE's input-token limit."""

import json
from math import ceil
from pathlib import Path
from statistics import median

from turbofan_copilot.ingestion.corpus_build import load_document_chunks
from turbofan_copilot.retrieval.bge_embedder import (
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    BgeEmbedder,
)

EXPECTED_CHUNK_COUNT = 503

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks.jsonl"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def nearest_rank_percentile(sorted_values: tuple[int, ...], percentile: int) -> int:
    """Return a percentile using the nearest-rank definition."""
    index = ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[index]


def main() -> None:
    """Print corpus token statistics and fail if BGE cannot encode every chunk."""
    chunks = load_document_chunks(CORPUS_PATH)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(f"corpus has {len(chunks)} chunks; expected {EXPECTED_CHUNK_COUNT}")

    embedder = BgeEmbedder(MODEL_CACHE)
    token_counts = tuple(embedder.passage_token_count(chunk.text) for chunk in chunks)
    sorted_counts = tuple(sorted(token_counts))
    maximum = sorted_counts[-1]
    longest_chunk = chunks[token_counts.index(maximum)]

    buckets = {
        "1-128": sum(count <= 128 for count in token_counts),
        "129-256": sum(128 < count <= 256 for count in token_counts),
        "257-384": sum(256 < count <= 384 for count in token_counts),
        "385-512": sum(384 < count <= MAX_TOKENS for count in token_counts),
        "over-512": sum(count > MAX_TOKENS for count in token_counts),
    }
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "corpus_path": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "chunk_count": len(chunks),
        "token_limit": MAX_TOKENS,
        "minimum_tokens": sorted_counts[0],
        "median_tokens": median(sorted_counts),
        "p95_tokens_nearest_rank": nearest_rank_percentile(sorted_counts, 95),
        "maximum_tokens": maximum,
        "token_buckets": buckets,
        "overflow_count": buckets["over-512"],
        "longest_chunk": {
            "source_id": longest_chunk.source_id,
            "chapter_number": longest_chunk.chapter_number,
            "pdf_page_number": longest_chunk.pdf_page_number,
            "printed_page_label": longest_chunk.printed_page_label,
            "chunk_index": longest_chunk.chunk_index,
            "token_count": maximum,
        },
    }
    print(json.dumps(report, indent=2))

    if buckets["over-512"]:
        raise RuntimeError("one or more corpus chunks exceed BGE's token limit")


if __name__ == "__main__":
    main()

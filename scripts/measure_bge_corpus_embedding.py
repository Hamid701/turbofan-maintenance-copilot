"""Measure one in-memory BGE embedding run over the validated FAA corpus."""

import json
from math import fsum, isfinite, sqrt
from pathlib import Path
from time import perf_counter

from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.ingestion.corpus_build import load_document_chunks
from turbofan_copilot.retrieval.bge_embedder import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    BgeEmbedder,
    EmbeddingVector,
)

EXPECTED_CHUNK_COUNT = 503
UNIT_LENGTH_TOLERANCE = 1e-5
ALIGNMENT_TOLERANCE = 1e-4
ALIGNMENT_PROBE_POSITIONS = (0, EXPECTED_CHUNK_COUNT // 2, EXPECTED_CHUNK_COUNT - 1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks.jsonl"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def vector_length(vector: EmbeddingVector) -> float:
    """Return the geometric length of a vector."""
    return sqrt(fsum(value * value for value in vector))


def cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Return the dot product of two unit-length vectors."""
    return fsum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def probe_alignment(
    embedder: BgeEmbedder,
    chunks: tuple[DocumentChunk, ...],
    vectors: tuple[EmbeddingVector, ...],
    position: int,
) -> dict[str, object]:
    """Re-embed one chunk alone and confirm it matches the batch vector at its index."""
    chunk = chunks[position]
    probe_vector = embedder.embed_passages((chunk.text,))[0]
    similarities = tuple(cosine_similarity(vector, probe_vector) for vector in vectors)
    ranked_positions = sorted(
        range(len(similarities)),
        key=lambda index: similarities[index],
        reverse=True,
    )
    best_position = ranked_positions[0]
    runner_up_position = ranked_positions[1]

    if best_position != position:
        raise RuntimeError(
            f"chunk at position {position} matches batch vector {best_position}; "
            "chunk and vector order are not aligned"
        )
    if abs(similarities[position] - 1.0) > ALIGNMENT_TOLERANCE:
        raise RuntimeError(
            f"chunk at position {position} has self-similarity {similarities[position]}; "
            "expected 1.0"
        )

    return {
        "position": position,
        "source_id": chunk.source_id,
        "pdf_page_number": chunk.pdf_page_number,
        "printed_page_label": chunk.printed_page_label,
        "chunk_index": chunk.chunk_index,
        "self_similarity": round(similarities[position], 6),
        "next_best_position": runner_up_position,
        "next_best_similarity": round(similarities[runner_up_position], 6),
    }


def main() -> None:
    """Embed the whole corpus once in memory and print a verification report."""
    chunks = load_document_chunks(CORPUS_PATH)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(f"corpus has {len(chunks)} chunks; expected {EXPECTED_CHUNK_COUNT}")

    load_started = perf_counter()
    embedder = BgeEmbedder(MODEL_CACHE)
    load_seconds = perf_counter() - load_started

    embed_started = perf_counter()
    vectors = embedder.embed_passages(chunk.text for chunk in chunks)
    embed_seconds = perf_counter() - embed_started

    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks")
    dimensions = {len(vector) for vector in vectors}
    if dimensions != {EMBEDDING_DIMENSION}:
        raise RuntimeError(f"embedding produced vector widths {sorted(dimensions)}")
    if not all(isfinite(value) for vector in vectors for value in vector):
        raise RuntimeError("embedding produced a non-finite value")

    worst_length_error = max(abs(vector_length(vector) - 1.0) for vector in vectors)
    if worst_length_error > UNIT_LENGTH_TOLERANCE:
        raise RuntimeError(f"worst vector length error {worst_length_error} exceeds tolerance")

    probes = [
        probe_alignment(embedder, chunks, vectors, position)
        for position in ALIGNMENT_PROBE_POSITIONS
    ]

    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "corpus_path": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "chunk_count": len(chunks),
        "vector_count": len(vectors),
        "vector_dimension": sorted(dimensions)[0],
        "batch_size": DEFAULT_BATCH_SIZE,
        "batch_count": -(-len(chunks) // DEFAULT_BATCH_SIZE),
        "model_load_seconds": round(load_seconds, 3),
        "embedding_seconds": round(embed_seconds, 3),
        "milliseconds_per_chunk": round(1000 * embed_seconds / len(chunks), 3),
        "chunks_per_second": round(len(chunks) / embed_seconds, 2),
        "worst_unit_length_error": f"{worst_length_error:.3e}",
        "alignment_probes": probes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

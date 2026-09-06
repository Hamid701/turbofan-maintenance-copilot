"""Check the selected BGE model with one real question and one real passage."""

import json
from math import sqrt
from pathlib import Path
from typing import cast

from turbofan_copilot.retrieval.bge_embedder import (
    MODEL_ID,
    MODEL_REVISION,
    BgeEmbedder,
    EmbeddingVector,
)

QUESTION = (
    "Which gas-turbine compressor design throws incoming air outward, "
    "and which design keeps the airflow moving straight through the engine?"
)
EXPECTED_SOURCE_ID = "faa-h-8083-32b-chapter-1"
EXPECTED_PDF_PAGE = 38
EXPECTED_CHUNK_INDEX = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "faa" / "manual_chunks.jsonl"
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def load_passage() -> str:
    """Load the single answer-bearing passage selected for this check."""
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        record = cast(dict[str, object], json.loads(line))
        if (
            record.get("source_id") == EXPECTED_SOURCE_ID
            and record.get("pdf_page_number") == EXPECTED_PDF_PAGE
            and record.get("chunk_index") == EXPECTED_CHUNK_INDEX
        ):
            text = record.get("text")
            if isinstance(text, str):
                return text
            raise RuntimeError("selected passage has no text")

    raise RuntimeError("selected passage was not found in the corpus")


def vector_length(vector: EmbeddingVector) -> float:
    """Return the geometric length of a vector."""
    return sqrt(sum(value * value for value in vector))


def main() -> None:
    """Run the one-question, one-passage connectivity check."""
    embedder = BgeEmbedder(MODEL_CACHE)
    passage = load_passage()

    question_tokens = embedder.question_token_count(QUESTION)
    passage_tokens = embedder.passage_token_count(passage)
    question_vector = embedder.embed_question(QUESTION)
    passage_vector = embedder.embed_passages((passage,))[0]

    question_length = vector_length(question_vector)
    passage_length = vector_length(passage_vector)
    cosine_similarity = sum(
        question_value * passage_value
        for question_value, passage_value in zip(question_vector, passage_vector, strict=True)
    )

    print(f"model: {MODEL_ID}")
    print(f"revision: {MODEL_REVISION}")
    print(f"question tokens: {question_tokens}")
    print(f"passage tokens: {passage_tokens}")
    print(f"question vector: shape=({len(question_vector)},), length={question_length:.6f}")
    print(f"passage vector: shape=({len(passage_vector)},), length={passage_length:.6f}")
    print(f"cosine similarity: {cosine_similarity:.6f}")


if __name__ == "__main__":
    main()

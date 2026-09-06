"""Ingest the accepted 700-character corpus and its BGE vectors into PostgreSQL."""

import json
from pathlib import Path
from time import perf_counter

from sqlalchemy.orm import Session

from turbofan_copilot.db.ingestion import ingest_chunks
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.ingestion.corpus_build import build_document_corpus
from turbofan_copilot.retrieval.bge_embedder import MODEL_ID, MODEL_REVISION, BgeEmbedder

MAX_CHARACTERS = 700
OVERLAP_CHARACTERS = 100

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
MODEL_CACHE = PROJECT_ROOT / "data" / "processed" / "models"


def main() -> None:
    """Rebuild the 700-character corpus, embed it once, and upsert it into PostgreSQL."""
    chunks = build_document_corpus(
        MANIFEST_PATHS,
        RAW_DIRECTORY,
        VARIANT_PATH,
        max_characters=MAX_CHARACTERS,
        overlap_characters=OVERLAP_CHARACTERS,
    )

    embedder = BgeEmbedder(MODEL_CACHE)
    started = perf_counter()
    vectors = embedder.embed_passages(chunk.text for chunk in chunks)
    embed_seconds = perf_counter() - started

    with Session(get_engine()) as session:
        result = ingest_chunks(
            session,
            chunks,
            vectors,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )
        session.commit()

    report = {
        "chunks_built": len(chunks),
        "embed_seconds": round(embed_seconds, 1),
        "chunk_rows": result.chunk_rows,
        "embedding_rows": result.embedding_rows,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_characters": MAX_CHARACTERS,
        "overlap_characters": OVERLAP_CHARACTERS,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

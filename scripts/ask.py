"""Answer one FAA-manual question with citations, using the 700-char hybrid retriever."""

import argparse
import json
from pathlib import Path

from sqlalchemy.orm import Session

from turbofan_copilot.core.config import get_settings
from turbofan_copilot.db.corpus import load_persisted_corpus
from turbofan_copilot.db.session import get_engine
from turbofan_copilot.db.vector_search import DbSemanticRetriever
from turbofan_copilot.evaluation.lexical_retrieval import LexicalRetriever
from turbofan_copilot.llm.openai_provider import build_openai_provider
from turbofan_copilot.llm.pipeline import answer_question
from turbofan_copilot.llm.router import EngineReference
from turbofan_copilot.llm.tools import (
    EngineHealthReport,
    build_engine_health_report,
    load_degradation_model,
    load_rul_regressor,
)
from turbofan_copilot.retrieval.bge_embedder import BgeEmbedder
from turbofan_copilot.retrieval.hybrid_retrieval import HybridRetriever

MODEL_CACHE = Path(__file__).resolve().parents[1] / "data" / "processed" / "models"


def main() -> None:
    """Retrieve evidence for the question and print the grounded answer as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    arguments = parser.parse_args()

    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("set TURBOFAN_OPENAI_API_KEY in your .env first")

    provider = build_openai_provider(settings)
    embedder = BgeEmbedder(MODEL_CACHE)

    with Session(get_engine()) as session:
        corpus = load_persisted_corpus(session)
        lexical = LexicalRetriever(corpus.chunks)
        semantic = DbSemanticRetriever(session, embedder, exact=True)
        hybrid = HybridRetriever(lexical, semantic, corpus_size=len(corpus.chunks))

        degradation_model = load_degradation_model(session)
        rul_regressor = load_rul_regressor(session)

        def engine_report_lookup(reference: EngineReference) -> EngineHealthReport:
            return build_engine_health_report(
                session,
                reference,
                degradation_model=degradation_model,
                rul_regressor=rul_regressor,
            )

        answer = answer_question(
            arguments.question,
            hybrid,
            provider,
            engine_report_lookup=engine_report_lookup,
        )

    print(json.dumps(answer.model_dump(), indent=2))


if __name__ == "__main__":
    main()

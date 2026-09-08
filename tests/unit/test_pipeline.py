"""Tests for the LangChain LCEL grounded-answer chain."""

from langchain_core.runnables import Runnable

from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import GroundedAnswer
from turbofan_copilot.llm.pipeline import build_answer_chain
from turbofan_copilot.llm.provider import ChatMessage, ResponseModel
from turbofan_copilot.retrieval.semantic_retrieval import ScoredChunk


class StaticProvider:
    """Return a fixed reply dict for any request."""

    def __init__(self, reply: dict[str, object]) -> None:
        self._reply = reply

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        return response_model.model_validate(self._reply)


_ANSWERS = {"answerable": True, "answer": "ok", "cited_passage_numbers": [1]}
_DECLINES = {"answerable": False, "answer": "", "cited_passage_numbers": []}


class StaticRetriever:
    """Return one Chapter 1 chunk for any question."""

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        chunk = DocumentChunk(
            source_id="faa-h-8083-32b-chapter-1",
            chapter_number=1,
            pdf_page_number=38,
            printed_page_label="1-38",
            chunk_index=1,
            text=f"evidence for {question}",
        )
        return ((0.9, chunk),)


def test_build_answer_chain_returns_an_invokable_runnable() -> None:
    chain = build_answer_chain(StaticRetriever(), StaticProvider(_ANSWERS))

    assert isinstance(chain, Runnable)
    result = chain.invoke("how do compressors work?")
    assert isinstance(result, GroundedAnswer)
    assert result.answer == "ok"
    assert result.abstained is False
    assert result.citations[0].printed_page_label == "1-38"


def test_chain_supports_lcel_batch() -> None:
    chain = build_answer_chain(StaticRetriever(), StaticProvider(_ANSWERS))

    answers = chain.batch(["question one", "question two"])

    assert [answer.answer for answer in answers] == ["ok", "ok"]
    assert all(isinstance(answer, GroundedAnswer) for answer in answers)


def test_no_evidence_branch_skips_the_model() -> None:
    class EmptyRetriever:
        def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
            return ()

    class ExplodingProvider:
        def complete(
            self,
            messages: list[ChatMessage],
            *,
            response_model: type[ResponseModel],
        ) -> ResponseModel:
            raise AssertionError("the model must not be called with no evidence")

    result = build_answer_chain(EmptyRetriever(), ExplodingProvider()).invoke("q")

    assert result.abstained is True
    assert result.citations == ()
    assert "No relevant" in result.answer


def test_model_declining_branch_returns_an_abstention() -> None:
    result = build_answer_chain(StaticRetriever(), StaticProvider(_DECLINES)).invoke(
        "what is the capital of France?"
    )

    assert result.abstained is True
    assert result.citations == ()
    assert result.abstention_reason is not None

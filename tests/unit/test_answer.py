"""Tests for the grounded answer pipeline with a fake language model."""

import pytest

from turbofan_copilot.health.engine_health import EngineHealthSummary, SensorTrend
from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import Citation, GroundedAnswer
from turbofan_copilot.llm.pipeline import answer_question
from turbofan_copilot.llm.provider import ChatMessage, ResponseModel
from turbofan_copilot.llm.router import EngineReference
from turbofan_copilot.llm.tools import EngineHealthReport, RulPrediction
from turbofan_copilot.retrieval.semantic_retrieval import ScoredChunk


class FakeLlmProvider:
    """Return a fixed reply and record the messages it was given."""

    def __init__(self, reply: dict[str, object]) -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        self.calls.append(messages)
        return response_model.model_validate(self._reply)


def reply(answer: str, *, cited: list[int], answerable: bool = True) -> dict[str, object]:
    """Build the dict a fake model returns."""
    return {"answerable": answerable, "answer": answer, "cited_passage_numbers": cited}


class FakeRetriever:
    """Return a fixed ranking regardless of the question."""

    def __init__(self, ranked: tuple[ScoredChunk, ...]) -> None:
        self._ranked = ranked
        self.questions: list[str] = []

    def rank(self, question: str, *, limit: int = 5) -> tuple[ScoredChunk, ...]:
        self.questions.append(question)
        return self._ranked[:limit]


def chunk(page: int, text: str, *, index: int = 1) -> DocumentChunk:
    return DocumentChunk(
        source_id="faa-h-8083-32b-chapter-6",
        chapter_number=6,
        pdf_page_number=page,
        printed_page_label=f"6-{page}",
        chunk_index=index,
        text=text,
    )


def fake_engine_report(unit_id: int = 47) -> EngineHealthReport:
    """A minimal valid engine-health report for pipeline tests."""
    trend = EngineHealthSummary(
        split="test",
        unit_id=unit_id,
        cycles_observed=120,
        cycles_analysed=30,
        first_analysed_cycle=91,
        latest_cycle=120,
        trends=(
            SensorTrend(
                sensor="sensor_4",
                slope_per_cycle=0.4,
                total_change=12.0,
                window_mean=1400.0,
                relative_change=0.0086,
            ),
        ),
        biggest_movers=("sensor_4",),
    )
    rul = RulPrediction(
        estimated_rul=38,
        model="boosted-trees",
        typical_error_cycles=8.4,
        cycles_observed=120,
    )
    return EngineHealthReport(trend=trend, rul=rul)


class FakeEngineLookup:
    """Return a fixed engine report and record which engines were asked about."""

    def __init__(self, report: EngineHealthReport) -> None:
        self._report = report
        self.calls: list[EngineReference] = []

    def __call__(self, reference: EngineReference) -> EngineHealthReport:
        self.calls.append(reference)
        return self._report


def test_answer_question_grounds_the_prompt_and_maps_citations() -> None:
    ranked: tuple[ScoredChunk, ...] = (
        (0.9, chunk(28, "The magnetic chip detector catches ferrous particles.")),
        (0.7, chunk(20, "Oil is stored in an external tank.")),
    )
    provider = FakeLlmProvider(reply("It catches ferrous debris.", cited=[1]))
    retriever = FakeRetriever(ranked)

    result = answer_question("how does a chip detector work?", retriever, provider)

    prompt = provider.calls[0][1].content
    assert "magnetic chip detector catches ferrous particles" in prompt
    assert "[1] (faa-h-8083-32b-chapter-6, 6-28)" in prompt
    assert result == GroundedAnswer(
        answer="It catches ferrous debris.",
        citations=(
            Citation(
                source_id="faa-h-8083-32b-chapter-6",
                pdf_page_number=28,
                printed_page_label="6-28",
            ),
        ),
    )


def test_answer_question_skips_the_model_when_nothing_is_retrieved() -> None:
    provider = FakeLlmProvider(reply("x", cited=[]))

    result = answer_question("anything", FakeRetriever(()), provider)

    assert result.abstained is True
    assert result.citations == ()
    assert "No relevant" in result.answer
    assert provider.calls == []


def test_answer_question_abstains_when_the_model_says_it_cannot_answer() -> None:
    ranked: tuple[ScoredChunk, ...] = ((0.4, chunk(28, "unrelated passage")),)
    provider = FakeLlmProvider(reply("", cited=[], answerable=False))

    result = answer_question("what is the capital of France?", FakeRetriever(ranked), provider)

    assert result.abstained is True
    assert result.citations == ()
    assert result.abstention_reason is not None
    assert len(provider.calls) == 1


def test_engine_data_is_fetched_and_shown_when_the_question_names_an_engine() -> None:
    provider = FakeLlmProvider(reply("Check the oil filter.", cited=[1]))
    lookup = FakeEngineLookup(fake_engine_report(47))
    ranked: tuple[ScoredChunk, ...] = ((0.9, chunk(28, "Inspect the magnetic chip detector.")),)

    result = answer_question(
        "engine 47 shows a worsening trend, what should be inspected?",
        FakeRetriever(ranked),
        provider,
        engine_report_lookup=lookup,
    )

    assert lookup.calls == [EngineReference(split="test", unit_id=47)]
    prompt = provider.calls[0][1].content
    assert "test engine #47" in prompt
    assert "Estimated remaining useful life: 38 cycles" in prompt
    # The prompt must mark the number as a prediction and carry its error, so the
    # model cannot relay it as a measurement.
    assert "PREDICTION" in prompt
    assert "boosted-trees" in prompt
    assert "about 8 cycles" in prompt
    assert "sensor_4 (+0.9%)" in prompt
    assert result.engine_health == fake_engine_report(47)
    assert [c.pdf_page_number for c in result.citations] == [28]


def test_engine_lookup_is_untouched_for_a_manual_only_question() -> None:
    provider = FakeLlmProvider(reply("a", cited=[]))
    lookup = FakeEngineLookup(fake_engine_report())
    ranked: tuple[ScoredChunk, ...] = ((0.5, chunk(28, "some passage")),)

    result = answer_question(
        "how does a magnetic chip detector work?",
        FakeRetriever(ranked),
        provider,
        engine_report_lookup=lookup,
    )

    assert lookup.calls == []
    assert result.engine_health is None
    assert "Engine sensor data" not in provider.calls[0][1].content


def test_answer_question_ignores_out_of_range_citation_numbers() -> None:
    ranked: tuple[ScoredChunk, ...] = ((0.5, chunk(28, "only passage")),)
    provider = FakeLlmProvider(reply("a", cited=[1, 2, 99]))

    result = answer_question("q", FakeRetriever(ranked), provider)

    assert [citation.pdf_page_number for citation in result.citations] == [28]


def test_grounded_answer_is_frozen_and_forbids_extras() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        GroundedAnswer.model_validate({"answer": "a", "citations": [], "surprise": 1})

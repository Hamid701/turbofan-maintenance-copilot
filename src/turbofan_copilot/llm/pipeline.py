"""The grounded-answer pipeline, composed as an explicit LangChain LCEL chain.

Every step is a named ``RunnableLambda``; the two ``RunnableBranch`` points are the
only control flow - one refuses before the model when nothing was retrieved, the
other refuses after the model when it reports the evidence does not answer.
``answer_question`` is a thin wrapper that invokes the chain.
"""

from dataclasses import dataclass, field

from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda

from turbofan_copilot.ingestion.chunking import DocumentChunk
from turbofan_copilot.llm.answer import (
    DEFAULT_EVIDENCE_LIMIT,
    NO_EVIDENCE_REASON,
    NOT_SUPPORTED_REASON,
    SYSTEM_PROMPT,
    EngineReportLookup,
    GroundedAnswer,
    ModelReply,
    abstention,
    build_citations,
    format_engine_report,
    format_evidence,
)
from turbofan_copilot.llm.provider import ChatMessage, LlmProvider
from turbofan_copilot.llm.router import EngineReference, detect_engine_reference
from turbofan_copilot.llm.tools import EngineHealthReport
from turbofan_copilot.retrieval.semantic_retrieval import SupportsRank


@dataclass
class _State:
    """What flows between the pipeline steps."""

    question: str
    reference: EngineReference | None = None
    report: EngineHealthReport | None = None
    chunks: list[DocumentChunk] = field(default_factory=list)
    reply: ModelReply | None = None


def build_answer_chain(
    retriever: SupportsRank,
    provider: LlmProvider,
    *,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    engine_report_lookup: EngineReportLookup | None = None,
) -> Runnable[str, GroundedAnswer]:
    """Compose the route -> retrieve -> (refuse | generate -> (refuse | cite)) pipeline."""

    def route(state: _State) -> _State:
        state.reference = detect_engine_reference(state.question)
        if state.reference is not None and engine_report_lookup is not None:
            state.report = engine_report_lookup(state.reference)
        return state

    def retrieve(state: _State) -> _State:
        state.chunks = [chunk for _, chunk in retriever.rank(state.question, limit=evidence_limit)]
        return state

    def generate(state: _State) -> _State:
        sections = [f"Question: {state.question}"]
        if state.chunks:
            sections.append(f"Evidence:\n{format_evidence(state.chunks)}")
        if state.reference is not None and state.report is not None:
            sections.append(format_engine_report(state.reference, state.report))
        state.reply = provider.complete(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content="\n\n".join(sections)),
            ],
            response_model=ModelReply,
        )
        return state

    def finalize(state: _State) -> GroundedAnswer:
        assert state.reply is not None
        return GroundedAnswer(
            answer=state.reply.answer,
            citations=build_citations(state.chunks, state.reply),
            engine_health=state.report,
        )

    generated: Runnable[_State, GroundedAnswer] = RunnableLambda(generate) | RunnableBranch(
        (
            lambda state: state.reply is None or not state.reply.answerable,
            RunnableLambda(lambda _state: abstention(NOT_SUPPORTED_REASON)),
        ),
        RunnableLambda(finalize),
    )
    return (
        RunnableLambda(lambda question: _State(question=question))
        | RunnableLambda(route)
        | RunnableLambda(retrieve)
        | RunnableBranch(
            (
                lambda state: not state.chunks and state.report is None,
                RunnableLambda(lambda _state: abstention(NO_EVIDENCE_REASON)),
            ),
            generated,
        )
    )


def answer_question(
    question: str,
    retriever: SupportsRank,
    provider: LlmProvider,
    *,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    engine_report_lookup: EngineReportLookup | None = None,
) -> GroundedAnswer:
    """Retrieve evidence, add engine data if the question names an engine, and answer."""
    chain = build_answer_chain(
        retriever,
        provider,
        evidence_limit=evidence_limit,
        engine_report_lookup=engine_report_lookup,
    )
    return chain.invoke(question)

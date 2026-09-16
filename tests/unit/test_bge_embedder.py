"""Tests for the local BGE encoding boundary.

The tokenizer, the model download, and the ONNX Runtime session are all replaced
with fakes, so these tests load no weights and need no network.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

import turbofan_copilot.retrieval.bge_embedder as bge_module
from turbofan_copilot.retrieval.bge_embedder import (
    EMBEDDING_DIMENSION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    ONNX_FILENAME,
    ONNX_OUTPUT,
    QUERY_PREFIX,
    BgeEmbedder,
)

INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


class FakeTokenizer:
    """Record text while returning small tokenizer-shaped values."""

    def __init__(self) -> None:
        self.counted_texts: list[str] = []
        self.encoded_batches: list[tuple[str, ...]] = []

    def __call__(self, text: str | list[str], **options: object) -> dict[str, object]:
        if isinstance(text, str):
            self.counted_texts.append(text)
            token_count = MAX_TOKENS + 1 if text == "too long" else 2
            return {"input_ids": list(range(token_count))}

        assert options["return_tensors"] == "np"
        self.encoded_batches.append(tuple(text))
        ones = np.ones((len(text), 2), dtype=np.int64)
        return {name: ones for name in INPUT_NAMES}


class FakeSession:
    """Return predictable CLS vectors without running a neural network."""

    def __init__(
        self, *, dimension: int = EMBEDDING_DIMENSION, cls: tuple[float, ...] = (3.0, 4.0)
    ):
        self.dimension = dimension
        self.cls = cls
        self.runs: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in INPUT_NAMES]

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs.append((output_names, feeds))
        batch = feeds["input_ids"].shape[0]
        token_vectors = np.zeros((batch, 2, self.dimension), dtype=np.float32)
        for index, value in enumerate(self.cls):
            token_vectors[:, 0, index] = value
        return [token_vectors]


def install_fakes(
    monkeypatch: MonkeyPatch,
    tokenizer: FakeTokenizer,
    session: FakeSession,
) -> dict[str, list[object]]:
    """Replace the tokenizer, download, and runtime; return what they were asked for."""
    calls: dict[str, list[object]] = {"tokenizer": [], "download": [], "session": []}

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"].append((args, kwargs))
            return tokenizer

    def fake_download(*args: object, **kwargs: object) -> str:
        calls["download"].append((args, kwargs))
        return "fake/model.onnx"

    def fake_session(path: str, **kwargs: object) -> FakeSession:
        calls["session"].append((path, kwargs))
        return session

    monkeypatch.setattr(bge_module, "BertTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(bge_module, "hf_hub_download", fake_download)
    monkeypatch.setattr(bge_module, "InferenceSession", fake_session)
    return calls


def test_loads_the_pinned_onnx_model_on_cpu_and_embeds_a_prefixed_question(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    session = FakeSession()
    calls = install_fakes(monkeypatch, tokenizer, session)

    embedder = BgeEmbedder(tmp_path)
    vector = embedder.embed_question("Why does the compressor stall?")

    assert calls["tokenizer"] == [
        ((MODEL_ID,), {"revision": MODEL_REVISION, "cache_dir": tmp_path})
    ]
    assert calls["download"] == [
        ((MODEL_ID, ONNX_FILENAME), {"revision": MODEL_REVISION, "cache_dir": tmp_path})
    ]
    assert calls["session"] == [("fake/model.onnx", {"providers": ["CPUExecutionProvider"]})]

    expected_text = QUERY_PREFIX + "Why does the compressor stall?"
    assert tokenizer.counted_texts == [expected_text]
    assert tokenizer.encoded_batches == [(expected_text,)]

    output_names, feeds = session.runs[0]
    assert output_names == [ONNX_OUTPUT]
    assert set(feeds) == set(INPUT_NAMES)
    assert all(feed.dtype == np.int64 for feed in feeds.values())

    # CLS vector (3, 4, 0, ...) normalized to unit length.
    assert len(vector) == EMBEDDING_DIMENSION
    assert vector[:2] == pytest.approx((0.6, 0.8))


def test_embeds_passages_without_prefix_in_limited_batches(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    session = FakeSession()
    install_fakes(monkeypatch, tokenizer, session)
    embedder = BgeEmbedder(tmp_path, batch_size=2)

    vectors = embedder.embed_passages(["first", "second", "third"])

    assert tokenizer.counted_texts == ["first", "second", "third"]
    assert tokenizer.encoded_batches == [("first", "second"), ("third",)]
    assert len(vectors) == 3
    assert len(session.runs) == 2


def test_rejects_long_passage_before_running_the_model(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    session = FakeSession()
    install_fakes(monkeypatch, tokenizer, session)
    embedder = BgeEmbedder(tmp_path)

    with pytest.raises(ValueError, match=r"passage 1 has 513 tokens; maximum is 512"):
        embedder.embed_passages(["too long"])

    assert session.runs == []


def test_rejects_wrong_embedding_dimension(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fakes(monkeypatch, FakeTokenizer(), FakeSession(dimension=3))
    embedder = BgeEmbedder(tmp_path)

    with pytest.raises(RuntimeError, match=r"embedding batch has shape \(1, 3\)"):
        embedder.embed_question("question")


def test_an_all_zero_vector_is_rejected_instead_of_dividing_by_zero(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A zero CLS vector cannot be normalized to unit length. The epsilon floor keeps
    # the arithmetic finite, and the unit-length check then refuses the result.
    install_fakes(monkeypatch, FakeTokenizer(), FakeSession(cls=(0.0,)))
    embedder = BgeEmbedder(tmp_path)

    with pytest.raises(RuntimeError, match="not unit length"):
        embedder.embed_question("question")


def test_rejects_blank_text_and_nonpositive_batch_size(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fakes(monkeypatch, FakeTokenizer(), FakeSession())

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        BgeEmbedder(tmp_path, batch_size=0)

    embedder = BgeEmbedder(tmp_path)
    with pytest.raises(ValueError, match="question must not be blank"):
        embedder.embed_question("   ")
    with pytest.raises(ValueError, match="passage 1 must not be blank"):
        embedder.embed_passages([""])

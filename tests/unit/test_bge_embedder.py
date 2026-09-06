"""Tests for the local BGE encoding boundary."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

import turbofan_copilot.retrieval.bge_embedder as bge_module
from turbofan_copilot.retrieval.bge_embedder import (
    EMBEDDING_DIMENSION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    QUERY_PREFIX,
    BgeEmbedder,
)


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

        self.encoded_batches.append(tuple(text))
        return {"input_ids": torch.ones((len(text), 2), dtype=torch.int64)}


class FakeModel:
    """Return predictable CLS vectors without loading a neural model."""

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.eval_called = False
        self.call_count = 0

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **model_inputs: object) -> SimpleNamespace:
        self.call_count += 1
        input_ids = cast(torch.Tensor, model_inputs["input_ids"])
        token_vectors = torch.zeros((input_ids.shape[0], 2, self.dimension))
        token_vectors[:, 0, 0] = 3.0
        token_vectors[:, 0, 1] = 4.0
        return SimpleNamespace(last_hidden_state=token_vectors)


def install_fakes(
    monkeypatch: MonkeyPatch,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> tuple[list[tuple[object, ...]], list[dict[str, object]]]:
    """Replace model loaders and return their recorded arguments."""
    positional_calls: list[tuple[object, ...]] = []
    keyword_calls: list[dict[str, object]] = []

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            positional_calls.append(args)
            keyword_calls.append(kwargs)
            return tokenizer

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            positional_calls.append(args)
            keyword_calls.append(kwargs)
            return model

    monkeypatch.setattr(bge_module, "BertTokenizer", FakeTokenizerLoader)
    monkeypatch.setattr(bge_module, "BertModel", FakeModelLoader)
    return positional_calls, keyword_calls


def test_loads_pinned_model_once_and_embeds_prefixed_question(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    positional_calls, keyword_calls = install_fakes(monkeypatch, tokenizer, model)

    embedder = BgeEmbedder(tmp_path)
    vector = embedder.embed_question("Why does the compressor stall?")

    assert positional_calls == [(MODEL_ID,), (MODEL_ID,)]
    assert keyword_calls == [
        {"revision": MODEL_REVISION, "cache_dir": tmp_path},
        {"revision": MODEL_REVISION, "cache_dir": tmp_path},
    ]
    assert model.eval_called
    expected_text = QUERY_PREFIX + "Why does the compressor stall?"
    assert tokenizer.counted_texts == [expected_text]
    assert tokenizer.encoded_batches == [(expected_text,)]
    assert len(vector) == EMBEDDING_DIMENSION
    assert vector[:2] == pytest.approx((0.6, 0.8))


def test_embeds_passages_without_prefix_in_limited_batches(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    install_fakes(monkeypatch, tokenizer, model)
    embedder = BgeEmbedder(tmp_path, batch_size=2)

    vectors = embedder.embed_passages(["first", "second", "third"])

    assert tokenizer.counted_texts == ["first", "second", "third"]
    assert tokenizer.encoded_batches == [("first", "second"), ("third",)]
    assert len(vectors) == 3
    assert model.call_count == 2


def test_rejects_long_passage_before_calling_model(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    install_fakes(monkeypatch, tokenizer, model)
    embedder = BgeEmbedder(tmp_path)

    with pytest.raises(ValueError, match=r"passage 1 has 513 tokens; maximum is 512"):
        embedder.embed_passages(["too long"])

    assert model.call_count == 0


def test_rejects_wrong_embedding_dimension(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(dimension=3)
    install_fakes(monkeypatch, tokenizer, model)
    embedder = BgeEmbedder(tmp_path)

    with pytest.raises(RuntimeError, match=r"embedding batch has shape \(1, 3\)"):
        embedder.embed_question("question")


def test_rejects_blank_text_and_nonpositive_batch_size(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    install_fakes(monkeypatch, tokenizer, model)

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        BgeEmbedder(tmp_path, batch_size=0)

    embedder = BgeEmbedder(tmp_path)
    with pytest.raises(ValueError, match="question must not be blank"):
        embedder.embed_question("   ")
    with pytest.raises(ValueError, match="passage 1 must not be blank"):
        embedder.embed_passages([""])

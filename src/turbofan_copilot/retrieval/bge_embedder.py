"""Local BGE text embeddings with explicit question and passage rules."""

from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32

type EmbeddingVector = tuple[float, ...]


class BgeEmbedder:
    """Load BGE once and encode questions and passages using its required rules."""

    def __init__(self, cache_dir: Path, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._batch_size = batch_size
        self._tokenizer = BertTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
        self._model = BertModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
        self._model.eval()

    def question_token_count(self, question: str) -> int:
        """Count a question after adding the BGE query prefix."""
        self._require_text(question, label="question")
        return self._count_tokens(QUERY_PREFIX + question)

    def passage_token_count(self, passage: str) -> int:
        """Count a passage without adding the query prefix."""
        self._require_text(passage, label="passage")
        return self._count_tokens(passage)

    def embed_question(self, question: str) -> EmbeddingVector:
        """Return one normalized question vector with the required BGE prefix."""
        token_count = self.question_token_count(question)
        self._require_token_limit(token_count, label="question")
        return self._encode_batch((QUERY_PREFIX + question,))[0]

    def embed_passages(self, passages: Iterable[str]) -> tuple[EmbeddingVector, ...]:
        """Return normalized passage vectors while limiting model batch size."""
        passage_batch = tuple(passages)
        for index, passage in enumerate(passage_batch, start=1):
            self._require_text(passage, label=f"passage {index}")
            self._require_token_limit(
                self._count_tokens(passage),
                label=f"passage {index}",
            )

        embeddings: list[EmbeddingVector] = []
        for start in range(0, len(passage_batch), self._batch_size):
            embeddings.extend(self._encode_batch(passage_batch[start : start + self._batch_size]))
        return tuple(embeddings)

    def _count_tokens(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def _encode_batch(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        model_inputs = self._tokenizer(
            list(texts),
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        with torch.inference_mode():
            token_vectors = self._model(**model_inputs).last_hidden_state
            cls_vectors = token_vectors[:, 0]
            normalized_vectors = torch.nn.functional.normalize(cls_vectors, p=2, dim=1)

        expected_shape = (len(texts), EMBEDDING_DIMENSION)
        if tuple(normalized_vectors.shape) != expected_shape:
            raise RuntimeError(
                f"embedding batch has shape {tuple(normalized_vectors.shape)}; "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(normalized_vectors).all()):
            raise RuntimeError("embedding batch contains a non-finite value")

        lengths = torch.linalg.vector_norm(normalized_vectors, dim=1)
        if not bool(torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)):
            raise RuntimeError("embedding batch contains a vector that is not unit length")

        return tuple(
            tuple(float(value) for value in vector.tolist()) for vector in normalized_vectors
        )

    @staticmethod
    def _require_text(text: str, *, label: str) -> None:
        if not text.strip():
            raise ValueError(f"{label} must not be blank")

    @staticmethod
    def _require_token_limit(token_count: int, *, label: str) -> None:
        if token_count > MAX_TOKENS:
            raise ValueError(f"{label} has {token_count} tokens; maximum is {MAX_TOKENS}")

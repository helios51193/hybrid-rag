from __future__ import annotations

import types
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from apps.rag.services import embeddings


class _FakeVec:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class EmbeddingsServiceTests(SimpleTestCase):
    def setUp(self) -> None:
        embeddings._ST_MODEL_CACHE.clear()
        embeddings._OPENAI_CLIENT_CACHE = None

    @override_settings(
        RAG_EMBEDDING_BACKEND="sentence_transformers",
        RAG_EMBEDDING_MODEL="fake-model",
        RAG_EMBEDDING_CACHE_ENABLED=True,
        RAG_EMBEDDING_CACHE_MAX_MODELS=2,
    )
    def test_sentence_transformer_is_cached_between_calls(self) -> None:
        init_calls: list[str] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, device: str | None = None) -> None:
                init_calls.append(f"{model_name}:{device}")

            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ARG002
                return [_FakeVec([0.1, 0.2, 0.3]) for _ in texts]

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            first = embeddings.embed_texts(["alpha"])
            second = embeddings.embed_texts(["beta"])

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    @override_settings(
        RAG_EMBEDDING_BACKEND="sentence_transformers",
        RAG_EMBEDDING_MODEL="fake-model",
        RAG_EMBEDDING_CACHE_ENABLED=False,
    )
    def test_sentence_transformer_is_not_cached_when_disabled(self) -> None:
        init_calls: list[str] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, device: str | None = None) -> None:
                init_calls.append(f"{model_name}:{device}")

            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ARG002
                return [_FakeVec([0.1, 0.2]) for _ in texts]

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            embeddings.embed_texts(["one"])
            embeddings.embed_texts(["two"])

        self.assertEqual(len(init_calls), 2)

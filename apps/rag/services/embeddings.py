from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from threading import Lock
from typing import Any

from django.conf import settings

from apps.rag.services.chunking import CodeChunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: CodeChunk
    vector: list[float]


_MODEL_CACHE_LOCK = Lock()
_ST_MODEL_CACHE: dict[str, Any] = {}
_OPENAI_CLIENT_CACHE: Any | None = None


def _deterministic_fallback_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Stable local fallback for tests/dev when no external embedding provider is wired.
    Not semantically meaningful, but deterministic and shape-stable.
    """
    digest = sha1(text.encode("utf-8")).digest()
    values: list[float] = []
    for i in range(dim):
        b = digest[i % len(digest)]
        values.append((b / 255.0) * 2.0 - 1.0)  # map to [-1, 1]
    return values


def _embedding_cache_enabled() -> bool:
    return bool(getattr(settings, "RAG_EMBEDDING_CACHE_ENABLED", True))


def _embedding_device() -> str | None:
    device = str(getattr(settings, "RAG_EMBEDDING_DEVICE", "") or "").strip()
    return device or None


def _max_cached_models() -> int:
    raw = int(getattr(settings, "RAG_EMBEDDING_CACHE_MAX_MODELS", 2))
    return max(1, raw)


def _get_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "RAG_EMBEDDING_BACKEND='sentence_transformers' requires "
            "'sentence-transformers' package."
        ) from exc

    if not _embedding_cache_enabled():
        device = _embedding_device()
        return SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)

    with _MODEL_CACHE_LOCK:
        cached = _ST_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        device = _embedding_device()
        encoder = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
        _ST_MODEL_CACHE[model_name] = encoder

        # Keep cache bounded to avoid runaway memory in long-lived workers.
        if len(_ST_MODEL_CACHE) > _max_cached_models():
            first_key = next(iter(_ST_MODEL_CACHE.keys()))
            if first_key != model_name:
                _ST_MODEL_CACHE.pop(first_key, None)

        return encoder


def _get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "RAG_EMBEDDING_BACKEND='openai' requires 'openai' package."
        ) from exc

    if not _embedding_cache_enabled():
        return OpenAI()

    global _OPENAI_CLIENT_CACHE
    with _MODEL_CACHE_LOCK:
        if _OPENAI_CLIENT_CACHE is None:
            _OPENAI_CLIENT_CACHE = OpenAI()
        return _OPENAI_CLIENT_CACHE


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return one embedding vector per input text (same order).
    """
    backend = getattr(settings, "RAG_EMBEDDING_BACKEND", "deterministic").lower()
    model = getattr(settings, "RAG_EMBEDDING_MODEL", "text-embedding-3-small")

    if backend == "deterministic":
        return [_deterministic_fallback_embedding(t) for t in texts]

    if backend == "sentence_transformers":
        encoder = _get_sentence_transformer(model)
        vectors = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    if backend == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    raise ValueError(
        "Unsupported RAG_EMBEDDING_BACKEND: "
        f"{backend}. Use one of deterministic|sentence_transformers|openai."
    )


def embed_chunks(chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    texts = [c.content for c in chunks]
    vectors = embed_texts(texts)

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count mismatch")

    return [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)]

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from django.conf import settings

from apps.rag.services.chunking import CodeChunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: CodeChunk
    vector: list[float]


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Return one embedding vector per input text (same order).
    """
    backend = getattr(settings, "RAG_EMBEDDING_BACKEND", "deterministic").lower()
    model = getattr(settings, "RAG_EMBEDDING_MODEL", "text-embedding-3-small")

    if backend == "deterministic":
        return [_deterministic_fallback_embedding(t) for t in texts]

    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG_EMBEDDING_BACKEND='sentence_transformers' requires "
                "'sentence-transformers' package."
            ) from exc

        encoder = SentenceTransformer(model)
        vectors = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "RAG_EMBEDDING_BACKEND='openai' requires 'openai' package."
            ) from exc

        client = OpenAI()
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

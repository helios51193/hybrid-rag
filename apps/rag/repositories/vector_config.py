from __future__ import annotations

import re
from hashlib import sha1

from django.conf import settings


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "default"


def get_vector_collection_name() -> str:
    """
    Build a stable, model-aware Qdrant collection name so different embedding
    backends/models do not collide on vector dimension.
    """
    base = getattr(settings, "RAG_VECTOR_COLLECTION", "rag_chunks")
    backend = getattr(settings, "RAG_EMBEDDING_BACKEND", "deterministic")
    model = getattr(settings, "RAG_EMBEDDING_MODEL", "text-embedding-3-small")

    model_slug = _slug(model)
    backend_slug = _slug(backend)
    short_hash = sha1(f"{backend}:{model}".encode("utf-8")).hexdigest()[:8]
    return f"{base}_{backend_slug}_{model_slug}_{short_hash}"

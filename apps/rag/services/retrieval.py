from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.services.embeddings import embed_texts


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    content: str


def vector_retrieve(
    *,
    project_id: str,
    query_text: str,
    vector_repo: QdrantRepository,
    top_k: int = 8,
) -> list[RetrievalHit]:
    query_vector = embed_texts([query_text])[0]
    hits = vector_repo.query(
        query_vector=query_vector,
        top_k=top_k,
        where={"project_id": project_id},
    )
    return [
        RetrievalHit(
            chunk_id=h.chunk_id,
            score=h.score,
            metadata=h.metadata,
            content=h.content,
        )
        for h in hits
    ]

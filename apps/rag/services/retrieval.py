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


def vector_retrieve_for_paths(
    *,
    project_id: str,
    query_text: str,
    paths: list[str],
    vector_repo: QdrantRepository,
    per_path_k: int = 2,
) -> list[RetrievalHit]:
    if not paths:
        return []

    query_vector = embed_texts([query_text])[0]
    all_hits: list[RetrievalHit] = []
    for path in paths:
        path = (path or "").strip()
        if not path:
            continue
        hits = vector_repo.query(
            query_vector=query_vector,
            top_k=per_path_k,
            where={"project_id": project_id, "relative_path": path},
        )
        for h in hits:
            all_hits.append(
                RetrievalHit(
                    chunk_id=h.chunk_id,
                    score=h.score,
                    metadata=h.metadata,
                    content=h.content,
                )
            )
    return all_hits
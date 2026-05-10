from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.services.graph_query import expand_related_files
from apps.rag.services.retrieval import RetrievalHit, vector_retrieve


@dataclass(frozen=True, slots=True)
class HybridHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    content: str
    source: str  # vector | graph


def run_hybrid_search(
    *,
    project_id: str,
    query_text: str,
    vector_repo: QdrantRepository,
    top_k: int = 8,
) -> list[HybridHit]:
    vector_hits = vector_retrieve(
        project_id=project_id,
        query_text=query_text,
        vector_repo=vector_repo,
        top_k=top_k,
    )

    graph_files = expand_related_files(project_id=project_id, seed_hits=vector_hits)
    graph_file_set = {g.file_path for g in graph_files}

    combined: list[HybridHit] = []
    for vh in vector_hits:
        path = str(vh.metadata.get("relative_path", ""))
        graph_bonus = 0.15 if path in graph_file_set else 0.0
        combined.append(
            HybridHit(
                chunk_id=vh.chunk_id,
                score=(0.85 * vh.score) + graph_bonus,
                metadata=vh.metadata,
                content=vh.content,
                source="hybrid" if graph_bonus > 0 else "vector",
            )
        )

    combined.sort(key=lambda item: item.score, reverse=True)
    return combined

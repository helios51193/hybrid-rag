from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.services.graph_query import expand_related_files
from apps.rag.services.retrieval import RetrievalHit, vector_retrieve, vector_retrieve_for_paths


@dataclass(frozen=True, slots=True)
class HybridHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    content: str
    source: str  # vector | graph | hybrid


def run_hybrid_search(
    *,
    project_id: str,
    query_text: str,
    vector_repo: QdrantRepository,
    top_k: int = 8,
) -> list[HybridHit]:
    # 1) Vector seeds.
    seed_hits = vector_retrieve(
        project_id=project_id,
        query_text=query_text,
        vector_repo=vector_repo,
        top_k=top_k,
    )

    # 2) Graph expansion from seeds.
    graph_files = expand_related_files(project_id=project_id, seed_hits=seed_hits)
    graph_scores_by_path = {g.file_path: g.graph_score for g in graph_files}

    # 3) Graph-driven vector retrieval from expanded files.
    graph_path_hits = vector_retrieve_for_paths(
        project_id=project_id,
        query_text=query_text,
        paths=list(graph_scores_by_path.keys()),
        vector_repo=vector_repo,
        per_path_k=2,
    )

    # 4) Merge + score.
    merged: dict[str, HybridHit] = {}

    # Vector-seed hits (strong vector, light graph bonus)
    for vh in seed_hits:
        path = str(vh.metadata.get("relative_path", "")).strip()
        graph_bonus = float(graph_scores_by_path.get(path, 0.0))
        score = (0.75 * vh.score) + (0.25 * graph_bonus)
        merged[vh.chunk_id] = HybridHit(
            chunk_id=vh.chunk_id,
            score=score,
            metadata=vh.metadata,
            content=vh.content,
            source="hybrid" if graph_bonus > 0 else "vector",
        )

    # Graph-path hits (explicit graph contribution)
    for gh in graph_path_hits:
        path = str(gh.metadata.get("relative_path", "")).strip()
        gscore = float(graph_scores_by_path.get(path, 0.0))
        score = (0.55 * gh.score) + (0.45 * gscore)
        existing = merged.get(gh.chunk_id)
        candidate = HybridHit(
            chunk_id=gh.chunk_id,
            score=score,
            metadata=gh.metadata,
            content=gh.content,
            source="graph" if gscore > 0 else "vector",
        )
        if existing is None or candidate.score > existing.score:
            merged[gh.chunk_id] = candidate

    combined = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return combined[: max(top_k * 2, 12)]
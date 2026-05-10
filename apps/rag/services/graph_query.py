from __future__ import annotations

from dataclasses import dataclass

from apps.rag.models import CodeEdge
from apps.rag.services.retrieval import RetrievalHit


@dataclass(frozen=True, slots=True)
class GraphExpandedFile:
    file_path: str
    graph_score: float


def expand_related_files(
    *,
    project_id: str,
    seed_hits: list[RetrievalHit],
    per_seed_limit: int = 6,
) -> list[GraphExpandedFile]:
    seed_paths = {
        str(hit.metadata.get("relative_path", "")).strip()
        for hit in seed_hits
        if hit.metadata.get("relative_path")
    }
    if not seed_paths:
        return []

    expanded: dict[str, float] = {}
    for seed in seed_paths:
        edges = (
            CodeEdge.objects.filter(project_id=project_id, source_node_id=seed)
            .order_by("id")[:per_seed_limit]
        )
        for edge in edges:
            if edge.target_node_id == seed:
                continue
            # Simple v1 graph score: reciprocal of local rank.
            rank_score = 1.0
            existing = expanded.get(edge.target_node_id, 0.0)
            expanded[edge.target_node_id] = max(existing, rank_score)

    return [
        GraphExpandedFile(file_path=path, graph_score=score)
        for path, score in expanded.items()
    ]

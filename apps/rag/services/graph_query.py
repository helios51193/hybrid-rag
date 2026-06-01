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
    per_seed_limit: int = 8,
    allowed_relations: set[str] | None = None,
) -> list[GraphExpandedFile]:
    if allowed_relations is None:
        allowed_relations = {"calls", "defines", "inherits", "test_targets", "imports"}

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
            .order_by("-weight", "id")[:per_seed_limit]
        )
        for edge in edges:
            if edge.target_node_id == seed:
                continue
            if edge.relation not in allowed_relations:
                continue

            # Relation-weighted graph score.
            relation_weight = float(edge.weight or 0.5)
            existing = expanded.get(edge.target_node_id, 0.0)
            expanded[edge.target_node_id] = max(existing, relation_weight)

    return [
        GraphExpandedFile(file_path=path, graph_score=score)
        for path, score in expanded.items()
    ]

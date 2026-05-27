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


def _query_tokens(query_text: str) -> set[str]:
    raw = (query_text or "").lower()
    return {t for t in raw.replace("?", " ").replace(",", " ").replace("-", " ").split() if t}


def _path_tokens(relative_path: str) -> set[str]:
    p = (relative_path or "").lower().replace("\\", "/")
    for ch in [".", "_", "-", "/"]:
        p = p.replace(ch, " ")
    return {t for t in p.split() if t}


def _intent_path_boost(query_text: str, relative_path: str) -> float:
    """
    Lightweight repository-agnostic path/query heuristics.
    Intentionally avoids library-specific hardcoded paths.
    """
    tokens = _query_tokens(query_text)
    path = (relative_path or "").lower().replace("\\", "/")
    p_tokens = _path_tokens(path)
    boost = 0.0

    # Semantic overlap between query words and path words.
    overlap = len(tokens & p_tokens)
    if overlap > 0:
        boost += min(0.2, 0.03 * overlap)

    # Encourage implementation locations for "where implemented/defined/handled" queries.
    implementation_intent = {"implemented", "implementation", "defined", "definition", "handled", "located", "where"}
    if tokens & implementation_intent:
        if "/tests/" in path or "/test/" in path:
            boost -= 0.08
        if "/docs/" in path or "/doc/" in path or "/examples/" in path or "/example/" in path:
            boost -= 0.12

    # Tests intent should prefer tests.
    test_intent = {"test", "tests", "tested", "testing", "coverage"}
    if tokens & test_intent:
        if "/tests/" in path or "/test/" in path:
            boost += 0.14
        else:
            boost -= 0.03

    # API/definition intent: prefer common source folders over docs/examples.
    api_intent = {"class", "function", "method", "api", "definition", "define"}
    if tokens & api_intent:
        if "/src/" in path or "/lib/" in path or "/core/" in path:
            boost += 0.05

    return max(-0.2, min(0.25, boost))


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
        score = (0.75 * vh.score) + (0.25 * graph_bonus) + _intent_path_boost(query_text, path)
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
        score = (0.55 * gh.score) + (0.45 * gscore) + _intent_path_boost(query_text, path)
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

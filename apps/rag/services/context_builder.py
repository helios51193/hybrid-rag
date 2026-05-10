from __future__ import annotations

from dataclasses import dataclass

from apps.rag.services.hybrid_search import HybridHit


@dataclass(frozen=True, slots=True)
class BuiltContext:
    contexts: list[str]
    citations: list[dict]


def build_context_and_citations(hits: list[HybridHit], max_items: int = 6) -> BuiltContext:
    selected = hits[:max_items]
    contexts = [h.content for h in selected]
    citations: list[dict] = []
    for h in selected:
        citations.append(
            {
                "chunk_id": h.chunk_id,
                "file_path": h.metadata.get("relative_path", ""),
                "start_line": h.metadata.get("start_line", 0),
                "end_line": h.metadata.get("end_line", 0),
                "score": round(h.score, 4),
                "retrieval_source": h.source,
            }
        )
    return BuiltContext(contexts=contexts, citations=citations)

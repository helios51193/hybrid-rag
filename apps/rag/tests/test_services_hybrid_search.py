from __future__ import annotations

from unittest.mock import patch

from django.test import SimpleTestCase

from apps.rag.services.graph_query import GraphExpandedFile
from apps.rag.services.hybrid_search import run_hybrid_search
from apps.rag.services.retrieval import RetrievalHit


class HybridSearchServiceTests(SimpleTestCase):
    def test_run_hybrid_search_includes_graph_path_candidates(self) -> None:
        seed_hits = [
            RetrievalHit(
                chunk_id="seed-1",
                score=0.9,
                metadata={"relative_path": "pkg/a.py"},
                content="seed chunk",
            )
        ]
        expanded_files = [GraphExpandedFile(file_path="pkg/related.py", graph_score=0.8)]
        graph_path_hits = [
            RetrievalHit(
                chunk_id="graph-only-1",
                score=0.7,
                metadata={"relative_path": "pkg/related.py"},
                content="graph retrieved chunk",
            )
        ]

        with (
            patch("apps.rag.services.hybrid_search.vector_retrieve", return_value=seed_hits),
            patch("apps.rag.services.hybrid_search.expand_related_files", return_value=expanded_files),
            patch("apps.rag.services.hybrid_search.vector_retrieve_for_paths", return_value=graph_path_hits),
        ):
            hits = run_hybrid_search(
                project_id="proj-1",
                query_text="where is auth handled?",
                vector_repo=object(),  # not used because retrieval calls are patched
                top_k=4,
            )

        ids = {h.chunk_id for h in hits}
        self.assertIn("seed-1", ids)
        self.assertIn("graph-only-1", ids)
        graph_hit = next(h for h in hits if h.chunk_id == "graph-only-1")
        self.assertEqual(graph_hit.source, "graph")

    def test_run_hybrid_search_prefers_higher_scored_duplicate(self) -> None:
        seed_hits = [
            RetrievalHit(
                chunk_id="dup-1",
                score=0.4,
                metadata={"relative_path": "pkg/a.py"},
                content="seed version",
            )
        ]
        expanded_files = [GraphExpandedFile(file_path="pkg/related.py", graph_score=0.9)]
        # Same chunk id appears in graph-path hits with stronger hybrid score.
        graph_path_hits = [
            RetrievalHit(
                chunk_id="dup-1",
                score=0.8,
                metadata={"relative_path": "pkg/related.py"},
                content="graph version",
            )
        ]

        with (
            patch("apps.rag.services.hybrid_search.vector_retrieve", return_value=seed_hits),
            patch("apps.rag.services.hybrid_search.expand_related_files", return_value=expanded_files),
            patch("apps.rag.services.hybrid_search.vector_retrieve_for_paths", return_value=graph_path_hits),
        ):
            hits = run_hybrid_search(
                project_id="proj-1",
                query_text="query",
                vector_repo=object(),
                top_k=4,
            )

        self.assertEqual(len([h for h in hits if h.chunk_id == "dup-1"]), 1)
        chosen = next(h for h in hits if h.chunk_id == "dup-1")
        self.assertEqual(chosen.source, "graph")
        self.assertGreater(chosen.score, 0.5)

from __future__ import annotations

from django.test import SimpleTestCase

from apps.rag.services.context_builder import build_context_and_citations
from apps.rag.services.hybrid_search import HybridHit


class ContextBuilderTests(SimpleTestCase):
    def test_build_context_and_citations_respects_max_items(self) -> None:
        hits = [
            HybridHit(
                chunk_id=f"c{i}",
                score=0.98765 - i * 0.1,
                metadata={"relative_path": f"pkg/file_{i}.py", "start_line": i + 1, "end_line": i + 10},
                content=f"chunk-content-{i}",
                source="vector",
            )
            for i in range(5)
        ]

        built = build_context_and_citations(hits, max_items=3)

        self.assertEqual(len(built.contexts), 3)
        self.assertEqual(len(built.citations), 3)
        self.assertEqual(built.contexts[0], "chunk-content-0")
        self.assertEqual(built.citations[0]["file_path"], "pkg/file_0.py")
        self.assertIn("score", built.citations[0])
        self.assertEqual(built.citations[0]["retrieval_source"], "vector")

    def test_build_context_and_citations_rounds_score(self) -> None:
        hits = [
            HybridHit(
                chunk_id="c1",
                score=0.123456,
                metadata={"relative_path": "x.py", "start_line": 1, "end_line": 2},
                content="ctx",
                source="hybrid",
            ),
        ]

        built = build_context_and_citations(hits, max_items=5)
        self.assertEqual(built.citations[0]["score"], 0.1235)

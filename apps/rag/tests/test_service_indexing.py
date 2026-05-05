from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from apps.rag.repositories.chroma_repository import ChromaRepository
from apps.rag.services.indexing import run_indexing


class IndexingServiceTests(SimpleTestCase):
    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.root = Path(self._tmp_dir.name)

        self.repo_dir = self.root / "repo"
        self.repo_dir.mkdir(parents=True, exist_ok=True)

        (self.repo_dir / "pkg").mkdir(parents=True, exist_ok=True)
        (self.repo_dir / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo_dir / "pkg" / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.repo_dir / "pkg" / "a.py").write_text(
            "import pkg.b\n\ndef run():\n    return pkg.b.VALUE\n",
            encoding="utf-8",
        )

        self.chroma_dir = self.root / "chroma"
        self.repo = ChromaRepository(
            collection_name="test_indexing_pipeline",
            persist_directory=str(self.chroma_dir),
        )

    def tearDown(self) -> None:
        self.repo.close()
        self._tmp_dir.cleanup()

    def test_run_indexing_returns_nonzero_pipeline_stats(self) -> None:
        result = run_indexing(
            project_id="proj-1",
            source_dir=str(self.repo_dir),
            chroma_repo=self.repo,
            chunk_size_lines=50,
            overlap_lines=10,
        )

        stats = result.stats
        self.assertEqual(stats.project_id, "proj-1")
        self.assertGreaterEqual(stats.documents_collected, 2)
        self.assertGreater(stats.chunks_created, 0)
        self.assertEqual(stats.vectors_upserted, stats.chunks_created)
        self.assertGreaterEqual(stats.graph_nodes, stats.documents_collected)
        self.assertGreaterEqual(stats.graph_edges, 1)
        self.assertGreaterEqual(stats.duration_seconds, 0.0)

    def test_run_indexing_persists_vectors_queryable_by_project(self) -> None:
        run_indexing(
            project_id="proj-2",
            source_dir=str(self.repo_dir),
            chroma_repo=self.repo,
        )

        # query with same deterministic embed shape (64 dims from fallback)
        query_vector = [0.1] * 64
        hits = self.repo.query(
            query_vector=query_vector,
            top_k=5,
            where={"project_id": "proj-2"},
        )
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0].metadata["project_id"], "proj-2")

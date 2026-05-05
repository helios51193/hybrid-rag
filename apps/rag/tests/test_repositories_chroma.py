from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from apps.rag.repositories.chroma_repository import ChromaRepository
from apps.rag.services.chunking import CodeChunk
from apps.rag.services.embeddings import EmbeddedChunk

import gc
import time
from chromadb.api.client import SharedSystemClient


def _embedded_chunk(
    *,
    chunk_id: str,
    content: str,
    vector: list[float],
    relative_path: str = "src/main.py",
    project_id: str = "proj-1",
) -> EmbeddedChunk:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        project_id=project_id,
        source_path="/tmp/repo",
        file_path=f"/tmp/repo/{relative_path}",
        relative_path=relative_path,
        language="python",
        chunk_index=0,
        start_line=1,
        end_line=3,
        content=content,
        token_estimate=max(1, len(content) // 4),
    )
    return EmbeddedChunk(chunk=chunk, vector=vector)


class ChromaRepositoryTests(SimpleTestCase):
    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.persist_dir = Path(self._tmp_dir.name) / "chroma"
        self.repo = ChromaRepository(
            collection_name="test_rag_chunks",
            persist_directory=str(self.persist_dir),
        )
        self.repo.connect()

    def tearDown(self) -> None:
        # Release references first
        self.repo._collection = None
        self.repo._client = None

        # Ask Chroma to clear shared client cache
        SharedSystemClient.clear_system_cache()

        # Force GC to release file handles on Windows
        gc.collect()

        # Retry cleanup briefly in case lock release is delayed
        for i in range(5):
            try:
                self._tmp_dir.cleanup()
                break
            except PermissionError:
                if i == 4:
                    raise
                time.sleep(0.2)


    def test_upsert_chunks_returns_count(self) -> None:
        items = [
            _embedded_chunk(chunk_id="c1", content="def add(a, b): return a + b", vector=[1.0, 0.0]),
            _embedded_chunk(chunk_id="c2", content="def sub(a, b): return a - b", vector=[0.0, 1.0]),
        ]

        count = self.repo.upsert_chunks(items)
        self.assertEqual(count, 2)

    def test_query_returns_ranked_hits(self) -> None:
        items = [
            _embedded_chunk(chunk_id="near", content="addition helper", vector=[1.0, 0.0]),
            _embedded_chunk(chunk_id="far", content="subtraction helper", vector=[0.0, 1.0]),
        ]
        self.repo.upsert_chunks(items)

        hits = self.repo.query(query_vector=[1.0, 0.0], top_k=2)

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "near")
        self.assertIn("relative_path", hits[0].metadata)
        self.assertEqual(hits[0].metadata["relative_path"], "src/main.py")
        self.assertIsInstance(hits[0].score, float)

    def test_query_with_where_filters_project(self) -> None:
        items = [
            _embedded_chunk(chunk_id="p1", content="project one", vector=[1.0, 0.0], project_id="proj-1"),
            _embedded_chunk(chunk_id="p2", content="project two", vector=[1.0, 0.0], project_id="proj-2"),
        ]
        self.repo.upsert_chunks(items)

        hits = self.repo.query(
            query_vector=[1.0, 0.0],
            top_k=5,
            where={"project_id": "proj-2"},
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "p2")
        self.assertEqual(hits[0].metadata["project_id"], "proj-2")

    def test_upsert_empty_list_returns_zero(self) -> None:
        self.assertEqual(self.repo.upsert_chunks([]), 0)

    def test_delete_project_then_query_returns_none_for_project(self) -> None:
        items = [
            _embedded_chunk(chunk_id="d1", content="to be deleted", vector=[1.0, 0.0], project_id="proj-del"),
            _embedded_chunk(chunk_id="k1", content="to be kept", vector=[0.0, 1.0], project_id="proj-keep"),
        ]
        self.repo.upsert_chunks(items)

        self.repo.delete_project("proj-del")

        deleted_hits = self.repo.query(
            query_vector=[1.0, 0.0],
            top_k=5,
            where={"project_id": "proj-del"},
        )
        kept_hits = self.repo.query(
            query_vector=[0.0, 1.0],
            top_k=5,
            where={"project_id": "proj-keep"},
        )

        self.assertEqual(deleted_hits, [])
        self.assertEqual(len(kept_hits), 1)
        self.assertEqual(kept_hits[0].chunk_id, "k1")

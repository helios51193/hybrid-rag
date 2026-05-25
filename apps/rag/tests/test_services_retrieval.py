from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from django.test import SimpleTestCase

from apps.rag.services.retrieval import vector_retrieve_for_paths


@dataclass(frozen=True, slots=True)
class _FakeVectorHit:
    chunk_id: str
    score: float
    metadata: dict
    content: str


class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, *, query_vector, top_k, where):  # noqa: ANN001
        self.calls.append({"query_vector": query_vector, "top_k": top_k, "where": where})
        path = where.get("relative_path", "")
        return [
            _FakeVectorHit(
                chunk_id=f"id:{path}",
                score=0.42,
                metadata={"relative_path": path, "project_id": where.get("project_id")},
                content=f"content for {path}",
            )
        ]


class RetrievalServiceTests(SimpleTestCase):
    def test_vector_retrieve_for_paths_ignores_blank_paths_and_queries_each_path(self) -> None:
        repo = _FakeRepo()
        with patch("apps.rag.services.retrieval.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            hits = vector_retrieve_for_paths(
                project_id="proj-1",
                query_text="where is auth",
                paths=["pkg/a.py", "", "  ", "pkg/b.py"],
                vector_repo=repo,
                per_path_k=3,
            )

        self.assertEqual(len(repo.calls), 2)
        self.assertEqual(repo.calls[0]["where"]["relative_path"], "pkg/a.py")
        self.assertEqual(repo.calls[1]["where"]["relative_path"], "pkg/b.py")
        self.assertEqual(repo.calls[0]["top_k"], 3)
        self.assertEqual(len(hits), 2)
        self.assertIn("pkg/a.py", {h.metadata["relative_path"] for h in hits})
        self.assertIn("pkg/b.py", {h.metadata["relative_path"] for h in hits})

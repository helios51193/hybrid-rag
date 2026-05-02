from __future__ import annotations

from django.test import SimpleTestCase

from apps.rag.services.chunking import chunk_document, chunk_documents
from apps.rag.services.ingestion import SourceDocument


def _make_doc(content: str) -> SourceDocument:
    return SourceDocument(
        project_id="proj-1",
        source_path="/tmp/project",
        file_path="/tmp/project/src/main.py",
        relative_path="src/main.py",
        language="python",
        content=content,
        size_bytes=len(content.encode("utf-8")),
    )


class ChunkingServiceTests(SimpleTestCase):
    def test_chunk_document_empty_content_returns_no_chunks(self) -> None:
        doc = _make_doc("")
        chunks = chunk_document(doc, chunk_size_lines=5, overlap_lines=1)
        self.assertEqual(chunks, [])

    def test_chunk_document_small_content_single_chunk(self) -> None:
        doc = _make_doc("line1\nline2\nline3\n")
        chunks = chunk_document(doc, chunk_size_lines=10, overlap_lines=2)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.end_line, 3)
        self.assertEqual(chunk.relative_path, "src/main.py")
        self.assertIn("line1", chunk.content)
        self.assertIn("line3", chunk.content)

    def test_chunk_document_overlapping_ranges(self) -> None:
        # 10 lines, size=4 overlap=1 => step=3
        # Expected windows: [1-4], [4-7], [7-10]
        content = "\n".join([f"line{i}" for i in range(1, 11)])
        doc = _make_doc(content)

        chunks = chunk_document(doc, chunk_size_lines=4, overlap_lines=1)

        self.assertEqual(len(chunks), 3)
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 4))
        self.assertEqual((chunks[1].start_line, chunks[1].end_line), (4, 7))
        self.assertEqual((chunks[2].start_line, chunks[2].end_line), (7, 10))

    def test_chunk_document_invalid_config_raises(self) -> None:
        doc = _make_doc("line1\nline2\n")

        with self.assertRaises(ValueError):
            chunk_document(doc, chunk_size_lines=0, overlap_lines=0)

        with self.assertRaises(ValueError):
            chunk_document(doc, chunk_size_lines=5, overlap_lines=-1)

        with self.assertRaises(ValueError):
            chunk_document(doc, chunk_size_lines=5, overlap_lines=5)

    def test_chunk_document_is_deterministic(self) -> None:
        doc = _make_doc("\n".join([f"x{i}" for i in range(1, 30)]))

        a = chunk_document(doc, chunk_size_lines=8, overlap_lines=2)
        b = chunk_document(doc, chunk_size_lines=8, overlap_lines=2)

        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])
        self.assertEqual([c.start_line for c in a], [c.start_line for c in b])
        self.assertEqual([c.end_line for c in a], [c.end_line for c in b])

    def test_chunk_documents_combines_multiple_docs(self) -> None:
        doc1 = _make_doc("a\nb\nc\n")
        doc2 = SourceDocument(
            project_id="proj-1",
            source_path="/tmp/project",
            file_path="/tmp/project/src/utils.py",
            relative_path="src/utils.py",
            language="python",
            content="u1\nu2\nu3\nu4\n",
            size_bytes=12,
        )

        chunks = chunk_documents([doc1, doc2], chunk_size_lines=2, overlap_lines=1)
        self.assertGreaterEqual(len(chunks), 5)
        self.assertIn("src/main.py", {c.relative_path for c in chunks})
        self.assertIn("src/utils.py", {c.relative_path for c in chunks})

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from apps.rag.services.ingestion import (
    MAX_FILE_SIZE_BYTES,
    collect_documents,
    discover_files,
    read_text_file,
)


class IngestionServiceTests(SimpleTestCase):
    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.tmp_root = Path(self._tmp_dir.name)

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def _write_text(self, rel_path: str, content: str) -> Path:
        path = self.tmp_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _write_bytes(self, rel_path: str, content: bytes) -> Path:
        path = self.tmp_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_discover_files_includes_supported_and_excludes_common_dirs(self) -> None:
        self._write_text("app/main.py", "print('ok')")
        self._write_text("docs/readme.md", "# hello")
        self._write_text("node_modules/pkg/index.js", "console.log('skip')")
        self._write_text(".git/config", "[core]")
        self._write_text("assets/logo.png", "not really png, but extension excluded")

        files = discover_files(self.tmp_root)

        rels = [p.relative_to(self.tmp_root).as_posix() for p in files]
        self.assertIn("app/main.py", rels)
        self.assertIn("docs/readme.md", rels)
        self.assertNotIn("node_modules/pkg/index.js", rels)
        self.assertNotIn(".git/config", rels)
        self.assertNotIn("assets/logo.png", rels)

    def test_discover_files_respects_exclude_globs(self) -> None:
        self._write_text("src/a.py", "a = 1")
        self._write_text("src/generated/b.py", "b = 2")
        self._write_text("tests/test_a.py", "def test_x(): pass")

        files = discover_files(
            self.tmp_root,
            exclude_globs=["src/generated/*", "tests/*"],
        )

        rels = [p.relative_to(self.tmp_root).as_posix() for p in files]
        self.assertIn("src/a.py", rels)
        self.assertNotIn("src/generated/b.py", rels)
        self.assertNotIn("tests/test_a.py", rels)

    def test_discover_files_skips_large_files(self) -> None:
        self._write_text("small.py", "x = 1")
        large_content = "a" * (MAX_FILE_SIZE_BYTES + 1)
        self._write_text("large.py", large_content)

        files = discover_files(self.tmp_root)
        rels = [p.relative_to(self.tmp_root).as_posix() for p in files]

        self.assertIn("small.py", rels)
        self.assertNotIn("large.py", rels)

    def test_read_text_file_returns_none_for_binary(self) -> None:
        binary_file = self._write_bytes("bin/data.bin", b"\x00\x01\x02\x03")
        self.assertIsNone(read_text_file(binary_file))

    def test_read_text_file_reads_utf8(self) -> None:
        text_file = self._write_text("src/unicode.py", "name = 'cafe'")
        content = read_text_file(text_file)
        self.assertEqual(content, "name = 'cafe'")

    def test_collect_documents_returns_normalized_documents(self) -> None:
        self._write_text("src/main.py", "def run():\n    return 1\n")
        self._write_text("README.md", "# Project\n")
        self._write_bytes("bin/blob.bin", b"\x00\x01")  # should be ignored

        docs = collect_documents(
            project_id="proj-1",
            source_dir=self.tmp_root,
        )

        self.assertEqual(len(docs), 2)

        by_rel = {d.relative_path: d for d in docs}
        self.assertIn("src/main.py", by_rel)
        self.assertIn("README.md", by_rel)

        py_doc = by_rel["src/main.py"]
        self.assertEqual(py_doc.project_id, "proj-1")
        self.assertEqual(py_doc.language, "python")
        self.assertEqual(Path(py_doc.file_path).name, "main.py")
        self.assertTrue(py_doc.source_path.endswith(self.tmp_root.name))
        self.assertIn("def run()", py_doc.content)
        self.assertGreater(py_doc.size_bytes, 0)

    def test_collect_documents_invalid_source_raises(self) -> None:
        invalid = self.tmp_root / "does-not-exist"
        with self.assertRaises(ValueError):
            collect_documents(project_id="proj-1", source_dir=invalid)

from __future__ import annotations

from django.test import SimpleTestCase

from apps.rag.services.graph_build import build_graph, extract_file_relations
from apps.rag.services.ingestion import SourceDocument


def _doc(relative_path: str, content: str, language: str = "python") -> SourceDocument:
    return SourceDocument(
        project_id="proj-1",
        source_path="/tmp/repo",
        file_path=f"/tmp/repo/{relative_path}",
        relative_path=relative_path,
        language=language,
        content=content,
        size_bytes=len(content.encode("utf-8")),
    )


class GraphBuildServiceTests(SimpleTestCase):
    def test_extract_file_relations_resolves_python_imports(self) -> None:
        docs = [
            _doc("pkg/__init__.py", ""),
            _doc("pkg/a.py", "import pkg.b\nfrom pkg import c"),
            _doc("pkg/b.py", "x = 1"),
            _doc("pkg/c.py", "y = 2"),
        ]

        edges = extract_file_relations(docs)
        triples = {(e.source, e.target, e.relation) for e in edges}

        self.assertIn(("pkg/a.py", "pkg/b.py", "imports"), triples)
        self.assertIn(("pkg/a.py", "pkg/c.py", "imports"), triples)

    def test_extract_file_relations_deduplicates_edges(self) -> None:
        docs = [
            _doc("pkg/a.py", "import pkg.b\nimport pkg.b\nfrom pkg import b"),
            _doc("pkg/b.py", "x = 1"),
        ]

        edges = extract_file_relations(docs)
        triples = [(e.source, e.target, e.relation) for e in edges]
        self.assertEqual(triples.count(("pkg/a.py", "pkg/b.py", "imports")), 1)

    def test_extract_file_relations_ignores_unresolved_and_non_python(self) -> None:
        docs = [
            _doc("pkg/a.py", "import external_lib\nfrom . import local_relative"),
            _doc("web/app.js", "import x from 'y';", language="javascript"),
        ]

        edges = extract_file_relations(docs)
        self.assertEqual(edges, [])

    def test_extract_file_relations_resolves_relative_single_dot_imports(self) -> None:
        docs = [
            _doc("pkg/a.py", "from . import b\nfrom .c import d"),
            _doc("pkg/b.py", "x = 1"),
            _doc("pkg/c.py", "y = 2"),
            _doc("pkg/c/d.py", "z = 3"),
        ]

        edges = extract_file_relations(docs)
        triples = {(e.source, e.target, e.relation) for e in edges}

        self.assertIn(("pkg/a.py", "pkg/b.py", "imports"), triples)
        self.assertIn(("pkg/a.py", "pkg/c.py", "imports"), triples)
        self.assertIn(("pkg/a.py", "pkg/c/d.py", "imports"), triples)

    def test_extract_file_relations_resolves_relative_double_dot_imports(self) -> None:
        docs = [
            _doc("pkg/sub/a.py", "from ..utils import helper"),
            _doc("pkg/utils.py", "VALUE = 1"),
            _doc("pkg/utils/helper.py", "def run(): pass"),
        ]

        edges = extract_file_relations(docs)
        triples = {(e.source, e.target, e.relation) for e in edges}

        self.assertIn(("pkg/sub/a.py", "pkg/utils.py", "imports"), triples)
        self.assertIn(("pkg/sub/a.py", "pkg/utils/helper.py", "imports"), triples)

    def test_build_graph_adds_all_document_nodes(self) -> None:
        docs = [
            _doc("pkg/a.py", ""),
            _doc("pkg/b.py", ""),
            _doc("README.md", "# hi", language="markdown"),
        ]

        graph = build_graph(docs, [])
        self.assertTrue(graph.has_node("pkg/a.py"))
        self.assertTrue(graph.has_node("pkg/b.py"))
        self.assertTrue(graph.has_node("README.md"))
        self.assertEqual(graph.nodes["pkg/a.py"]["node_type"], "file")

    def test_build_graph_adds_edges_with_relation(self) -> None:
        docs = [
            _doc("pkg/a.py", ""),
            _doc("pkg/b.py", ""),
        ]
        edges = extract_file_relations([
            _doc("pkg/a.py", "import pkg.b"),
            _doc("pkg/b.py", "x = 1"),
        ])

        graph = build_graph(docs, edges)
        self.assertTrue(graph.has_edge("pkg/a.py", "pkg/b.py"))
        self.assertEqual(graph.edges["pkg/a.py", "pkg/b.py"]["relation"], "imports")

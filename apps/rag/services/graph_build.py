from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

import networkx as nx

from apps.rag.services.ingestion import SourceDocument


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    relation: str = "imports"


def _build_python_module_index(documents: list[SourceDocument]) -> dict[str, str]:
    """
    Maps python module path -> relative file path.
    Example: src/pkg/utils.py -> pkg.utils (or src.pkg.utils if you keep src as package root)
    """
    index: dict[str, str] = {}
    for doc in documents:
        if doc.language != "python":
            continue

        rel = doc.relative_path.replace("\\", "/")
        p = PurePosixPath(rel)

        if p.suffix != ".py":
            continue

        # file module, e.g. pkg/utils.py -> pkg.utils
        parts = list(p.with_suffix("").parts)

        if not parts:
            continue

        # __init__.py maps to package module
        if parts[-1] == "__init__":
            mod = ".".join(parts[:-1])
            if mod:
                index[mod] = rel
        else:
            mod = ".".join(parts)
            index[mod] = rel

    return index


def _extract_python_import_modules(content: str) -> set[str]:
    modules: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # import a.b as c, x.y
        if line.startswith("import "):
            rest = line[len("import ") :]
            for part in rest.split(","):
                token = part.strip().split(" as ")[0].strip()
                if token:
                    modules.add(token)
            continue

        # from a.b import c, d as e
        if line.startswith("from ") and " import " in line:
            left, right = line[len("from ") :].split(" import ", 1)
            base = left.strip()
            if base and not base.startswith("."):  # ignore relative imports in v1
                modules.add(base)  # fallback to package/__init__.py if needed
                for imported in right.split(","):
                    name = imported.strip().split(" as ")[0].strip()
                    if not name or name == "*":
                        continue
                    modules.add(f"{base}.{name}")

    return modules


def _resolve_module_to_file(module: str, module_index: dict[str, str]) -> str | None:
    # Exact module match first.
    if module in module_index:
        return module_index[module]

    # Fall back: try parent modules (a.b.c -> a.b -> a)
    parts = module.split(".")
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_index:
            return module_index[candidate]
    return None


def extract_file_relations(documents: list[SourceDocument]) -> list[GraphEdge]:
    module_index = _build_python_module_index(documents)
    seen: set[tuple[str, str, str]] = set()
    edges: list[GraphEdge] = []

    for doc in documents:
        if doc.language != "python":
            continue

        source_rel = doc.relative_path.replace("\\", "/")
        imported_modules = _extract_python_import_modules(doc.content)

        for module in imported_modules:
            target_rel = _resolve_module_to_file(module, module_index)
            if not target_rel:
                continue
            if target_rel == source_rel:
                continue

            key = (source_rel, target_rel, "imports")
            if key in seen:
                continue
            seen.add(key)
            edges.append(GraphEdge(source=source_rel, target=target_rel, relation="imports"))

    return edges


def build_graph(documents: list[SourceDocument], edges: list[GraphEdge]) -> nx.DiGraph:
    graph = nx.DiGraph()

    # Add file nodes for all documents.
    for doc in documents:
        node_id = doc.relative_path.replace("\\", "/")
        graph.add_node(
            node_id,
            node_type="file",
            language=doc.language,
            file_path=node_id,
        )

    # Add directed edges.
    for edge in edges:
        if edge.source not in graph:
            graph.add_node(edge.source, node_type="file", file_path=edge.source)
        if edge.target not in graph:
            graph.add_node(edge.target, node_type="file", file_path=edge.target)

        graph.add_edge(edge.source, edge.target, relation=edge.relation)

    return graph

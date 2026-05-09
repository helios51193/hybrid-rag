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


def _module_parts_from_source_path(source_rel: str) -> list[str]:
    p = PurePosixPath(source_rel)
    parts = list(p.with_suffix("").parts)
    if not parts:
        return []
    return parts


def _resolve_relative_base_module(source_rel: str, relative_base: str) -> str | None:
    """
    Resolve relative import base against current source file.
    Examples:
      source=pkg/a.py, relative_base='.'      -> pkg
      source=pkg/a.py, relative_base='..x'    -> x (or parent.x when deeper)
      source=pkg/sub/a.py, relative_base='.b' -> pkg.sub.b
    """
    if not relative_base.startswith("."):
        return relative_base

    module_parts = _module_parts_from_source_path(source_rel)
    if not module_parts:
        return None

    # Current package parts (drop module leaf)
    package_parts = module_parts[:-1]

    dots = 0
    for ch in relative_base:
        if ch == ".":
            dots += 1
        else:
            break
    remainder = relative_base[dots:]
    remainder_parts = [part for part in remainder.split(".") if part]

    # from .x import ...  => stay in same package
    # from ..x import ... => one level up
    up_levels = max(0, dots - 1)
    if up_levels > len(package_parts):
        return None
    anchor_parts = package_parts[: len(package_parts) - up_levels]
    resolved_parts = anchor_parts + remainder_parts
    if not resolved_parts:
        return None
    return ".".join(resolved_parts)


def _extract_python_import_modules(content: str, source_rel: str) -> set[str]:
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
            if not base:
                continue

            resolved_base = _resolve_relative_base_module(source_rel=source_rel, relative_base=base)
            if not resolved_base:
                continue

            modules.add(resolved_base)  # fallback to package/__init__.py if needed
            for imported in right.split(","):
                name = imported.strip().split(" as ")[0].strip()
                if not name or name == "*":
                    continue
                modules.add(f"{resolved_base}.{name}")

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
        imported_modules = _extract_python_import_modules(doc.content, source_rel=source_rel)

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
            print("adding edge")
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

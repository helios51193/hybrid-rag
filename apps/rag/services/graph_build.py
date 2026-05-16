from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import networkx as nx

from apps.rag.services.ingestion import SourceDocument


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: str = "file"  # file | class | function | method | symbol_external
    language: str = ""
    file_path: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    relation_type: str = "imports"
    weight: float = 1.0
    evidence: dict[str, Any] | None = None

    @property
    def relation(self) -> str:
        # Backwards-compatible alias for older callsites/tests.
        return self.relation_type


RELATION_WEIGHTS: dict[str, float] = {
    "calls": 1.00,
    "defines": 0.95,
    "inherits": 0.90,
    "test_targets": 0.85,
    "uses_type": 0.75,
    "uses_symbol": 0.70,
    "imports": 0.55,
    "same_module": 0.35,
}


def _relation_weight(relation_type: str) -> float:
    return float(RELATION_WEIGHTS.get(relation_type, 0.5))


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


def _symbol_id(file_rel: str, symbol_name: str) -> str:
    return f"{file_rel}::{symbol_name}"


def _is_test_file(file_rel: str) -> bool:
    name = PurePosixPath(file_rel).name
    parts = set(PurePosixPath(file_rel).parts)
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts


def _imported_symbol_map(tree: ast.AST, source_rel: str, module_index: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                local = alias.asname or module_name.split(".")[-1]
                target_file = _resolve_module_to_file(module_name, module_index)
                if not target_file:
                    continue
                resolved[local] = target_file
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            base = ("." * int(node.level)) + (node.module or "")
            resolved_base = _resolve_relative_base_module(source_rel=source_rel, relative_base=base)
            if not resolved_base:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                candidate = f"{resolved_base}.{alias.name}"
                target_file = _resolve_module_to_file(candidate, module_index)
                if not target_file:
                    target_file = _resolve_module_to_file(resolved_base, module_index)
                if not target_file:
                    continue
                resolved[local] = target_file
    return resolved


def _line_evidence(node: ast.AST) -> dict[str, Any]:
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", start) or start)
    return {"line_start": start, "line_end": end}


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
            relation_type = "test_targets" if _is_test_file(source_rel) else "imports"
            edges.append(
                GraphEdge(
                    source=source_rel,
                    target=target_rel,
                    relation_type=relation_type,
                    weight=_relation_weight(relation_type),
                    evidence={"source_file": source_rel, "target_file": target_rel},
                )
            )

        try:
            tree = ast.parse(doc.content)
        except SyntaxError:
            continue

        imported_symbol_targets = _imported_symbol_map(tree, source_rel=source_rel, module_index=module_index)
        local_symbols: set[str] = set()

        # Defines + inherits + calls (symbol-level)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn_id = _symbol_id(source_rel, node.name)
                local_symbols.add(node.name)
                key = (source_rel, fn_id, "defines")
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        GraphEdge(
                            source=source_rel,
                            target=fn_id,
                            relation_type="defines",
                            weight=_relation_weight("defines"),
                            evidence={"symbol": node.name, **_line_evidence(node)},
                        )
                    )
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        called = None
                        if isinstance(inner.func, ast.Name):
                            called = inner.func.id
                        elif isinstance(inner.func, ast.Attribute):
                            called = inner.func.attr
                        if not called:
                            continue
                        if called in local_symbols:
                            target_id = _symbol_id(source_rel, called)
                        elif called in imported_symbol_targets:
                            target_id = imported_symbol_targets[called]
                        else:
                            continue
                        edge_key = (fn_id, target_id, "calls")
                        if edge_key in seen:
                            continue
                        seen.add(edge_key)
                        edges.append(
                            GraphEdge(
                                source=fn_id,
                                target=target_id,
                                relation_type="calls",
                                weight=_relation_weight("calls"),
                                evidence={"symbol": called, **_line_evidence(inner)},
                            )
                        )

            elif isinstance(node, ast.ClassDef):
                cls_id = _symbol_id(source_rel, node.name)
                local_symbols.add(node.name)
                key = (source_rel, cls_id, "defines")
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        GraphEdge(
                            source=source_rel,
                            target=cls_id,
                            relation_type="defines",
                            weight=_relation_weight("defines"),
                            evidence={"symbol": node.name, **_line_evidence(node)},
                        )
                    )

                for base in node.bases:
                    target_id = None
                    if isinstance(base, ast.Name):
                        if base.id in local_symbols:
                            target_id = _symbol_id(source_rel, base.id)
                        elif base.id in imported_symbol_targets:
                            target_id = imported_symbol_targets[base.id]
                    elif isinstance(base, ast.Attribute):
                        if base.attr in imported_symbol_targets:
                            target_id = imported_symbol_targets[base.attr]
                    if not target_id:
                        continue
                    edge_key = (cls_id, target_id, "inherits")
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)
                    edges.append(
                        GraphEdge(
                            source=cls_id,
                            target=target_id,
                            relation_type="inherits",
                            weight=_relation_weight("inherits"),
                            evidence={"base": ast.unparse(base) if hasattr(ast, "unparse") else "", **_line_evidence(base)},
                        )
                    )

                for child in node.body:
                    if not isinstance(child, ast.FunctionDef):
                        continue
                    method_id = _symbol_id(source_rel, f"{node.name}.{child.name}")
                    edge_key = (cls_id, method_id, "defines")
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append(
                            GraphEdge(
                                source=cls_id,
                                target=method_id,
                                relation_type="defines",
                                weight=_relation_weight("defines"),
                                evidence={"symbol": f"{node.name}.{child.name}", **_line_evidence(child)},
                            )
                        )
                    for inner in ast.walk(child):
                        if isinstance(inner, ast.Call):
                            called = None
                            if isinstance(inner.func, ast.Name):
                                called = inner.func.id
                            elif isinstance(inner.func, ast.Attribute):
                                called = inner.func.attr
                            if not called:
                                continue
                            if called in local_symbols:
                                target_id = _symbol_id(source_rel, called)
                            elif called in imported_symbol_targets:
                                target_id = imported_symbol_targets[called]
                            else:
                                continue
                            call_key = (method_id, target_id, "calls")
                            if call_key in seen:
                                continue
                            seen.add(call_key)
                            edges.append(
                                GraphEdge(
                                    source=method_id,
                                    target=target_id,
                                    relation_type="calls",
                                    weight=_relation_weight("calls"),
                                    evidence={"symbol": called, **_line_evidence(inner)},
                                )
                            )

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

    def _infer_node_type(node_id: str) -> str:
        if "::" not in node_id:
            return "file"
        symbol = node_id.split("::", 1)[1]
        if "." in symbol:
            return "method"
        if symbol and symbol[0].isupper():
            return "class"
        return "function"

    # Add directed edges.
    for edge in edges:
        if edge.source not in graph:
            graph.add_node(
                edge.source,
                node_type=_infer_node_type(edge.source),
                file_path=edge.source.split("::", 1)[0],
            )
        if edge.target not in graph:
            graph.add_node(
                edge.target,
                node_type=_infer_node_type(edge.target),
                file_path=edge.target.split("::", 1)[0],
            )

        graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation_type,
            relation_type=edge.relation_type,
            weight=edge.weight,
            evidence=edge.evidence or {},
        )

    return graph

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from django.db import transaction

from apps.rag.models import CodeEdge, CodeNode


@dataclass(frozen=True, slots=True)
class GraphPersistStats:
    nodes_saved: int
    edges_saved: int


class GraphRepository:
    @transaction.atomic
    def save_graph(self, *, project_id: str, graph: nx.DiGraph) -> GraphPersistStats:
        # Replace snapshot for a project to keep the state consistent with latest indexing run.
        CodeEdge.objects.filter(project_id=project_id).delete()
        CodeNode.objects.filter(project_id=project_id).delete()

        node_rows: list[CodeNode] = []
        for node_id, attrs in graph.nodes(data=True):
            node_rows.append(
                CodeNode(
                    project_id=project_id,
                    node_id=str(node_id),
                    node_type=str(attrs.get("node_type", "file")),
                    language=str(attrs.get("language", "")),
                    file_path=str(attrs.get("file_path", "")),
                    metadata={k: v for k, v in attrs.items() if k not in {"node_type", "language", "file_path"}},
                )
            )
        if node_rows:
            CodeNode.objects.bulk_create(node_rows, batch_size=1000)

        edge_rows: list[CodeEdge] = []
        for source, target, attrs in graph.edges(data=True):
            relation = str(attrs.get("relation", "imports"))
            weight = float(attrs.get("weight", 1.0))
            edge_rows.append(
                CodeEdge(
                    project_id=project_id,
                    source_node_id=str(source),
                    target_node_id=str(target),
                    relation=relation,
                    weight=weight,
                    metadata={k: v for k, v in attrs.items() if k not in {"relation", "weight"}},
                )
            )
        if edge_rows:
            CodeEdge.objects.bulk_create(edge_rows, batch_size=1000)

        return GraphPersistStats(nodes_saved=len(node_rows), edges_saved=len(edge_rows))

    def load_graph(self, *, project_id: str) -> nx.DiGraph:
        graph = nx.DiGraph()

        for node in CodeNode.objects.filter(project_id=project_id).iterator():
            attrs = dict(node.metadata or {})
            attrs.update(
                {
                    "node_type": node.node_type,
                    "language": node.language,
                    "file_path": node.file_path,
                }
            )
            graph.add_node(node.node_id, **attrs)

        for edge in CodeEdge.objects.filter(project_id=project_id).iterator():
            attrs = dict(edge.metadata or {})
            attrs.update({"relation": edge.relation, "weight": edge.weight})
            graph.add_edge(edge.source_node_id, edge.target_node_id, **attrs)

        return graph

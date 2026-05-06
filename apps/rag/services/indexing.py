from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import networkx as nx

from apps.rag.repositories.graph_repository import GraphRepository
from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.services.chunking import chunk_documents
from apps.rag.services.embeddings import embed_chunks
from apps.rag.services.graph_build import GraphEdge, build_graph, extract_file_relations
from apps.rag.services.ingestion import SourceDocument, collect_documents


@dataclass(frozen=True, slots=True)
class IndexingStats:
    project_id: str
    documents_collected: int
    chunks_created: int
    vectors_upserted: int
    graph_nodes: int
    graph_edges: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class IndexingResult:
    stats: IndexingStats
    documents: list[SourceDocument]
    edges: list[GraphEdge]
    graph: nx.DiGraph


def run_indexing(
    *,
    project_id: str,
    source_dir: str,
    vector_repo: QdrantRepository | None = None,
    chroma_repo: QdrantRepository | None = None,
    chunk_size_lines: int = 120,
    overlap_lines: int = 20,
    graph_repo: GraphRepository | None = None,
) -> IndexingResult:
    started = perf_counter()
    if vector_repo is None:
        vector_repo = chroma_repo
    if vector_repo is None:
        raise ValueError("A vector repository instance is required.")

    documents = collect_documents(project_id=project_id, source_dir=source_dir)

    chunks = chunk_documents(
        documents,
        chunk_size_lines=chunk_size_lines,
        overlap_lines=overlap_lines,
    )

    embedded_chunks = embed_chunks(chunks)

    vector_repo.connect()
    vectors_upserted = vector_repo.upsert_chunks(embedded_chunks)

    edges = extract_file_relations(documents)
    graph = build_graph(documents, edges)
    if graph_repo is not None:
        graph_repo.save_graph(project_id=project_id, graph=graph)

    stats = IndexingStats(
        project_id=project_id,
        documents_collected=len(documents),
        chunks_created=len(chunks),
        vectors_upserted=vectors_upserted,
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        duration_seconds=round(perf_counter() - started, 4),
    )

    return IndexingResult(
        stats=stats,
        documents=documents,
        edges=edges,
        graph=graph,
    )

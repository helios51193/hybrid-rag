from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.client import SharedSystemClient
from chromadb.api.models.Collection import Collection

from apps.rag.services.embeddings import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    content: str


class ChromaRepository:
    def __init__(self, collection_name: str, persist_directory: str = ".chroma") -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: chromadb.PersistentClient | None = None
        self._collection: Collection | None = None

    def connect(self) -> None:
        # idempotent connect
        if self._collection is not None:
            return

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """
        Explicitly release references and clear shared cache.
        Helps avoid Windows file lock issues in tests.
        """
        self._collection = None
        self._client = None
        SharedSystemClient.clear_system_cache()

    def _require_collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Chroma collection not initialized. Call connect() first.")
        return self._collection

    def upsert_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        if not embedded_chunks:
            return 0

        collection = self._require_collection()

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        for item in embedded_chunks:
            c = item.chunk
            ids.append(c.chunk_id)
            docs.append(c.content)
            vectors.append(item.vector)
            metas.append(
                {
                    "project_id": c.project_id,
                    "source_path": c.source_path,
                    "file_path": c.file_path,
                    "relative_path": c.relative_path,
                    "language": c.language,
                    "chunk_index": c.chunk_index,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "token_estimate": c.token_estimate,
                }
            )

        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        return len(ids)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        collection = self._require_collection()

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for chunk_id, meta, doc, dist in zip(ids, metas, docs, dists):
            # cosine distance -> similarity-like score
            score = 1.0 - float(dist)
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    score=score,
                    metadata=meta or {},
                    content=doc or "",
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def delete_project(self, project_id: str) -> int:
        collection = self._require_collection()
        collection.delete(where={"project_id": project_id})
        return 0

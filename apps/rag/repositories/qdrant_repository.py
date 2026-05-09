from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from apps.rag.services.embeddings import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    content: str


class QdrantRepository:
    def __init__(
        self,
        collection_name: str,
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: QdrantClient | None = None
        self._vector_size: int | None = None

    def connect(self) -> None:
        if self._client is not None:
            return
        self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)

    def close(self) -> None:
        self._client = None

    def _require_client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("Qdrant client is not initialized. Call connect() first.")
        return self._client

    def _ensure_collection(self, vector_size: int) -> None:
        client = self._require_client()
        if self._vector_size is None:
            self._vector_size = vector_size

        try:
            exists = client.collection_exists(self.collection_name)
        except Exception:
            # Fallback for older server/client combinations.
            exists = False
            try:
                client.get_collection(self.collection_name)
                exists = True
            except Exception:
                exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            )
            return

        # Validate vector size for existing collection
        collection_info = client.get_collection(self.collection_name)
        config = collection_info.config.params.vectors
        if hasattr(config, "size"):
            existing_size = int(config.size)
            if existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' vector size mismatch: "
                    f"existing={existing_size}, incoming={vector_size}"
                )

    @staticmethod
    def _payload_from_chunk(item: EmbeddedChunk) -> dict[str, Any]:
        c = item.chunk
        return {
            "project_id": c.project_id,
            "source_path": c.source_path,
            "file_path": c.file_path,
            "relative_path": c.relative_path,
            "language": c.language,
            "chunk_index": c.chunk_index,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "token_estimate": c.token_estimate,
            "content": c.content,
        }

    @staticmethod
    def _numeric_point_id(chunk_id: str) -> int:
        """
        Qdrant deployment here expects numeric point IDs.
        Use a deterministic 63-bit integer derived from chunk_id.
        """
        digest = sha1(chunk_id.encode("utf-8")).digest()
        # Positive signed 63-bit range
        return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)

    def upsert_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        if not embedded_chunks:
            return 0

        client = self._require_client()
        vector_size = len(embedded_chunks[0].vector)
        if vector_size <= 0:
            raise ValueError("Embedding vectors must be non-empty.")
        self._ensure_collection(vector_size=vector_size)

        points: list[qm.PointStruct] = []
        for item in embedded_chunks:
            points.append(
                qm.PointStruct(
                    id=self._numeric_point_id(item.chunk.chunk_id),
                    vector=item.vector,
                    payload=self._payload_from_chunk(item),
                )
            )

        client.upsert(collection_name=self.collection_name, wait=True, points=points)
        return len(points)

    @staticmethod
    def _build_filter(where: dict[str, Any] | None) -> qm.Filter | None:
        if not where:
            return None
        conditions: list[qm.FieldCondition] = []
        for key, value in where.items():
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        return qm.Filter(must=conditions)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        client = self._require_client()
        filt = self._build_filter(where)
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filt,
            with_payload=True,
        )

        results: list[VectorHit] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            content = str(payload.pop("content", ""))
            results.append(
                VectorHit(
                    chunk_id=str(hit.id),
                    score=float(hit.score),
                    metadata=payload,
                    content=content,
                )
            )
        return results

    def delete_project(self, project_id: str) -> int:
        client = self._require_client()
        project_filter = qm.Filter(
            must=[qm.FieldCondition(key="project_id", match=qm.MatchValue(value=project_id))]
        )
        client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(filter=project_filter),
            wait=True,
        )
        return 0

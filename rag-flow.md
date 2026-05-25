# Hybrid RAG Flow (Current)

## Objective

Build a code-intelligence assistant that answers natural-language questions over a repository using:

- Vector retrieval (Qdrant)
- Graph retrieval (NetworkX graph persisted in SQLite models)
- Grounded answer synthesis with citations

---

## Runtime Components

- Ingestion: `apps/rag/services/ingestion.py`
- Chunking: `apps/rag/services/chunking.py`
- Embeddings: `apps/rag/services/embeddings.py`
- Vector store: `apps/rag/repositories/qdrant_repository.py`
- Graph build: `apps/rag/services/graph_build.py`
- Graph persistence: `apps/rag/repositories/graph_repository.py`
- Retrieval:
  - `apps/rag/services/retrieval.py`
  - `apps/rag/services/graph_query.py`
  - `apps/rag/services/hybrid_search.py`
- Context assembly: `apps/rag/services/context_builder.py`
- Answer synthesis: `apps/rag/services/answering.py`
- Query views/UI: `apps/rag/views.py`, `apps/rag/templates/rag/...`

---

## Indexing Flow

1. Collect source documents from uploaded zip/folder.
2. Chunk each document into deterministic line-based segments.
3. Embed chunks using configured backend (`deterministic` | `sentence_transformers` | `openai`).
4. Upsert vectors to Qdrant (model-aware collection naming).
5. Build code graph with relation types:
   - `imports`, `test_targets`, `defines`, `calls`, `inherits`
6. Persist graph snapshot into `CodeNode`/`CodeEdge`.
7. Save indexing stats to `IndexingJob`.

---

## Query Flow (Dual Retrieval)

1. User submits `query_text` in conversation page.
2. Vector seed retrieval:
   - Retrieve top-k chunks from Qdrant for `project_id`.
3. Graph expansion:
   - Expand related file/symbol paths from `CodeEdge` using relation filtering.
4. Graph-path retrieval:
   - Query Qdrant again for expanded paths (`relative_path` filtered).
5. Merge + rank:
   - Combine seed vector hits and graph-path hits.
   - Deduplicate by `chunk_id`.
   - Use hybrid scoring to rank final candidates.
6. Build contexts + citations.
7. Generate answer (`fallback` or `openai`) with output contract.
8. Persist conversation turn:
   - user message
   - assistant message with citations
   - trace payload (includes answer contract and graph snapshot)
9. Return updated workspace (conversation, graph, citations) via HTMX partial.

---

## Graph UI Flow

1. On first query page open:
   - Use latest assistant-turn graph snapshot if present.
2. On each new assistant response:
   - Save query-scoped graph elements in assistant `trace_json`.
3. User can select historical answer:
   - `View This Graph` loads that turn’s graph + citations (`query_turn` endpoint).
4. Graph controls:
   - off-canvas drawer
   - relation filters
   - max visible node cap
   - one-hop expansion

---

## Citation Flow

1. Context builder returns top citations with `file_path`, `line range`, `score`, `retrieval_source`.
2. Citation panel shows compact list.
3. `View All` opens modal with scrollable full citation list.

---

## Key Settings

- Embedding backend/model:
  - `RAG_EMBEDDING_BACKEND`
  - `RAG_EMBEDDING_MODEL`
- Embedding cache/cold-start controls:
  - `RAG_EMBEDDING_CACHE_ENABLED`
  - `RAG_EMBEDDING_CACHE_MAX_MODELS`
  - `RAG_EMBEDDING_DEVICE`
- Answer synthesis:
  - `RAG_ANSWER_BACKEND`
  - `RAG_ANSWER_MODEL`
  - `RAG_ANSWER_TEMPERATURE`
- Qdrant:
  - `RAG_VECTOR_COLLECTION`
  - `QDRANT_URL`
  - `QDRANT_API_KEY`

---

## Current Strengths

- End-to-end indexing + query loop is functional.
- True dual retrieval (vector seeds + graph-path retrieval) is implemented.
- Conversation persistence includes per-turn graph/citation traceability.
- UI supports historical graph inspection per assistant turn.

---

## Current Gaps / Next Targets

1. Add stronger retrieval trace panel (seed hits vs graph hits contribution).
2. Improve hybrid score calibration with relation weights and hub penalties.
3. Add Qdrant + graph repository deeper integration tests.
4. Add embedding warmup command for first-query latency elimination.

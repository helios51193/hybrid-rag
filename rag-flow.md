# Hybrid RAG for Code Intelligence

## What This Project Does

This project builds a **hybrid retrieval-augmented generation (RAG)** system for source code.

Given a codebase, it:

1. indexes code chunks into a vector database (ChromaDB),
2. builds a structural graph of code relationships (NetworkX),
3. answers natural language questions using both semantic and graph-aware retrieval,
4. returns grounded citations to files and line ranges.

The goal is to make unfamiliar codebases easier to understand and query.

---

## Why Hybrid RAG

Vector search is strong at semantic similarity, but weak at structural reasoning.
Graph search is strong at structural relationships, but weak at semantic intent.

This project combines both:

- **Vector retrieval** finds semantically relevant code.
- **Graph expansion** pulls related implementation context (imports/references/neighbors).
- **Hybrid scoring** improves relevance and reduces missed context.

---

## Scope (Portfolio v1)

- Ingest a local repository.
- Chunk and embed source code.
- Store embeddings in ChromaDB.
- Build a code relationship graph with NetworkX.
- Execute hybrid retrieval for NL queries.
- Generate answers with citations.

Out of scope for v1:

- Production multi-tenant infra
- Advanced incremental indexing
- Deep language-specific static analysis across many languages

---

## Architecture (Current)

Single Django app: `apps/rag`

Main boundaries:

- `services/ingestion.py`: file discovery and extraction
- `services/chunking.py`: chunk creation + metadata
- `services/embeddings.py`: embedding generation
- `services/indexing.py`: indexing orchestration
- `repositories/chroma_repository.py`: vector persistence/retrieval
- `services/graph_build.py`: graph construction
- `services/graph_query.py`: neighborhood/path retrieval
- `repositories/graph_repository.py`: graph operations
- `services/hybrid_search.py`: vector + graph merge
- `services/rerank.py`: candidate scoring
- `services/context_builder.py`: final context assembly
- `services/answering.py`: answer + citations formatting
- `views.py` + templates: user interaction

---

## End-to-End Flow

1. **Ingest**
- Read code files from configured project path.
- Apply include/exclude rules.
- Skip binary/unsupported files.

2. **Chunk**
- Split files into deterministic chunks.
- Attach metadata (path, language, line range, symbol when available).

3. **Embed + Index**
- Generate embeddings for chunks.
- Upsert vectors + metadata into ChromaDB.

4. **Build Graph**
- Create nodes for files/symbols (v1 can start file-level only).
- Add edges such as `imports` / `references`.

5. **Retrieve (Hybrid)**
- Run vector top-k search.
- Expand from top results in graph (neighbors / hop-based expansion).
- Merge candidates.

6. **Rerank**
- Compute final score from semantic + graph signals.
- Select top contexts under token budget.

7. **Answer**
- Build response from selected contexts.
- Return citations (file + line range) for concrete claims.

---

## Contracts

## Query Input

- `project_id`
- `query_text`
- `top_k` (default: 10)
- `graph_hops` (default: 1)
- `debug` (default: false)

## Query Output

- `answer`
- `citations[]`
- `contexts[]`
- `debug` (optional retrieval trace)

Citation fields:

- `file_path`
- `start_line`
- `end_line`
- `chunk_id`
- `score`
- `retrieval_source` (`vector`, `graph`, `hybrid`)

---

## Metadata Schema (Chunk)

- `chunk_id`
- `project_id`
- `repo_snapshot_id` (or commit hash when available)
- `file_path`
- `language`
- `symbol_name` (nullable)
- `symbol_kind` (nullable)
- `start_line`
- `end_line`
- `chunk_type`
- `content_hash`
- `indexed_at`

---

## Hybrid Scoring (v1)

Initial weighted scoring:

`final_score = 0.75 * vector_score + 0.25 * graph_score`

Where:

- `vector_score`: normalized embedding similarity
- `graph_score`: neighborhood/path-based relevance heuristic

This weighting is intentionally simple and easy to explain, then tune.

---

## Design Tradeoffs I Chose

1. **Single Django app first**
- Faster iteration, less overhead, clearer ownership for solo development.

2. **Simple graph semantics in v1**
- Start with file-level relationships before complex call graphs.

3. **Deterministic chunking over aggressive optimization**
- Better reproducibility for debugging and evaluation.

4. **Debuggability as a feature**
- Retrieval traces and citations are first-class for learning and trust.

---

## How To Run Demo (Planned UX)

1. Open RAG page.
2. Register/select a local project path.
3. Trigger indexing.
4. Wait for indexing status to complete.
5. Ask a natural language question.
6. Review answer and citations.

Demo success criteria:

- Answer references correct files.
- Citations are traceable and meaningful.
- Hybrid retrieval surfaces context vector-only misses.

---

## Quality & Evaluation

Portfolio-level evaluation focus:

- Retrieval relevance on a fixed query set
- Citation correctness (file + line accuracy)
- Response groundedness (low hallucination)
- Latency for small/medium repos

Planned tests:

- Unit: chunking, graph edge extraction, reranking
- Integration: Chroma and graph repositories
- E2E smoke: index tiny repo -> query -> expected citation present

---

## Known Limitations (v1)

- Graph quality depends on lightweight parsing heuristics.
- Cross-language deep symbol resolution is limited.
- Large repository indexing performance is not fully optimized.
- Answer quality depends on embedding model and prompt strategy.

---

## Next Iterations

1. Incremental indexing by file hash changes.
2. Better symbol extraction and richer graph edges.
3. Query-time intent classification for retrieval strategy selection.
4. Evaluation harness with benchmark questions and scoring.
5. Async indexing pipeline and progress streaming UX.

---

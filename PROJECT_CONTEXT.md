# Project Context

## Project Goal

Build a hybrid RAG system for code intelligence using:

- Qdrant for vector retrieval
- NetworkX for graph retrieval
- Django as the application framework

The system should ingest a codebase and answer natural-language questions with grounded citations.

## Current Architecture Direction

- Main app: `apps/rag`
- Templates: Jinja (`.jinja`) with shared `base.jinja` planned
- Styling: Tailwind + DaisyUI via existing theme setup
- Static assets: shared root static folder (no app-local static folder)

## Implemented So Far

### Ingestion

- File discovery with include/exclude behavior
- Skip common non-source directories
- Size-limited file selection
- Binary detection and safe text decoding
- Normalized `SourceDocument` output

### Chunking

- Deterministic line-based chunking
- Configurable chunk size and overlap
- Stable chunk IDs
- Chunk metadata for downstream retrieval

### Graph Build (v1)

- File-level directed graph model
- Python import-based relation extraction (`imports`)
- Module-to-file resolution from indexed documents
- Relative import resolution (`from .x import y`, `from ..x import y`)
- Edge de-duplication and self-loop avoidance
- NetworkX `DiGraph` construction with node/edge metadata

### Graph Build (v2)

- Upgraded from import-only edges to multi-relation extraction for Python AST:
  - `imports`
  - `test_targets` (for test-file import links)
  - `defines` (file->class/function and class->method)
  - `calls` (function/method call relations)
  - `inherits` (class inheritance where resolvable)
- Added weighted edge schema in graph service:
  - relation type
  - relation weight
  - edge evidence (line metadata/symbol hints)
- Added symbol-level node IDs (`file.py::SymbolName`, `file.py::Class.method`) in graph build.
- Graph builder now infers node type (`file`, `class`, `function`, `method`) for symbol nodes.

### Vector + Indexing (v1)

- Embedding service supports configurable backends via Django settings:
  - `deterministic` (default fallback for tests/dev)
  - `sentence_transformers`
  - `openai`
- Embedding service now supports in-process model/client caching:
  - `RAG_EMBEDDING_CACHE_ENABLED`
  - `RAG_EMBEDDING_CACHE_MAX_MODELS`
  - `RAG_EMBEDDING_DEVICE`
- Qdrant repository implemented for:
  - collection connect/create
  - chunk upsert with payload metadata
  - top-k query with filtering
  - project-level delete
- Indexing orchestrator now runs:
  - collect documents
  - chunk documents
  - embed chunks
  - upsert vectors
  - build graph
  - return indexing stats/result payload
- Celery indexing worker now uses Qdrant repository + graph repository.

### Graph Persistence (SQLite via Django ORM)

- Added graph persistence models:
  - `CodeNode`
  - `CodeEdge`
- Added `IndexingJob` foreign keys on `CodeNode` and `CodeEdge` with cascade delete.
- Added DB-backed graph repository:
  - save graph snapshot per `project_id`
  - load graph back into `networkx.DiGraph`
- Indexing flow supports automatic graph persistence when `graph_repo` is provided.

### Dashboard UI (v1)

- Added dashboard page and reusable Jinja components under:
  - `templates/rag/pages/`
  - `templates/rag/components/`
- Added add-repository modal flow with Django form validation.
- Add-repository uses HTMX form submit and returns modal-only partial.
- Dashboard table refreshes via HTMX trigger (`repoListChanged`) instead of full page reload.
- Source input supports **either**:
  - zip upload, or
  - folder upload (`webkitdirectory`, multi-file)
- Added client-side field toggle so only selected source input is shown.
- Added CSRF header hook for HTMX actions.
- Switched DaisyUI theme to dark (`night`).
- Added processing row polling component with progress bar.
- Added custom DaisyUI delete-confirmation modal (replaced browser confirm).
- Process now reuses the same `IndexingJob` row instead of creating a new one.

### Query UI + Execution Pipeline (v1)

- Added fixed-frame dark query layout matching wireframe:
  - header (project + node/edge metadata)
  - left column: scrollable conversation + fixed question box
  - right column: fixed graph area + fixed citations area
- All major query areas are split into reusable components so they can be HTMX-updated independently.
- Added Cytoscape graph visualization with graph data injected as JSON.
- Added cached graph element loading in query view to avoid rebuilding graph on every query request.
- Added cache invalidation after successful indexing completion.
- Added HTMX query run endpoint and partial workspace response.
- Added minimal hybrid retrieval pipeline:
  - vector retrieval from Qdrant by `project_id`
  - graph-based related-file expansion from persisted `CodeEdge`
  - dual retrieval merge:
    - seed vector hits
    - graph-expanded path retrieval hits
  - hybrid score merge/ranking with dedupe
  - context + citation assembly
  - grounded answer synthesis with backend routing (`fallback` | `openai`)
- Added generalized query-intent path boosting in hybrid scoring (repository-agnostic heuristics).
- Added model-aware Qdrant collection naming based on embedding backend/model fingerprint.
- Added missing-collection-safe query/delete behavior in Qdrant repository.
- Query submit button now disables and shows loader during HTMX request.

### Query Answer Contract + Tests (v2)

- Added structured answer output contract in `services/answering.py`:
  - `contract_version`
  - `backend`
  - `status`
  - `answer_text`
  - `key_points`
  - `citations_doc_numbers`
  - `error_message`
- `query_run` now stores answer-contract trace in `ConversationMessage.trace_json`.
- Citation list for the UI now respects LLM-selected `[DOC n]` references when provided.
- Added tests:
  - `test_services_answering.py`
  - `test_services_context_builder.py`

### Query Graph UX + Performance (v2)

- First query page load now shows empty graph state (prevents rendering huge full graph immediately).
- Added query-scoped subgraph generation from citation seed files.
- Added one-hop graph expansion endpoint (`query/graph/expand/`) for progressive exploration.
- Added graph controls drawer (off-canvas overlay) with:
  - max visible node cap
  - relation visibility toggles
- Added node/edge legends and relation-specific edge styling in Cytoscape:
  - distinct colors
  - distinct line styles
  - distinct arrow shapes by relation type
- Added citation "View All" modal with scrollable long lists.
- Added per-assistant-turn graph history support:
  - graph snapshot persisted in assistant `trace_json`
  - refresh restores latest answer graph
  - per-answer "View This Graph" action loads historical graph + citations

### Explainability + Evaluation (v1)

- Added per-answer explainability modal launched from conversation items.
- Explainability data is loaded via HTMX partial from `query_explainability` endpoint.
- Assistant `trace_json` now stores retrieval trace summary:
  - final ranked hits
  - source mix (vector vs graph/hybrid)
  - answer contract snapshot
- Added evaluation harness:
  - `apps/rag/eval/eval_set.json` for benchmark queries
  - `manage.py rag_eval` command for retrieval benchmarking
  - optional `--warmup` mode to reduce cold-start latency distortion
- Eval output includes:
  - `file_hit_rate_at_k`
  - `mean_file_recall_at_k`
  - `symbol_hit_rate_at_k`
  - `mean_graph_or_hybrid_ratio`
  - `graph_helped_hit_rate_at_k` (queries where graph/hybrid hits succeeded while pure vector hits did not)
  - latency distribution (`p50`, `p95`, `mean`, `max`)
- Added eval comparison utility:
  - `manage.py rag_eval_compare --baseline <json> --candidate <json>`
  - aggregate delta summary + per-query improvement/regression listing

### Conversation Flow (v1)

- Query action now opens a conversation selector page.
- Added "Start New Conversation" flow that creates a unique conversation record.
- Added conversation runtime page requiring `repo_id` + `conversation_id`.
- Added conversation persistence models:
  - `Conversation`
  - `ConversationMessage`
- Query run now appends user + assistant messages and renders multi-turn history.
- Repository delete now also deletes conversations/messages for that project.

### Upload Safety Limits

- Added request/file size and count guardrails:
  - `DATA_UPLOAD_MAX_NUMBER_FILES`
  - `DATA_UPLOAD_MAX_MEMORY_SIZE`
  - `FILE_UPLOAD_MAX_MEMORY_SIZE`
- Added form-level friendly validation:
  - max folder file count
  - max total upload size (15 MB)

### Deletion Integrity

- Repository delete now removes:
  - all `IndexingJob` rows for the project
  - persisted graph rows (`CodeNode`/`CodeEdge`)
  - project vectors in Qdrant
  - uploaded source files under `var/uploads/repositories/<project_id>`
- Delete response triggers dashboard table refresh via HTMX event.

## Test Coverage Added

- `apps/rag/tests/test_services_ingestion.py`
  - discovery rules
  - exclusion behavior
  - binary handling
  - normalized document output
  - invalid source path errors

- `apps/rag/tests/test_services_chunking.py`
  - empty/small document behavior
  - overlap windows and line ranges
  - invalid configuration validation
  - deterministic chunk IDs
  - multi-document aggregation

- `apps/rag/tests/test_services_graph_build.py`
  - resolvable import relations
  - edge de-duplication
  - unresolved/non-python handling
  - graph node and edge construction

- `apps/rag/tests/test_services_answering.py`
  - fallback contract behavior
  - unknown backend handling
  - OpenAI JSON parse path
  - OpenAI non-JSON fallback path

- `apps/rag/tests/test_services_context_builder.py`
  - max item selection
  - citation score rounding/shape

- `apps/rag/tests/test_services_embeddings.py`
  - sentence-transformer caching enabled/disabled behavior

- `apps/rag/tests/test_services_hybrid_search.py`
  - dual retrieval includes graph-path candidates
  - duplicate merge keeps stronger candidate

- `apps/rag/tests/test_services_retrieval.py`
  - path-based retrieval filtering/query behavior

- `apps/rag/tests/test_views_query_graph_history.py`
  - latest graph restore on query page
  - historical turn graph load via `query_turn`

## Notable Fixes During Development

- Switched ingestion tests to `TemporaryDirectory` for safe isolation.
- Updated a Windows-sensitive path assertion to use `Path(...)` semantics.
- Identified and fixed `from pkg import x` relation resolution gap by including `pkg.x` during import extraction.

## Immediate Next Steps

1. Add tests for Qdrant repository query/filter/delete behavior (including missing collection fallback).
2. Add tests for graph repository save/load behavior and relation distribution sanity.
3. Tune dual-retrieval scoring incrementally and keep `rag_eval_compare` as a guardrail for regressions.
4. Add optional local LLM backend path (OpenAI-compatible base URL) for offline generation runs.
5. Add optional model warmup command for embedding cold-start elimination before first query.

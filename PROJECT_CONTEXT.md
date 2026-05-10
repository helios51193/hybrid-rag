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

### Vector + Indexing (v1)

- Embedding service supports configurable backends via Django settings:
  - `deterministic` (default fallback for tests/dev)
  - `sentence_transformers`
  - `openai`
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
  - hybrid score merge/ranking
  - context + citation assembly
  - placeholder answer synthesis from retrieved context

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

## Notable Fixes During Development

- Switched ingestion tests to `TemporaryDirectory` for safe isolation.
- Updated a Windows-sensitive path assertion to use `Path(...)` semantics.
- Identified and fixed `from pkg import x` relation resolution gap by including `pkg.x` during import extraction.

## Immediate Next Steps

1. Add persistent query history model (`QueryLog`) and render multi-turn chat history.
2. Replace placeholder answer synthesis with LLM-based grounded response generation.
3. Add tests for Qdrant repository query/filter/delete behavior.
4. Add tests for graph repository save/load behavior.
5. Add tests for query execution endpoint and HTMX partial rendering flow.

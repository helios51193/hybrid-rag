# Project Context

## Project Goal

Build a hybrid RAG system for code intelligence using:

- ChromaDB for vector retrieval
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
- Edge de-duplication and self-loop avoidance
- NetworkX `DiGraph` construction with node/edge metadata

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

1. Finish and stabilize `graph_build.py` import parsing behavior.
2. Implement `graph_query.py` for neighborhood/path retrieval.
3. Add Chroma repository integration for vector upsert/search.
4. Implement `hybrid_search.py` merge and weighted scoring.
5. Start thin UI after service pipeline is stable end-to-end.

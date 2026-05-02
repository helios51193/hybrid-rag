from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from apps.rag.services.ingestion import SourceDocument


@dataclass(frozen=True, slots=True)
class CodeChunk:
    chunk_id: str
    project_id: str
    source_path: str
    file_path: str
    relative_path: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    # Lightweight heuristic for v1.
    return max(1, len(text) // 4)


def _make_chunk_id(
    project_id: str,
    relative_path: str,
    start_line: int,
    end_line: int,
    content: str,
) -> str:
    raw = f"{project_id}|{relative_path}|{start_line}|{end_line}|{content}"
    return sha1(raw.encode("utf-8")).hexdigest()


def chunk_document(
    doc: SourceDocument,
    chunk_size_lines: int = 120,
    overlap_lines: int = 20,
) -> list[CodeChunk]:
    if chunk_size_lines <= 0:
        raise ValueError("chunk_size_lines must be > 0")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be >= 0")
    if overlap_lines >= chunk_size_lines:
        raise ValueError("overlap_lines must be < chunk_size_lines")

    lines = doc.content.splitlines()
    if not lines:
        return []

    step = chunk_size_lines - overlap_lines
    chunks: list[CodeChunk] = []

    start_idx = 0
    chunk_index = 0
    total = len(lines)

    while start_idx < total:
        end_exclusive = min(start_idx + chunk_size_lines, total)
        chunk_lines = lines[start_idx:end_exclusive]
        chunk_text = "\n".join(chunk_lines).strip()

        if chunk_text:
            start_line = start_idx + 1
            end_line = end_exclusive
            chunk_id = _make_chunk_id(
                project_id=doc.project_id,
                relative_path=doc.relative_path,
                start_line=start_line,
                end_line=end_line,
                content=chunk_text,
            )
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    project_id=doc.project_id,
                    source_path=doc.source_path,
                    file_path=doc.file_path,
                    relative_path=doc.relative_path,
                    language=doc.language,
                    chunk_index=chunk_index,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_text,
                    token_estimate=_estimate_tokens(chunk_text),
                )
            )
            chunk_index += 1

        if end_exclusive == total:
            break

        start_idx += step

    return chunks


def chunk_documents(
    docs: list[SourceDocument],
    chunk_size_lines: int = 120,
    overlap_lines: int = 20,
) -> list[CodeChunk]:
    all_chunks: list[CodeChunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                doc=doc,
                chunk_size_lines=chunk_size_lines,
                overlap_lines=overlap_lines,
            )
        )
    return all_chunks

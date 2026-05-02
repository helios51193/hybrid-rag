from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt",
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".html", ".css", ".scss",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".sh",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB (v1 safety limit)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    project_id: str
    source_path: str           # absolute path to project root
    file_path: str             # absolute file path
    relative_path: str         # path relative to source_path
    language: str
    content: str
    size_bytes: int


def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext == ".md":
        return "markdown"
    if ext == ".json":
        return "json"
    if ext == ".toml":
        return "toml"
    if ext in {".html"}:
        return "html"
    if ext in {".css", ".scss"}:
        return "css"
    return "text"


def _is_probably_binary(data: bytes) -> bool:
    # NUL byte is a strong binary signal.
    return b"\x00" in data


def discover_files(
    source_dir: str | Path,
    include_globs: Iterable[str] | None = None,
    exclude_globs: Iterable[str] | None = None,
) -> list[Path]:
    root = Path(source_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid source directory: {root}")

    include_globs = list(include_globs or ["**/*"])
    exclude_globs = set(exclude_globs or [])

    results: list[Path] = []
    for include_pattern in include_globs:
        for path in root.glob(include_pattern):
            if not path.is_file():
                continue

            if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            rel = path.relative_to(root).as_posix()
            if any(path.match(pattern) or rel.startswith(pattern.rstrip("/")) for pattern in exclude_globs):
                continue

            try:
                size = path.stat().st_size
            except OSError:
                continue

            if size > MAX_FILE_SIZE_BYTES:
                continue

            results.append(path)

    # Deduplicate + stable order
    return sorted(set(results))


def read_text_file(path: str | Path) -> str | None:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        return None

    if _is_probably_binary(raw):
        return None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def collect_documents(
    project_id: str,
    source_dir: str | Path,
    include_globs: Iterable[str] | None = None,
    exclude_globs: Iterable[str] | None = None,
) -> list[SourceDocument]:
    root = Path(source_dir).resolve()
    files = discover_files(
        source_dir=root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )

    documents: list[SourceDocument] = []
    for file_path in files:
        text = read_text_file(file_path)
        if text is None:
            continue

        try:
            size_bytes = file_path.stat().st_size
        except OSError:
            continue

        documents.append(
            SourceDocument(
                project_id=project_id,
                source_path=str(root),
                file_path=str(file_path.resolve()),
                relative_path=file_path.relative_to(root).as_posix(),
                language=_guess_language(file_path),
                content=text,
                size_bytes=size_bytes,
            )
        )

    return documents

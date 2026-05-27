from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.repositories.vector_config import get_vector_collection_name
from apps.rag.services.context_builder import build_context_and_citations
from apps.rag.services.hybrid_search import run_hybrid_search


@dataclass(frozen=True, slots=True)
class QueryEvalResult:
    eval_id: str
    query: str
    difficulty: str
    top_k: int
    latency_ms: float
    retrieved_files: list[str]
    expected_files: list[str]
    expected_symbols: list[str]
    file_hit_at_k: int
    file_recall_at_k: float
    symbol_hit_at_k: int
    graph_or_hybrid_ratio: float


def _normalize_path(value: str) -> str:
    return (value or "").replace("\\", "/").strip().lower()


def _canonical_code_path(value: str) -> str:
    """
    Normalize paths to a code-relative anchor so eval set paths can match
    retrieved paths that include repo-folder prefixes.
    Example:
      retrieved: "networkx-main/networkx/algorithms/dag.py"
      expected:  "networkx/algorithms/dag.py"
      -> both canonicalize to "networkx/algorithms/dag.py"
    """
    p = _normalize_path(value)
    if not p:
        return p

    anchors = ("networkx/", "apps/", "src/")
    for anchor in anchors:
        idx = p.find(anchor)
        if idx >= 0:
            return p[idx:]
    return p


def _file_match(expected: str, retrieved: str) -> bool:
    exp = _canonical_code_path(expected)
    got = _canonical_code_path(retrieved)
    if not exp or not got:
        return False

    # Exact match for file paths.
    if exp == got:
        return True

    # Prefix match for directory-level expectations in dataset.
    # Example: "networkx/algorithms/community" should match any file under that directory.
    if "." not in Path(exp).name:
        return got.startswith(exp.rstrip("/") + "/") or got == exp.rstrip("/")

    return False


def _symbol_match(expected_symbols: list[str], contexts: list[str]) -> int:
    if not expected_symbols:
        return 0
    haystack = "\n".join(contexts).lower()
    for symbol in expected_symbols:
        sym = (symbol or "").strip().lower()
        if sym and sym in haystack:
            return 1
    return 0


class Command(BaseCommand):
    help = "Run retrieval evaluation harness against apps/rag/eval/eval_set.json"

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument("--project-id", required=True, help="Indexed project_id to evaluate against.")
        parser.add_argument(
            "--eval-file",
            default=str(Path("apps/rag/eval/eval_set.json")),
            help="Path to evaluation JSON file.",
        )
        parser.add_argument("--top-k", type=int, default=8, help="Retrieval top_k for hybrid search.")
        parser.add_argument(
            "--context-k",
            type=int,
            default=8,
            help="Number of final contexts to evaluate symbols against.",
        )
        parser.add_argument(
            "--output",
            default="",
            help="Optional output JSON path; default writes to logs/eval/",
        )
        parser.add_argument(
            "--warmup",
            action="store_true",
            help="Run one unmeasured warmup retrieval per query before timing.",
        )

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        project_id: str = options["project_id"]
        eval_file = Path(options["eval_file"])
        top_k = int(options["top_k"])
        context_k = int(options["context_k"])
        output_path_raw = (options.get("output") or "").strip()
        warmup = bool(options.get("warmup", False))

        if not eval_file.exists():
            raise CommandError(f"Evaluation file not found: {eval_file}")

        try:
            eval_rows: list[dict[str, Any]] = json.loads(eval_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON in eval file: {exc}") from exc

        if not isinstance(eval_rows, list) or not eval_rows:
            raise CommandError("Evaluation file must be a non-empty JSON list.")

        vector_repo = QdrantRepository(
            collection_name=get_vector_collection_name(),
            url=getattr(settings, "QDRANT_URL", "http://127.0.0.1:6333"),
            api_key=getattr(settings, "QDRANT_API_KEY", None),
            timeout=getattr(settings, "QDRANT_TIMEOUT", 30),
        )

        results: list[QueryEvalResult] = []
        latencies: list[float] = []
        total_file_hits = 0
        total_symbol_hits = 0
        total_graph_ratio = 0.0

        vector_repo.connect()
        try:
            for row in eval_rows:
                eval_id = str(row.get("id", "")).strip() or "unknown"
                query = str(row.get("query", "")).strip()
                if not query:
                    self.stdout.write(self.style.WARNING(f"Skipping {eval_id}: empty query"))
                    continue

                expected_files = [str(x).strip() for x in row.get("expected_files", []) if str(x).strip()]
                expected_symbols = [str(x).strip() for x in row.get("expected_symbols", []) if str(x).strip()]
                difficulty = str(row.get("difficulty", "unknown")).strip() or "unknown"

                if warmup:
                    run_hybrid_search(
                        project_id=project_id,
                        query_text=query,
                        vector_repo=vector_repo,
                        top_k=top_k,
                    )

                started = perf_counter()
                hits = run_hybrid_search(
                    project_id=project_id,
                    query_text=query,
                    vector_repo=vector_repo,
                    top_k=top_k,
                )
                latency_ms = round((perf_counter() - started) * 1000.0, 2)
                latencies.append(latency_ms)

                built = build_context_and_citations(hits, max_items=context_k)
                retrieved_files = [
                    _normalize_path(str(c.get("file_path", "")))
                    for c in built.citations
                    if str(c.get("file_path", "")).strip()
                ]

                matched_expected = 0
                for exp in expected_files:
                    if any(_file_match(exp, got) for got in retrieved_files):
                        matched_expected += 1

                file_hit_at_k = 1 if matched_expected > 0 else 0
                denom = max(1, len(expected_files))
                file_recall_at_k = round(matched_expected / denom, 4)
                symbol_hit_at_k = _symbol_match(expected_symbols, built.contexts)
                graph_or_hybrid = sum(1 for c in built.citations if str(c.get("retrieval_source", "")) in {"graph", "hybrid"})
                total_citations = max(1, len(built.citations))
                graph_or_hybrid_ratio = round(graph_or_hybrid / total_citations, 4)

                total_file_hits += file_hit_at_k
                total_symbol_hits += symbol_hit_at_k
                total_graph_ratio += graph_or_hybrid_ratio

                result = QueryEvalResult(
                    eval_id=eval_id,
                    query=query,
                    difficulty=difficulty,
                    top_k=top_k,
                    latency_ms=latency_ms,
                    retrieved_files=retrieved_files,
                    expected_files=expected_files,
                    expected_symbols=expected_symbols,
                    file_hit_at_k=file_hit_at_k,
                    file_recall_at_k=file_recall_at_k,
                    symbol_hit_at_k=symbol_hit_at_k,
                    graph_or_hybrid_ratio=graph_or_hybrid_ratio,
                )
                results.append(result)

                self.stdout.write(
                    f"[{eval_id}] hit@{top_k}={file_hit_at_k} recall@{top_k}={file_recall_at_k} "
                    f"symbol_hit={symbol_hit_at_k} graph_ratio={graph_or_hybrid_ratio} latency_ms={latency_ms}"
                )
        finally:
            vector_repo.close()

        if not results:
            raise CommandError("No evaluation results produced.")

        n = len(results)
        aggregate = {
            "query_count": n,
            "file_hit_rate_at_k": round(total_file_hits / n, 4),
            "mean_file_recall_at_k": round(sum(r.file_recall_at_k for r in results) / n, 4),
            "symbol_hit_rate_at_k": round(total_symbol_hits / n, 4),
            "mean_graph_or_hybrid_ratio": round(total_graph_ratio / n, 4),
            "latency_ms": {
                "p50": round(median(latencies), 2),
                "p95": round(sorted(latencies)[max(0, min(n - 1, int(0.95 * n) - 1))], 2),
                "mean": round(mean(latencies), 2),
                "max": round(max(latencies), 2),
            },
        }

        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if output_path_raw:
            output_path = Path(output_path_raw)
        else:
            output_dir = Path(settings.BASE_DIR) / "logs" / "eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"rag_eval_{project_id}_{now}.json"

        payload = {
            "project_id": project_id,
            "eval_file": str(eval_file),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aggregate": aggregate,
            "results": [asdict(r) for r in results],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Evaluation complete"))
        self.stdout.write(self.style.SUCCESS(f"Output: {output_path}"))
        self.stdout.write(self.style.SUCCESS(f"file_hit_rate@{top_k}: {aggregate['file_hit_rate_at_k']}"))
        self.stdout.write(self.style.SUCCESS(f"mean_file_recall@{top_k}: {aggregate['mean_file_recall_at_k']}"))
        self.stdout.write(self.style.SUCCESS(f"symbol_hit_rate@{top_k}: {aggregate['symbol_hit_rate_at_k']}"))
        self.stdout.write(self.style.SUCCESS(f"mean_graph_or_hybrid_ratio: {aggregate['mean_graph_or_hybrid_ratio']}"))
        self.stdout.write(self.style.SUCCESS(f"latency p50/p95 ms: {aggregate['latency_ms']['p50']}/{aggregate['latency_ms']['p95']}"))

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CommandError(f"File not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CommandError(f"Expected JSON object in {path}")
    return data


def _agg_value(payload: dict[str, Any], key: str) -> float:
    aggregate = payload.get("aggregate") or {}
    raw = aggregate.get(key, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _latency_value(payload: dict[str, Any], key: str) -> float:
    aggregate = payload.get("aggregate") or {}
    latency = aggregate.get("latency_ms") or {}
    raw = latency.get(key, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _results_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("results") or []
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        eval_id = str(row.get("eval_id", "")).strip()
        if eval_id:
            mapped[eval_id] = row
    return mapped


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


class Command(BaseCommand):
    help = "Compare two rag_eval JSON reports and print metric deltas."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument("--baseline", required=True, help="Path to baseline eval JSON")
        parser.add_argument("--candidate", required=True, help="Path to candidate eval JSON")
        parser.add_argument(
            "--topn",
            type=int,
            default=5,
            help="Number of top improvements/regressions to print (default: 5)",
        )

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        baseline_path = Path(options["baseline"])
        candidate_path = Path(options["candidate"])
        topn = max(1, int(options.get("topn", 5)))

        baseline = _load_json(baseline_path)
        candidate = _load_json(candidate_path)

        b_project = str(baseline.get("project_id", "")).strip()
        c_project = str(candidate.get("project_id", "")).strip()
        if b_project and c_project and b_project != c_project:
            self.stdout.write(self.style.WARNING(f"Project mismatch: baseline={b_project} candidate={c_project}"))

        def delta(metric_key: str) -> float:
            return _agg_value(candidate, metric_key) - _agg_value(baseline, metric_key)

        self.stdout.write(self.style.MIGRATE_HEADING("Aggregate Delta (candidate - baseline)"))
        self.stdout.write(f"file_hit_rate_at_k: {delta('file_hit_rate_at_k'):+.4f}")
        self.stdout.write(f"mean_file_recall_at_k: {delta('mean_file_recall_at_k'):+.4f}")
        self.stdout.write(f"symbol_hit_rate_at_k: {delta('symbol_hit_rate_at_k'):+.4f}")
        self.stdout.write(f"mean_graph_or_hybrid_ratio: {delta('mean_graph_or_hybrid_ratio'):+.4f}")
        self.stdout.write(f"latency_p50_ms: {_latency_value(candidate, 'p50') - _latency_value(baseline, 'p50'):+.2f}")
        self.stdout.write(f"latency_p95_ms: {_latency_value(candidate, 'p95') - _latency_value(baseline, 'p95'):+.2f}")
        self.stdout.write(f"latency_mean_ms: {_latency_value(candidate, 'mean') - _latency_value(baseline, 'mean'):+.2f}")
        self.stdout.write(f"latency_max_ms: {_latency_value(candidate, 'max') - _latency_value(baseline, 'max'):+.2f}")
        self.stdout.write("")

        b_rows = _results_by_id(baseline)
        c_rows = _results_by_id(candidate)
        common_ids = sorted(set(b_rows.keys()) & set(c_rows.keys()))
        if not common_ids:
            self.stdout.write(self.style.WARNING("No overlapping eval_id rows found to compare."))
            return

        per_query_changes: list[dict[str, Any]] = []
        for eval_id in common_ids:
            b = b_rows[eval_id]
            c = c_rows[eval_id]
            recall_delta = _float(c, "file_recall_at_k") - _float(b, "file_recall_at_k")
            hit_delta = _float(c, "file_hit_at_k") - _float(b, "file_hit_at_k")
            symbol_delta = _float(c, "symbol_hit_at_k") - _float(b, "symbol_hit_at_k")
            latency_delta = _float(c, "latency_ms") - _float(b, "latency_ms")
            score = (2.0 * recall_delta) + hit_delta + (0.5 * symbol_delta) - (0.001 * latency_delta)
            per_query_changes.append(
                {
                    "eval_id": eval_id,
                    "query": str(c.get("query", "")),
                    "recall_delta": recall_delta,
                    "hit_delta": hit_delta,
                    "symbol_delta": symbol_delta,
                    "latency_delta_ms": latency_delta,
                    "score": score,
                }
            )

        improves = sorted(per_query_changes, key=lambda x: x["score"], reverse=True)[:topn]
        regresses = sorted(per_query_changes, key=lambda x: x["score"])[:topn]

        self.stdout.write(self.style.SUCCESS(f"Top {topn} Improvements"))
        for row in improves:
            self.stdout.write(
                f"- {row['eval_id']}: recall {row['recall_delta']:+.3f}, "
                f"hit {row['hit_delta']:+.0f}, symbol {row['symbol_delta']:+.0f}, "
                f"latency {row['latency_delta_ms']:+.2f}ms"
            )

        self.stdout.write("")
        self.stdout.write(self.style.WARNING(f"Top {topn} Regressions"))
        for row in regresses:
            self.stdout.write(
                f"- {row['eval_id']}: recall {row['recall_delta']:+.3f}, "
                f"hit {row['hit_delta']:+.0f}, symbol {row['symbol_delta']:+.0f}, "
                f"latency {row['latency_delta_ms']:+.2f}ms"
            )

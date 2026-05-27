from __future__ import annotations

import json
import logging
import random
from time import perf_counter

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from apps.rag.models import IndexingJob
from apps.rag.repositories.graph_repository import GraphRepository
from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.repositories.vector_config import get_vector_collection_name
from apps.rag.services.indexing import run_indexing

logger = logging.getLogger(__name__)


def _log_event(event: str, **payload) -> None:
    logger.info(json.dumps({"event": event, **payload}, default=str))


def _metric_incr(key: str, by: int = 1) -> None:
    # Cache-based lightweight counters for quick observability.
    if cache.get(key) is None:
        cache.set(key, 0, timeout=None)
    try:
        cache.incr(key, by)
    except ValueError:
        current = int(cache.get(key, 0) or 0)
        cache.set(key, current + by, timeout=None)


def _retry_countdown(retry_index: int) -> int:
    base = int(getattr(settings, "RAG_INDEXING_RETRY_BACKOFF_SECONDS", 30))
    max_wait = int(getattr(settings, "RAG_INDEXING_RETRY_BACKOFF_MAX_SECONDS", 600))
    jitter = int(getattr(settings, "RAG_INDEXING_RETRY_JITTER_SECONDS", 10))
    wait = min(max_wait, base * (2 ** retry_index))
    if jitter > 0:
        wait += random.randint(0, jitter)
    return wait


def _should_retry(exc: Exception) -> bool:
    # Conservative retry set for infra/transient faults.
    retryable_types = (ConnectionError, TimeoutError, OSError)
    return isinstance(exc, retryable_types)


@shared_task(
    bind=True,
    max_retries=getattr(settings, "RAG_INDEXING_MAX_RETRIES", 3),
    soft_time_limit=getattr(settings, "RAG_INDEXING_TASK_SOFT_TIME_LIMIT", 3000),
    time_limit=getattr(settings, "RAG_INDEXING_TASK_TIME_LIMIT", 3600),
)
def run_indexing_job(self, job_id: int) -> None:
    started_perf = perf_counter()
    job = IndexingJob.objects.get(id=job_id)

    _metric_incr("rag:metrics:indexing:started")
    _log_event(
        "indexing.started",
        job_id=job.id,
        project_id=job.project_id,
        task_id=self.request.id,
        retry=self.request.retries,
    )

    job.status = "RUNNING"
    job.started_at = timezone.now()
    job.error_message = ""
    metadata = dict(job.metadata or {})
    metadata["attempt"] = int(self.request.retries) + 1
    metadata["task_id"] = self.request.id
    job.metadata = metadata
    job.save(update_fields=["status", "started_at", "error_message", "metadata"])

    vector_repo: QdrantRepository | None = None
    try:
        vector_repo = QdrantRepository(
            collection_name=get_vector_collection_name(),
            url=getattr(settings, "QDRANT_URL", "http://127.0.0.1:6333"),
            api_key=getattr(settings, "QDRANT_API_KEY", None),
            timeout=getattr(settings, "QDRANT_TIMEOUT", 30),
        )
        graph_repo = GraphRepository()

        result = run_indexing(
            project_id=job.project_id,
            source_dir=job.source_dir,
            vector_repo=vector_repo,
            graph_repo=graph_repo,
            indexing_job_id=job.id,
        )

        stats = result.stats
        job.status = "DONE"
        job.finished_at = timezone.now()
        job.documents_collected = stats.documents_collected
        job.chunks_created = stats.chunks_created
        job.vectors_upserted = stats.vectors_upserted
        job.graph_nodes = stats.graph_nodes
        job.graph_edges = stats.graph_edges
        job.duration_seconds = round(perf_counter() - started_perf, 4)

        metadata = dict(job.metadata or {})
        metadata["completed_attempt"] = int(self.request.retries) + 1
        metadata["metrics"] = {
            "documents_collected": stats.documents_collected,
            "chunks_created": stats.chunks_created,
            "vectors_upserted": stats.vectors_upserted,
            "graph_nodes": stats.graph_nodes,
            "graph_edges": stats.graph_edges,
            "duration_seconds": job.duration_seconds,
        }
        job.metadata = metadata
        job.save()

        cache.delete(f"rag:graph:elements:{job.project_id}")
        _metric_incr("rag:metrics:indexing:success")
        _log_event(
            "indexing.succeeded",
            job_id=job.id,
            project_id=job.project_id,
            duration_seconds=job.duration_seconds,
            stats=metadata["metrics"],
        )
        return

    except SoftTimeLimitExceeded as exc:
        _metric_incr("rag:metrics:indexing:soft_timeout")
        retry_on_soft = bool(getattr(settings, "RAG_INDEXING_RETRY_ON_SOFT_TIMEOUT", False))
        if retry_on_soft and self.request.retries < self.max_retries:
            countdown = _retry_countdown(self.request.retries)
            _log_event(
                "indexing.retry.soft_timeout",
                job_id=job.id,
                project_id=job.project_id,
                retry=self.request.retries + 1,
                countdown_seconds=countdown,
            )
            raise self.retry(exc=exc, countdown=countdown)

        job.status = "FAILED"
        job.finished_at = timezone.now()
        job.duration_seconds = round(perf_counter() - started_perf, 4)
        job.error_message = "Soft time limit exceeded during indexing."
        job.save(update_fields=["status", "finished_at", "duration_seconds", "error_message"])
        _log_event(
            "indexing.failed.soft_timeout",
            job_id=job.id,
            project_id=job.project_id,
            duration_seconds=job.duration_seconds,
        )
        raise

    except Exception as exc:
        _metric_incr("rag:metrics:indexing:failed")
        if _should_retry(exc) and self.request.retries < self.max_retries:
            countdown = _retry_countdown(self.request.retries)
            _metric_incr("rag:metrics:indexing:retried")
            _log_event(
                "indexing.retry",
                job_id=job.id,
                project_id=job.project_id,
                retry=self.request.retries + 1,
                countdown_seconds=countdown,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            raise self.retry(exc=exc, countdown=countdown)

        job.status = "FAILED"
        job.finished_at = timezone.now()
        job.duration_seconds = round(perf_counter() - started_perf, 4)
        job.error_message = str(exc)[:4000]
        job.save(update_fields=["status", "finished_at", "duration_seconds", "error_message"])
        _log_event(
            "indexing.failed",
            job_id=job.id,
            project_id=job.project_id,
            duration_seconds=job.duration_seconds,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        raise

    finally:
        if vector_repo is not None:
            vector_repo.close()
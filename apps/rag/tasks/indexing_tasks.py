from celery import shared_task
from django.conf import settings
from django.utils import timezone

from apps.rag.models import IndexingJob
from apps.rag.repositories.graph_repository import GraphRepository
from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.services.indexing import run_indexing

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def run_indexing_job(self, job_id: int) -> None:
    job = IndexingJob.objects.get(id=job_id)
    job.status = "RUNNING"
    job.started_at = timezone.now()
    job.error_message = ""
    job.save(update_fields=["status", "started_at", "error_message"])

    vector_repo: QdrantRepository | None = None
    try:
        vector_repo = QdrantRepository(
            collection_name=getattr(settings, "RAG_VECTOR_COLLECTION", "rag_chunks"),
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
        job.save()
    except Exception as exc:
        job.status = "FAILED"
        job.finished_at = timezone.now()
        job.error_message = str(exc)[:4000]
        job.save(update_fields=["status", "finished_at", "error_message"])
        raise
    finally:
        if vector_repo is not None:
            vector_repo.close()

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from apps.rag.forms import RepositoryInputForm
from apps.rag.models import CodeEdge, CodeNode, IndexingJob
from apps.rag.repositories.qdrant_repository import QdrantRepository

try:
    from apps.rag.tasks.indexing_tasks import run_indexing_job
except Exception:  # Celery task may not be wired yet
    run_indexing_job = None


def _uploads_root() -> Path:
    root = Path(settings.BASE_DIR) / "var" / "uploads" / "repositories"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _materialize_uploaded_source(form: RepositoryInputForm, request: HttpRequest, project_id: str) -> str:
    source_type = form.cleaned_data["source_type"]
    project_root = _uploads_root() / project_id

    if project_root.exists():
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    if source_type == "zip":
        zip_input = form.cleaned_data["zip_file"]
        zip_path = project_root / "source.zip"
        with zip_path.open("wb") as destination:
            for chunk in zip_input.chunks():
                destination.write(chunk)

        extracted_root = project_root / "extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_root)
        return str(extracted_root)

    folder_root = project_root / "folder"
    folder_root.mkdir(parents=True, exist_ok=True)

    for uploaded in request.FILES.getlist("folder_files"):
        rel_name = (uploaded.name or "").replace("\\", "/").lstrip("/")
        target = folder_root / rel_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as destination:
            for chunk in uploaded.chunks():
                destination.write(chunk)
    return str(folder_root)


def _latest_repository_rows() -> list[dict]:
    """
    Build dashboard rows from latest IndexingJob per project_id.
    """
    latest_jobs: list[IndexingJob] = []
    seen: set[str] = set()

    for job in IndexingJob.objects.all().order_by("-queued_at"):
        if job.project_id in seen:
            continue
        seen.add(job.project_id)
        latest_jobs.append(job)

    rows: list[dict] = []
    for job in latest_jobs:
        source_type = (job.metadata or {}).get("source_type", "folder")
        name = (job.metadata or {}).get("name", job.project_id)
        rows.append(
            {
                "id": job.id,
                "project_id": job.project_id,
                "name": name,
                "source_type": source_type,
                "status": _ui_status_from_job_status(job.status),
                "updated_at": job.queued_at,
                "source_dir": job.source_dir,
            }
        )
    return rows


def _ui_status_from_job_status(job_status: str) -> str:
    mapping = {
        IndexingJob.Status.PENDING: "not_processed",
        IndexingJob.Status.RUNNING: "processing",
        IndexingJob.Status.DONE: "ready",
        IndexingJob.Status.FAILED: "failed",
    }
    return mapping.get(job_status, "not_processed")


@require_GET
def dashboard(request: HttpRequest) -> HttpResponse:
    context = {
        "repositories": _latest_repository_rows(),
        "repo_form": RepositoryInputForm(),
    }
    return render(request, "rag/pages/dashboard.jinja", context)


@require_GET
def repository_table(request: HttpRequest) -> HttpResponse:
    context = {"repositories": _latest_repository_rows()}
    return render(request, "rag/components/repository_table.jinja", context)


@require_POST
def add_repository(request: HttpRequest) -> HttpResponse:
    form = RepositoryInputForm(request.POST, request.FILES)
    if not form.is_valid():
        response = render(
            request,
            "rag/components/add_repository_modal.jinja",
            {"repo_form": form},
            status=400,
        )
        return response

    name = form.cleaned_data["name"].strip()
    source_type = form.cleaned_data["source_type"]
    project_id = f"repo-{uuid.uuid4().hex[:10]}"
    source_dir = _materialize_uploaded_source(form, request, project_id)

    zip_input = form.cleaned_data.get("zip_file")

    IndexingJob.objects.create(
        project_id=project_id,
        source_dir=source_dir,
        status=IndexingJob.Status.PENDING,
        metadata={
            "name": name or project_id,
            "source_type": source_type,
            "uploaded_filename": zip_input.name if zip_input else "",
        },
    )

    response = render(
        request,
        "rag/components/add_repository_modal.jinja",
        {"repo_form": RepositoryInputForm()},
    )
    response["HX-Trigger"] = '{"repoListChanged": true, "repoAdded": true}'
    return response


@require_POST
def process_repository(request: HttpRequest, repo_id: int) -> HttpResponse:
    """
    Reuses the same indexing job row and triggers Celery if available.
    """
    job = get_object_or_404(IndexingJob, id=repo_id)

    # Reset job fields for reprocessing in place.
    job.status = IndexingJob.Status.PENDING
    job.task_id = ""
    job.started_at = None
    job.finished_at = None
    job.documents_collected = 0
    job.chunks_created = 0
    job.vectors_upserted = 0
    job.graph_nodes = 0
    job.graph_edges = 0
    job.duration_seconds = 0.0
    job.error_message = ""

    if run_indexing_job is not None:
        task = run_indexing_job.delay(job.id)
        job.task_id = task.id
        job.status = IndexingJob.Status.RUNNING
    job.save()

    # Rebuild one row payload for HTMX row replacement
    row = {
        "id": job.id,
        "project_id": job.project_id,
        "name": (job.metadata or {}).get("name", job.project_id),
        "source_type": (job.metadata or {}).get("source_type", "folder"),
        "status": _ui_status_from_job_status(job.status),
        "updated_at": job.queued_at,
        "source_dir": job.source_dir,
    }
    return render(request, "rag/components/repository_row.jinja", {"repo": row})


@require_http_methods(["DELETE"])
def delete_repository(request: HttpRequest, repo_id: int) -> HttpResponse:
    """
    Deletes project data across jobs, graph tables, vector DB, and uploaded source files.
    """
    job = get_object_or_404(IndexingJob, id=repo_id)
    project_id = job.project_id

    # 1) Delete vectors in Qdrant for this project.
    vector_repo = QdrantRepository(
        collection_name=getattr(settings, "RAG_VECTOR_COLLECTION", "rag_chunks"),
        url=getattr(settings, "QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=getattr(settings, "QDRANT_API_KEY", None),
        timeout=getattr(settings, "QDRANT_TIMEOUT", 30),
    )
    try:
        vector_repo.connect()
        vector_repo.delete_project(project_id=project_id)
    finally:
        vector_repo.close()

    # 2) Delete persisted graph rows.
    CodeEdge.objects.filter(project_id=project_id).delete()
    CodeNode.objects.filter(project_id=project_id).delete()

    # 3) Delete uploaded source directory.
    project_upload_dir = _uploads_root() / project_id
    if project_upload_dir.exists():
        shutil.rmtree(project_upload_dir, ignore_errors=True)

    # 4) Delete indexing jobs for this project.
    IndexingJob.objects.filter(project_id=project_id).delete()

    response = HttpResponse(status=204)
    response["HX-Trigger"] = '{"repoListChanged": true, "repoDeleted": true}'
    return response


@require_GET
def query_page(request: HttpRequest) -> HttpResponse:
    repo_id = request.GET.get("repo_id")
    selected = None
    if repo_id:
        selected = IndexingJob.objects.filter(id=repo_id).first()

    context = {
        "selected_repo_id": int(repo_id) if repo_id and repo_id.isdigit() else None,
        "selected_project_id": selected.project_id if selected else "",
    }
    return render(request, "rag/pages/query.jinja", context)


@require_GET
def repository_status_row(request: HttpRequest, repo_id: int) -> HttpResponse:
    """
    Poll endpoint for HTMX to refresh one row status.
    """
    job = get_object_or_404(IndexingJob, id=repo_id)
    row = {
        "id": job.id,
        "project_id": job.project_id,
        "name": (job.metadata or {}).get("name", job.project_id),
        "source_type": (job.metadata or {}).get("source_type", "folder"),
        "status": _ui_status_from_job_status(job.status),
        "updated_at": job.queued_at,
        "source_dir": job.source_dir,
    }
    return render(request, "rag/components/repository_row.jinja", {"repo": row})

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
from apps.rag.models import IndexingJob

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
                "status": job.status.lower(),  # matches template badges
                "updated_at": job.queued_at,
                "source_dir": job.source_dir,
            }
        )
    return rows


@require_GET
def dashboard(request: HttpRequest) -> HttpResponse:
    context = {
        "repositories": _latest_repository_rows(),
        "repo_form": RepositoryInputForm(),
    }
    return render(request, "rag/pages/dashboard.jinja", context)


@require_POST
def add_repository(request: HttpRequest) -> HttpResponse:
    form = RepositoryInputForm(request.POST, request.FILES)
    if not form.is_valid():
        context = {
            "repositories": _latest_repository_rows(),
            "repo_form": form,
        }
        return render(request, "rag/pages/dashboard.jinja", context, status=400)

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

    context = {
        "repositories": _latest_repository_rows(),
        "repo_form": RepositoryInputForm(),
    }
    return render(request, "rag/pages/dashboard.jinja", context)


@require_POST
def process_repository(request: HttpRequest, repo_id: int) -> HttpResponse:
    """
    Creates a new indexing job for the same project/source and triggers Celery if available.
    """
    latest_for_repo = get_object_or_404(IndexingJob, id=repo_id)

    new_job = IndexingJob.objects.create(
        project_id=latest_for_repo.project_id,
        source_dir=latest_for_repo.source_dir,
        status=IndexingJob.Status.PENDING,
        metadata=latest_for_repo.metadata or {},
    )

    if run_indexing_job is not None:
        task = run_indexing_job.delay(new_job.id)
        new_job.task_id = task.id
        new_job.status = IndexingJob.Status.RUNNING
        new_job.save(update_fields=["task_id", "status"])

    # Rebuild one row payload for HTMX row replacement
    row = {
        "id": new_job.id,
        "project_id": new_job.project_id,
        "name": (new_job.metadata or {}).get("name", new_job.project_id),
        "source_type": (new_job.metadata or {}).get("source_type", "folder"),
        "status": new_job.status.lower(),
        "updated_at": new_job.queued_at,
        "source_dir": new_job.source_dir,
    }
    return render(request, "rag/components/repository_row.jinja", {"repo": row})


@require_http_methods(["DELETE"])
def delete_repository(request: HttpRequest, repo_id: int) -> HttpResponse:
    """
    Deletes all jobs for the selected project_id so the dashboard row disappears.
    """
    job = get_object_or_404(IndexingJob, id=repo_id)
    IndexingJob.objects.filter(project_id=job.project_id).delete()
    return HttpResponse(status=204)


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
        "status": job.status.lower(),
        "updated_at": job.queued_at,
        "source_dir": job.source_dir,
    }
    return render(request, "rag/components/repository_row.jinja", {"repo": row})

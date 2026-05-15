from __future__ import annotations

import shutil
import uuid
import zipfile
import json
from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from apps.rag.forms import RepositoryInputForm
from apps.rag.models import CodeEdge, CodeNode, Conversation, ConversationMessage, IndexingJob
from apps.rag.repositories.graph_repository import GraphRepository
from apps.rag.repositories.qdrant_repository import QdrantRepository
from apps.rag.repositories.vector_config import get_vector_collection_name
from apps.rag.services.answering import synthesize_answer
from apps.rag.services.context_builder import build_context_and_citations
from apps.rag.services.hybrid_search import run_hybrid_search

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


def _graph_cache_key(project_id: str) -> str:
    return f"rag:graph:elements:{project_id}"


def _build_graph_elements_for_project(project_id: str) -> list[dict]:
    graph = GraphRepository().load_graph(project_id=project_id)
    elements: list[dict] = []

    for node_id, attrs in graph.nodes(data=True):
        elements.append(
            {
                "data": {
                    "id": str(node_id),
                    "label": str(node_id).split("/")[-1],
                    "node_type": attrs.get("node_type", "file"),
                    "language": attrs.get("language", ""),
                    "file_path": attrs.get("file_path", str(node_id)),
                }
            }
        )

    for source, target, attrs in graph.edges(data=True):
        elements.append(
            {
                "data": {
                    "id": f"{source}->{target}",
                    "source": str(source),
                    "target": str(target),
                    "relation": attrs.get("relation", "imports"),
                }
            }
        )

    return elements


def _get_cached_graph_elements(project_id: str) -> list[dict]:
    key = _graph_cache_key(project_id)
    cached = cache.get(key)
    if cached is not None:
        return cached
    elements = _build_graph_elements_for_project(project_id)
    # 15 minutes cache; can be invalidated on indexing completion in task flow later.
    cache.set(key, elements, timeout=15 * 60)
    return elements


def _vector_repo() -> QdrantRepository:
    return QdrantRepository(
        collection_name=get_vector_collection_name(),
        url=getattr(settings, "QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=getattr(settings, "QDRANT_API_KEY", None),
        timeout=getattr(settings, "QDRANT_TIMEOUT", 30),
    )


def _list_conversations(project_id: str) -> list[Conversation]:
    if not project_id:
        return []
    return list(
        Conversation.objects.filter(project_id=project_id, is_archived=False).order_by("-updated_at")
    )


def _qa_history_from_conversation(conversation: Conversation | None) -> list[dict]:
    if conversation is None:
        return []
    history: list[dict] = []
    pending_question: str | None = None
    for msg in conversation.messages.all().order_by("created_at"):
        if msg.role == ConversationMessage.Role.USER:
            pending_question = msg.content
            continue
        if msg.role == ConversationMessage.Role.ASSISTANT:
            history.append(
                {
                    "question": pending_question or "",
                    "answer": msg.content,
                }
            )
            pending_question = None
    return history


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
        collection_name=get_vector_collection_name(),
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

    # 4) Delete conversations for this project.
    Conversation.objects.filter(project_id=project_id).delete()

    # 5) Delete indexing jobs for this project.
    IndexingJob.objects.filter(project_id=project_id).delete()

    response = HttpResponse(status=204)
    response["HX-Trigger"] = '{"repoListChanged": true, "repoDeleted": true}'
    return response


@require_GET
def query_home(request: HttpRequest) -> HttpResponse:
    repo_id = request.GET.get("repo_id")
    selected = None
    if repo_id:
        selected = IndexingJob.objects.filter(id=repo_id).first()

    selected_project_id = selected.project_id if selected else ""
    conversation_rows = _list_conversations(selected_project_id)

    context = {
        "selected_repo_id": int(repo_id) if repo_id and repo_id.isdigit() else None,
        "selected_project_id": selected_project_id,
        "repositories": _latest_repository_rows(),
        "conversation_rows": conversation_rows,
    }
    return render(request, "rag/pages/query_home.jinja", context)


@require_GET
def query_start(request: HttpRequest) -> HttpResponse:
    repo_id = request.GET.get("repo_id")
    selected = None
    if repo_id and repo_id.isdigit():
        selected = IndexingJob.objects.filter(id=int(repo_id)).first()
    if selected is None:
        return HttpResponse("Valid repo_id is required", status=400)

    conversation = Conversation.objects.create(
        project_id=selected.project_id,
        title="New Conversation",
    )
    return redirect(f"{request.path.replace('/start/', '/chat/')}?repo_id={selected.id}&conversation_id={conversation.id}")


@require_GET
def query_page(request: HttpRequest) -> HttpResponse:
    repo_id = request.GET.get("repo_id")
    conversation_id = (request.GET.get("conversation_id") or "").strip()
    selected = None
    if repo_id:
        selected = IndexingJob.objects.filter(id=repo_id).first()
    if selected is None:
        return HttpResponse("Valid repo_id is required", status=400)
    if not conversation_id.isdigit():
        return HttpResponse("conversation_id is required", status=400)

    selected_project_id = selected.project_id
    selected_repo_id = int(repo_id) if repo_id and repo_id.isdigit() else None
    selected_conversation = Conversation.objects.filter(
        id=int(conversation_id),
        project_id=selected_project_id,
        is_archived=False,
    ).first()
    if selected_conversation is None:
        return HttpResponse("Conversation not found for this project", status=404)

    graph_elements = _get_cached_graph_elements(selected_project_id) if selected_project_id else []
    graph_node_count = sum(1 for e in graph_elements if "source" not in (e.get("data") or {}))
    graph_edge_count = sum(1 for e in graph_elements if "source" in (e.get("data") or {}))
    qa_history = _qa_history_from_conversation(selected_conversation)
    last_assistant = selected_conversation.messages.filter(
        role=ConversationMessage.Role.ASSISTANT
    ).order_by("-created_at").first()
    citations = list(last_assistant.citations_json) if last_assistant else []

    context = {
        "selected_repo_id": selected_repo_id,
        "selected_project_id": selected_project_id,
        "graph_elements_json": json.dumps(graph_elements),
        "graph_node_count": graph_node_count,
        "graph_edge_count": graph_edge_count,
        "citations": citations,
        "qa_history": qa_history,
        "selected_conversation_id": selected_conversation.id,
    }
    return render(request, "rag/pages/query.jinja", context)


@require_POST
def query_run(request: HttpRequest) -> HttpResponse:
    repo_id = (request.POST.get("repo_id") or "").strip()
    project_id = (request.POST.get("project_id") or "").strip()
    query_text = (request.POST.get("query_text") or "").strip()
    conversation_id = (request.POST.get("conversation_id") or "").strip()
    if not project_id or not query_text:
        return HttpResponse("project_id and query_text are required", status=400)

    conversation: Conversation | None = None
    if conversation_id.isdigit():
        conversation = Conversation.objects.filter(
            id=int(conversation_id),
            project_id=project_id,
            is_archived=False,
        ).first()
    if conversation is None:
        conversation = Conversation.objects.create(
            project_id=project_id,
            title=query_text[:80],
        )

    vector_repo = _vector_repo()
    try:
        vector_repo.connect()
        hits = run_hybrid_search(
            project_id=project_id,
            query_text=query_text,
            vector_repo=vector_repo,
            top_k=8,
        )
    finally:
        vector_repo.close()

    built = build_context_and_citations(hits)
    try:
        answer_text = synthesize_answer(
            query_text=query_text,
            contexts=built.contexts,
            citations=built.citations,
        )
    except Exception as exc:
        answer_text = (
            "Answer generation failed. Showing retrieval-only fallback.\n\n"
            f"Reason: {str(exc)}"
        )

    ConversationMessage.objects.create(
        conversation=conversation,
        role=ConversationMessage.Role.USER,
        content=query_text,
    )
    ConversationMessage.objects.create(
        conversation=conversation,
        role=ConversationMessage.Role.ASSISTANT,
        content=answer_text,
        citations_json=built.citations,
        trace_json={"hit_count": len(hits)},
    )
    if not conversation.title:
        conversation.title = query_text[:80]
    conversation.save(update_fields=["title", "updated_at"])

    graph_elements = _get_cached_graph_elements(project_id)
    graph_node_count = sum(1 for e in graph_elements if "source" not in (e.get("data") or {}))
    graph_edge_count = sum(1 for e in graph_elements if "source" in (e.get("data") or {}))

    conversation_rows = _list_conversations(project_id)
    qa_history = _qa_history_from_conversation(conversation)
    context = {
        "selected_project_id": project_id,
        "selected_repo_id": int(repo_id) if repo_id.isdigit() else None,
        "conversation_rows": conversation_rows,
        "selected_conversation_id": conversation.id,
        "graph_elements_json": json.dumps(graph_elements),
        "graph_node_count": graph_node_count,
        "graph_edge_count": graph_edge_count,
        "citations": built.citations,
        "qa_history": qa_history,
    }
    return render(request, "rag/components/query_workspace.jinja", context)


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

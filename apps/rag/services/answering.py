from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from django.conf import settings


@dataclass(frozen=True, slots=True)
class AnswerOutput:
    contract_version: str
    backend: str
    status: str  # ok | fallback | error
    answer_text: str
    key_points: list[str] = field(default_factory=list)
    citations_doc_numbers: list[int] = field(default_factory=list)
    error_message: str | None = None
    raw_output: str = ""

    def to_trace(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "backend": self.backend,
            "status": self.status,
            "key_points": self.key_points,
            "citations_doc_numbers": self.citations_doc_numbers,
            "error_message": self.error_message,
        }


def _extract_doc_numbers(text: str) -> list[int]:
    found = re.findall(r"\[DOC\s+(\d+)\]", text or "")
    numbers: list[int] = []
    for item in found:
        value = int(item)
        if value not in numbers:
            numbers.append(value)
    return numbers


def _fallback_answer(query_text: str, contexts: list[str], citations: list[dict]) -> AnswerOutput:
    if not contexts:
        return AnswerOutput(
            contract_version="v1",
            backend="fallback",
            status="fallback",
            answer_text=(
                "I could not find enough relevant context in the indexed codebase for this question. "
                "Try rephrasing the question or re-indexing the repository."
            ),
            key_points=[
                "No relevant chunks were retrieved.",
                "Try a more specific query with file/module/function names.",
            ],
            citations_doc_numbers=[],
        )

    first = contexts[0].strip()
    preview = first[:420] + ("..." if len(first) > 420 else "")
    refs = ", ".join(
        f"[{idx + 1}] {c.get('file_path', '')}:{c.get('start_line', 0)}-{c.get('end_line', 0)}"
        for idx, c in enumerate(citations[:3])
    )
    answer_text = (
        f"Question: {query_text}\n\n"
        "Top retrieved context preview:\n"
        f"{preview}\n\n"
        f"References: {refs if refs else 'N/A'}\n\n"
        "This is fallback synthesis. Set RAG_ANSWER_BACKEND=openai for model-generated answers."
    )
    return AnswerOutput(
        contract_version="v1",
        backend="fallback",
        status="fallback",
        answer_text=answer_text,
        key_points=[
            "Answer is retrieval-only fallback.",
            "Preview is from top-ranked chunk.",
        ],
        citations_doc_numbers=list(range(1, min(3, len(citations)) + 1)),
    )


def _build_context_block(contexts: list[str], citations: list[dict]) -> str:
    blocks: list[str] = []
    for idx, (ctx, cit) in enumerate(zip(contexts, citations), start=1):
        file_path = cit.get("file_path", "")
        start = cit.get("start_line", 0)
        end = cit.get("end_line", 0)
        blocks.append(f"[DOC {idx}] file={file_path} lines={start}-{end}\n{ctx}")
    return "\n\n".join(blocks)


def _openai_answer(query_text: str, contexts: list[str], citations: list[dict]) -> AnswerOutput:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("RAG_ANSWER_BACKEND='openai' requires the 'openai' package.") from exc

    if not contexts:
        return _fallback_answer(query_text=query_text, contexts=contexts, citations=citations)

    client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))
    model = getattr(settings, "RAG_ANSWER_MODEL", "gpt-4.1-mini")
    temperature = float(getattr(settings, "RAG_ANSWER_TEMPERATURE", 0.1))

    context_block = _build_context_block(contexts, citations)
    system_prompt = (
        "You are a codebase assistant. Answer using ONLY the provided snippets. "
        "If insufficient context, say so clearly."
    )
    user_prompt = (
        f"Question:\n{query_text}\n\n"
        f"Context snippets:\n{context_block}\n\n"
        "Return STRICT JSON only:\n"
        "{"
        "\"answer_text\": \"string\", "
        "\"key_points\": [\"string\"], "
        "\"citations_doc_numbers\": [1], "
        "\"insufficient_context\": false"
        "}"
    )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = (response.output_text or "").strip()
    try:
        payload = json.loads(raw_text)
        answer_text = str(payload.get("answer_text", "")).strip()
        key_points = [str(x).strip() for x in payload.get("key_points", []) if str(x).strip()]
        citation_numbers = []
        for x in payload.get("citations_doc_numbers", []):
            try:
                citation_numbers.append(int(x))
            except (TypeError, ValueError):
                continue
        if not answer_text:
            answer_text = "Insufficient context to answer confidently from retrieved snippets."
        return AnswerOutput(
            contract_version="v1",
            backend="openai",
            status="ok",
            answer_text=answer_text,
            key_points=key_points[:8],
            citations_doc_numbers=citation_numbers,
            raw_output=raw_text,
        )
    except Exception:
        return AnswerOutput(
            contract_version="v1",
            backend="openai",
            status="ok",
            answer_text=raw_text or "No model output.",
            key_points=[],
            citations_doc_numbers=_extract_doc_numbers(raw_text),
            raw_output=raw_text,
        )


def synthesize_answer(query_text: str, contexts: list[str], citations: list[dict] | None = None) -> AnswerOutput:
    citations = citations or []
    backend = getattr(settings, "RAG_ANSWER_BACKEND", "fallback").lower()

    if backend == "fallback":
        return _fallback_answer(query_text=query_text, contexts=contexts, citations=citations)
    if backend == "openai":
        return _openai_answer(query_text=query_text, contexts=contexts, citations=citations)

    raise ValueError(f"Unsupported RAG_ANSWER_BACKEND={backend}. Use one of fallback|openai.")

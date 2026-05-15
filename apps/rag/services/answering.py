from __future__ import annotations

from django.conf import settings


def _fallback_answer(query_text: str, contexts: list[str], citations: list[dict]) -> str:
    if not contexts:
        return (
            "I could not find enough relevant context in the indexed codebase for this question. "
            "Try rephrasing the question or re-indexing the repository."
        )

    first = contexts[0].strip()
    preview = first[:420] + ("..." if len(first) > 420 else "")
    refs = ", ".join(
        f"[{idx + 1}] {c.get('file_path', '')}:{c.get('start_line', 0)}-{c.get('end_line', 0)}"
        for idx, c in enumerate(citations[:3])
    )
    return (
        f"Question: {query_text}\n\n"
        "Top retrieved context preview:\n"
        f"{preview}\n\n"
        f"References: {refs if refs else 'N/A'}\n\n"
        "This is fallback synthesis. Set RAG_ANSWER_BACKEND=openai for model-generated answers."
    )


def _build_context_block(contexts: list[str], citations: list[dict]) -> str:
    blocks: list[str] = []
    for idx, (ctx, cit) in enumerate(zip(contexts, citations), start=1):
        file_path = cit.get("file_path", "")
        start = cit.get("start_line", 0)
        end = cit.get("end_line", 0)
        blocks.append(
            f"[DOC {idx}] file={file_path} lines={start}-{end}\n"
            f"{ctx}"
        )
    return "\n\n".join(blocks)


def _openai_answer(query_text: str, contexts: list[str], citations: list[dict]) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("RAG_ANSWER_BACKEND='openai' requires the 'openai' package.") from exc

    client = OpenAI()
    model = getattr(settings, "RAG_ANSWER_MODEL", "gpt-4.1-mini")
    temperature = float(getattr(settings, "RAG_ANSWER_TEMPERATURE", 0.1))

    context_block = _build_context_block(contexts, citations)
    system_prompt = (
        "You are a codebase assistant. Answer using ONLY the provided context snippets. "
        "If the context is insufficient, explicitly say so. "
        "Cite relevant snippets using [DOC n] markers."
    )
    user_prompt = (
        f"Question:\n{query_text}\n\n"
        f"Context snippets:\n{context_block}\n\n"
        "Return:\n"
        "1) concise answer\n"
        "2) bullet list of key points\n"
        "3) citations as [DOC n]"
    )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


def synthesize_answer(query_text: str, contexts: list[str], citations: list[dict] | None = None) -> str:
    citations = citations or []
    backend = getattr(settings, "RAG_ANSWER_BACKEND", "fallback").lower()

    if backend == "fallback":
        return _fallback_answer(query_text=query_text, contexts=contexts, citations=citations)
    if backend == "openai":
        return _openai_answer(query_text=query_text, contexts=contexts, citations=citations)

    raise ValueError(
        f"Unsupported RAG_ANSWER_BACKEND={backend}. Use one of fallback|openai."
    )

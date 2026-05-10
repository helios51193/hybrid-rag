from __future__ import annotations


def synthesize_answer(query_text: str, contexts: list[str]) -> str:
    """
    Placeholder answer synthesis for v1.
    Replace with LLM call later.
    """
    if not contexts:
        return (
            "I could not find enough relevant context in the indexed codebase for this question. "
            "Try rephrasing the question or re-indexing the repository."
        )

    first = contexts[0].strip()
    preview = first[:420] + ("..." if len(first) > 420 else "")
    return (
        f"Question: {query_text}\n\n"
        "Top retrieved context preview:\n"
        f"{preview}\n\n"
        "This answer is currently generated from retrieved snippets without full LLM synthesis."
    )

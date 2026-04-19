"""Utility functions for magic-generator."""


def build_prompt(
    query: str,
    context: list[str],
    system_template: str | None = None,
) -> str:
    """Build a prompt from query and context.

    Args:
        query: User query.
        context: List of context strings.
        system_template: Optional system prompt template.

    Returns:
        Formatted prompt string.
    """
    if not context:
        return query

    context_text = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(context))
    return (
        f"Context information:\n{context_text}\n\n"
        f"---\n\n"
        f"User question: {query}\n\n"
        f"Based on the context information above, please answer the question. "
        f"If the context doesn't contain relevant information, say so."
    )

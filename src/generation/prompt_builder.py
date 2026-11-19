from textwrap import dedent
from langchain_core.documents import Document
from src.generation.context_extractor import extract_context

SYSTEM_INSTRUCTIONS = """
You are K8s Copilot, a Kubernetes documentation assistant.

Your task is to answer the user's question using ONLY the provided
documentation context.

GROUNDING RULES:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. Do not invent Kubernetes APIs, features, commands, behavior, or examples.
4. If the context does not contain enough information to answer the
   question, clearly state that the documentation does not provide
   enough information.
5. Ignore context that is unrelated to the user's question.
6. When multiple context sections are relevant, combine them into one
   coherent answer.
7. Do not claim something unless it is supported by the context.

ANSWER STYLE:
1. Start with a direct answer to the question.
2. Explain only the concepts necessary to answer the question.
3. Use short paragraphs and bullet points when they improve clarity.
4. Preserve important Kubernetes terminology.
5. Do not include tangential information from the context.
6. Avoid unnecessary repetition.
7. Do not mention the RAG system, retrieved context, embeddings,
   BM25, RRF, reranking, or query translation.
8. Do not begin with "Based on the provided context" or similar phrases.
9. Keep the answer concise and technically precise.
10. If the context is insufficient, explicitly say so instead of guessing.
11. Do not list every piece of information in the context; select only
    information directly relevant to the user's question.
""".strip()


def build_prompt(
    query: str,
    chunks: list[Document],
    max_chunks: int = 5
) -> str:
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        raise ValueError("No retrieved documents were provided.")

    # Build context internally
    context = extract_context(
        chunks,
        max_chunks=max_chunks
    )

    prompt = dedent(
        f"""
        {SYSTEM_INSTRUCTIONS}

        ========================================
        USER QUESTION
        ========================================

        {query}

        ========================================
        DOCUMENTATION CONTEXT
        ========================================

        {context}

        ========================================
        ANSWER
        ========================================

        Answer the user's question using only the
        documentation context above.
        """
    ).strip()

    return prompt


if __name__ == "__main__":

    chunks = [
        Document(
            page_content=(
                "The PodGroup scheduling cycle evaluates a group "
                "of Pods as a single unit. If sufficient resources "
                "are not available for the entire group, none of "
                "the Pods are bound."
            ),
            metadata={
                "source": "Scheduling, Preemption and Eviction.pdf"
            }
        ),
        Document(
            page_content=(
                "Gang scheduling ensures that a group of Pods "
                "are scheduled on an all-or-nothing basis."
            ),
            metadata={
                "source": "Scheduling, Preemption and Eviction.pdf"
            }
        )
    ]

    query = (
        "How does Kubernetes prevent a group of Pods from being "
        "partially scheduled when resources are insufficient?"
    )

    prompt = build_prompt(
        query=query,
        chunks=chunks
    )

    print("=" * 70)
    print("GENERATED PROMPT")
    print("=" * 70)
    print(prompt)
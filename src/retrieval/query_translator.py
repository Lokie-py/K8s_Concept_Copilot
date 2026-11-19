import logging
import os
import re

from langchain_groq import ChatGroq


logger = logging.getLogger(__name__)


MODEL_NAME = "llama-3.1-8b-instant"
NUM_QUERIES = 4

_llm = None


def get_query_translator() -> ChatGroq:
    """
    Return a cached Groq LLM instance for query translation.
    """

    global _llm

    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the environment."
            )

        _llm = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0,
        )

    return _llm


def _clean_query(query: str) -> str:
    """
    Clean a single generated query.
    """

    query = query.strip()

    # Remove common numbering:
    # 1. query
    # 2) query
    # - query
    query = re.sub(
        r"^(?:\d+[\.\)]\s*|[-*]\s*)",
        "",
        query,
    )

    # Remove surrounding quotes
    query = query.strip("\"'")

    return query.strip()


def generate_queries(
    query: str,
    num_queries: int = NUM_QUERIES,
) -> list[str]:
    """
    Generate multiple search-oriented variations of the user's query.

    The original query is always preserved as the first query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    llm = get_query_translator()

    prompt = f"""
You are a query expansion component for a Kubernetes documentation
retrieval system.

Generate exactly {num_queries} alternative search queries for the
user's question.

These queries will be used for:
- semantic vector retrieval
- BM25 lexical retrieval
- reciprocal rank fusion
- cross-encoder reranking

The purpose of query expansion is to improve document retrieval
coverage while preserving the exact information need of the user.

QUERY GENERATION STRATEGY:

Generate the following five types of queries internally:

1. A complete semantic reformulation of the user's question.
2. A Kubernetes terminology-focused query using important technical
   terms already implied by the question.
3. A concept-focused query emphasizing the core Kubernetes concepts,
   resources, or components involved.
4. A mechanism/behavior-focused query emphasizing what Kubernetes does,
   how it behaves, or how the relevant mechanism works.
5. A short keyword-focused query containing the most important terms
   for lexical/BM25 retrieval.

IMPORTANT:
These are internal generation categories.

NEVER output the category names.

NEVER output labels.

NEVER output numbering.

NEVER output explanations.

NEVER output the original question as a separate labelled item.

OUTPUT REQUIREMENTS:

- Return exactly {num_queries} lines.
- One query per line.
- Each line must contain only one search query.
- No numbering.
- No bullets.
- No labels.
- No explanations.
- No quotation marks.
- No blank lines.

RETRIEVAL QUALITY REQUIREMENTS:

1. Preserve the user's original information need.
2. Do not answer the question.
3. Do not introduce facts that are not present or implied by the
   user's question.
4. Do not invent Kubernetes APIs, resources, mechanisms, or features.
5. Preserve Kubernetes terminology when it is explicitly present or
   clearly implied by the question.
6. Make the five queries meaningfully different.
7. Do not produce five simple paraphrases.
8. At least one query should contain terminology useful for BM25.
9. At least one query should emphasize the underlying concept.
10. At least one query should emphasize the mechanism or behavior.
11. The final query should be substantially shorter than the original.
12. Avoid overly broad queries.
13. Avoid duplicate or nearly identical queries.

SELF-CHECK BEFORE OUTPUT:

Before returning the queries, verify internally:

- There are exactly {num_queries} queries.
- Every query represents the same information need.
- No query answers the question.
- No query contains a category label.
- No query contains an explanation.
- The queries are not simple duplicates.
- The final query is concise and keyword-oriented.

Return ONLY the queries.

USER QUESTION:
{query}
"""

    response = llm.invoke(prompt)

    raw_text = response.content

    # LangChain/Groq can return either a string or structured content.
    if isinstance(raw_text, list):
        raw_text = "".join(
            item.get("text", "")
            for item in raw_text
            if isinstance(item, dict)
        )

    generated_queries = []

    for line in raw_text.splitlines():
        cleaned = _clean_query(line)

        if not cleaned:
            continue

        # Avoid duplicates
        if cleaned.lower() == query.strip().lower():
            continue

        if cleaned.lower() not in {
            q.lower() for q in generated_queries
        }:
            generated_queries.append(cleaned)

    # Always keep the user's original query.
    final_queries = [query.strip()]

    final_queries.extend(
        generated_queries[:num_queries]
    )

    return final_queries
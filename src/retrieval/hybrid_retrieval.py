from langchain_core.documents import Document
from src.retrieval.bm25_retrieval import search_bm25
from src.retrieval.vector_retrieval import get_vector_store, search
from src.retrieval.reranker import rerank
from src.retrieval.query_translator import generate_queries

def get_dense_results(
    query: str,
    k: int
) -> list[Document]:

    vector_store = get_vector_store()

    return search(
        vector_store,
        query,
        k
    )

def get_bm25_results(
    query: str,
    k: int
) -> list[Document]:

    return search_bm25(
        query,
        k
    )

def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    rrf_k: int = 60
) -> list[Document]:

    rrf_scores = {}
    chunk_lookup = {}

    for results in result_lists:

        for rank, doc in enumerate(
            results,
            start=1
        ):

            text = doc.page_content

            rrf_scores[text] = (
                rrf_scores.get(text, 0)
                + 1 / (rrf_k + rank)
            )

            chunk_lookup[text] = doc

    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk_lookup[text]
        for text, _ in sorted_results
    ]

def retrieve(
    query: str,
    k_retrieval: int = 10,
    k_final: int = 5
) -> list[Document]:

    queries = generate_queries(query)

    print("\n" + "=" * 70)
    print("QUERY TRANSLATION")
    print("=" * 70)

    for i, translated_query in enumerate(
        queries,
        start=1
    ):
        print(f"{i}. {translated_query}")

    dense_results = []
    bm25_results = []

    for translated_query in queries:

        dense_results.append(
            get_dense_results(
                translated_query,
                k_retrieval
            )
        )

        bm25_results.append(
            get_bm25_results(
                translated_query,
                k_retrieval
            )
        )

    all_results = dense_results + bm25_results

    fused_results = reciprocal_rank_fusion(
        all_results
    )

    final_results = rerank(
        query,
        fused_results,
        top_k=k_final
    )

    return final_results


if __name__ == "__main__":

    query = "How does Kubernetes prevent a group of Pods from being partially scheduled when the cluster does not have enough CPU and memory resources to satisfy the requirements of the entire group?"

    results = retrieve(
        query,
        k_retrieval=10,
        k_final=5
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for i, doc in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Rank {i} ---")

        print(
            f"Source: "
            f"{doc.metadata.get('source')}"
        )

        print(
            doc.page_content[:1000]
        )
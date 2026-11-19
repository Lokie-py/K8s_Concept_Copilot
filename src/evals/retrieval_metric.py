import json
from src.retrieval.hybrid_retrieval import retrieve
from src.config import PROJECT_ROOT


def load_eval_set(path=None) -> list[dict]:
    if path is None:
        path = PROJECT_ROOT / "eval_sets" / "eval_set.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_chunk_relevant(chunk_text: str, expected_keywords: list[str]) -> bool:
    """A retrieved chunk counts as 'relevant' if it contains at least one
    of the expected source keywords for that question."""
    chunk_lower = chunk_text.lower()
    return any(keyword.lower() in chunk_lower for keyword in expected_keywords)


def recall_at_k(retrieved_chunks: list, expected_keywords: list[str]) -> int:
    """Returns 1 if ANY retrieved chunk is relevant, else 0."""
    for chunk in retrieved_chunks:
        if is_chunk_relevant(chunk.page_content, expected_keywords):
            return 1
    return 0


def reciprocal_rank(retrieved_chunks: list, expected_keywords: list[str]) -> float:
    """Returns 1/rank of the first relevant chunk, or 0 if none found."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_chunk_relevant(chunk.page_content, expected_keywords):
            return 1 / rank
    return 0.0


def evaluate_retrieval(eval_set: list[dict], k: int = 5) -> dict:
    recall_scores = []
    rr_scores = []
    per_question_results = []

    for item in eval_set:
        question = item["question"]
        expected_keywords = item["expected_source_keywords"]

        retrieved = retrieve(question, k_final=k)

        recall = recall_at_k(retrieved, expected_keywords)
        rr = reciprocal_rank(retrieved, expected_keywords)

        recall_scores.append(recall)
        rr_scores.append(rr)

        per_question_results.append({
            "id": item["id"],
            "question": question,
            "recall": recall,
            "reciprocal_rank": rr
        })

    avg_recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(rr_scores) / len(rr_scores)

    return {
        "avg_recall_at_k": avg_recall,
        "mrr": mrr,
        "per_question": per_question_results
    }


if __name__ == "__main__":
    eval_set = load_eval_set()
    results = evaluate_retrieval(eval_set, k=5)

    print(f"Average Recall@5: {results['avg_recall_at_k']:.2f}")
    print(f"MRR: {results['mrr']:.2f}")
    print()
    for r in results["per_question"]:
        status = "✓" if r["recall"] == 1 else "✗"
        print(f"{status} [{r['id']}] RR={r['reciprocal_rank']:.2f} — {r['question'][:60]}")
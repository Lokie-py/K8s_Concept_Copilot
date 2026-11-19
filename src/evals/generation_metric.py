import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.config import PROJECT_ROOT
from src.retrieval.hybrid_retrieval import retrieve
from src.generation.prompt_builder import build_prompt

load_dotenv()

judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

JUDGE_PROMPT = """
Evaluate the following Kubernetes RAG answer using ONLY the provided context.

Question:
{question}

Expected key points:
{points}

Context:
{context}

Answer:
{answer}

Score each from 0 to 1:

FAITHFULNESS: Are the claims supported by the context?
RELEVANCY: Does the answer directly answer the question?
CORRECTNESS: Does it cover the expected key points?

Return ONLY:
FAITHFULNESS: <score>
RELEVANCY: <score>
CORRECTNESS: <score>
"""

def load_eval_set(path=None) -> list[dict]:
    if path is None:
        path = PROJECT_ROOT / "eval_sets" / "eval_set.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_context(chunks):
    return "\n\n".join(
        f"[Context {i}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks, 1)
    )

def judge(question, answer, context, points):
    prompt = JUDGE_PROMPT.format(
        question=question,
        points="\n".join(f"- {p}" for p in points),
        context=context,
        answer=answer
    )

    response = judge_llm.invoke(prompt)
    result = response.content.strip()

    scores = {}

    for line in result.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)

            try:
                scores[key.strip().lower()] = float(value.strip())
            except ValueError:
                pass

    return scores

def evaluate():
    eval_set = load_eval_set()
    results = []

    for item in eval_set:
        question = item["question"]

        chunks = retrieve(question, k_final=5)
        context = build_context(chunks)

        prompt = build_prompt(question, chunks)
        response = judge_llm.invoke(prompt)

        scores = judge(
            question,
            response,
            context,
            item.get("expected_key_points", [])
        )

        results.append({
            "id": item["id"],
            "faithfulness": scores.get("faithfulness"),
            "relevancy": scores.get("relevancy"),
            "correctness": scores.get("correctness")
        })

    def avg(key):
        values = [
            r[key]
            for r in results
            if r[key] is not None
        ]

        return sum(values) / len(values) if values else 0

    print("=" * 50)
    print("GENERATION EVALUATION")
    print("=" * 50)

    print(f"Faithfulness: {avg('faithfulness'):.2f}")
    print(f"Relevancy:    {avg('relevancy'):.2f}")
    print(f"Correctness:  {avg('correctness'):.2f}")

    print("\nPer-query results:")
    print("-" * 50)

    for r in results:
        print(
            f"[{r['id']}] "
            f"F={r['faithfulness']} "
            f"R={r['relevancy']} "
            f"C={r['correctness']}"
        )

if __name__ == "__main__":
    evaluate()
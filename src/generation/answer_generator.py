import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from src.retrieval.hybrid_retrieval import retrieve
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate

def answer_question(
    question: str,
    k_retrieval: int = 15,
    k_final: int = 5,
) -> str:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunks = retrieve(
        query=question,
        k_retrieval=k_retrieval,
        k_final=k_final,
    )

    if not chunks:
        return (
            "I couldn't find enough information in the "
            "Kubernetes documentation to answer this question."
        )

    prompt = build_prompt(
        query=question,
        chunks=chunks,
    )

    answer = generate(prompt)
    return answer.strip()


if __name__ == "__main__":

    question = (
        "What is a Kubernetes Pod and what resources "
        "do containers inside a Pod share?"
    )

    print(answer_question(question))
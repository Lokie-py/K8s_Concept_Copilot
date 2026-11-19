import pickle
import nltk
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import PROJECT_ROOT

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())

def load_bm25():
    save_path = PROJECT_ROOT / "Data" / "bm_25" / "bm25.pkl"

    with open(save_path, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["documents"]

def search_bm25(
    query: str,
    k: int = 5
) -> list[Document]:

    bm25, documents = load_bm25()

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [
        documents[index]
        for index in top_indices
    ]

if __name__ == "__main__":

    query = "What is a Kubernetes Pod?"

    results = search_bm25(
        query,
        k=5
    )

    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for i, doc in enumerate(results, start=1):

        print(f"\n--- Rank {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content[:1000])
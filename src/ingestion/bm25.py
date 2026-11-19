import nltk
import pickle

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import PROJECT_ROOT
from src.ingestion.vector_store import create_vector_store
from src.retrieval.embedder import get_embedding_model


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

def build_bm25(chunks: list[Document]):

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    save_path = PROJECT_ROOT / "Data" / "bm_25" / "bm25.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "documents": chunks
            },
            f
        )

    print(f"BM25 index saved to: {save_path}")
    print(f"Documents indexed: {len(chunks)}")

def load_bm25():

    save_path = PROJECT_ROOT / "Data" / "bm_25" / "bm25.pkl"

    with open(save_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
 
    embedding_model = get_embedding_model()

    vector_store = create_vector_store(embedding_model)

    data = vector_store.get(
        include=["documents", "metadatas"]
    )

    chunks = [
        Document(
            page_content=doc,
            metadata=metadata
        )
        for doc, metadata in zip(
            data["documents"],
            data["metadatas"]
        )
    ]

    print(f"Retrieved {len(chunks)} chunks from ChromaDB")
    build_bm25(chunks)
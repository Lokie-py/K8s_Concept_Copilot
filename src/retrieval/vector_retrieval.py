from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import FOLDER
from src.retrieval.embedder import get_embedding_model

_vector_store = None

def get_vector_store():

    global _vector_store

    if _vector_store is None:

      #  print("Opening Chroma vector store...")

        _vector_store = Chroma(
            embedding_function=get_embedding_model(),
            persist_directory=str(FOLDER),
            collection_name="Vector_DB"
        )

    return _vector_store


def search(
    vector_store: Chroma,
    query: str,
    k: int = 5
) -> list[Document]:

    return vector_store.similarity_search(
        query=query,
        k=k
    )

''' if __name__ == "__main__":

    vector_store = get_vector_store()

    query = "What is a Kubernetes Pod?"

    results = search(
        vector_store,
        query,
        k=5
    )

    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Rank {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content[:1000])  '''
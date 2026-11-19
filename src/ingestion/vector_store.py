from pathlib import Path

from langchain_chroma import Chroma

from src.config import FOLDER, PROJECT_ROOT
from src.retrieval.embedder import get_embedding_model
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents

RAW_DATA_DIR = PROJECT_ROOT / "Data" / "Raw" / "Concepts" # path

def create_vector_store(embedding_model):
   
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(FOLDER),
        collection_name="Vector_DB"
    )

    return vector_store

def build_vector_store():

    # Load embedding model ONLY ONCE
    embedding_model = get_embedding_model()

    # Create Chroma once
    vector_store = create_vector_store(embedding_model)

    # Find all PDFs
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    if not pdf_files:
        print("No PDF files found.")
        return

    total_chunks = 0

    for pdf_path in pdf_files:

        print("\n" + "=" * 60)
        print(f"Processing: {pdf_path.name}")
        print("=" * 60)

        # 1. Load PDF
        docs = load_pdf(str(pdf_path))

        print(f"Pages loaded: {len(docs)}")

        # 2. Chunk PDF
        chunks = chunk_documents(docs, embedding_model)

        print(f"Chunks created: {len(chunks)}")

        # 3. Add chunks to Chroma
        # Chroma automatically generates embeddings
        # using the embedding_function.
        vector_store.add_documents(chunks)

        total_chunks += len(chunks)

        print(f"Stored: {len(chunks)} chunks")

    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 60)

    print(f"PDFs processed : {len(pdf_files)}")
    print(f"Total chunks   : {total_chunks}")
    print(f"Chroma path    : {FOLDER}")


if __name__ == "__main__":
    build_vector_store()
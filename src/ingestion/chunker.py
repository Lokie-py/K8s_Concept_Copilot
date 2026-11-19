import logging
from langchain_core.documents import Document
from src.retrieval.embedder import get_embedding_model
from langchain_experimental.text_splitter import SemanticChunker
from src.config import SEMANTIC_CHUNKER_BREAKPOINT_TYPE, SEMANTIC_CHUNKER_BREAKPOINT_AMOUNT

logger= logging.getLogger(__name__)

def chunk_documents(docs: list[Document], embedding_model) -> list[Document]:
    chunker= SemanticChunker(
        embedding_model,
        breakpoint_threshold_type= SEMANTIC_CHUNKER_BREAKPOINT_TYPE,
        breakpoint_threshold_amount= SEMANTIC_CHUNKER_BREAKPOINT_AMOUNT
    )

    chunks= chunker.split_documents(docs)
    logger.info("Created %d chunks from %d pages", len(chunks), {len(docs)})

    return chunks

import logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_instance = None  # cached, so we don't reload the model every call

def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
        _reranker_instance = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_instance


def rerank(query: str, chunks: list[Document], top_k: int = 5) -> list[Document]:
    if not chunks:
        return []

    model = get_reranker()
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = model.predict(pairs)

    scored_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
    return ranked_chunks


import pymupdf
import logging
from typing import List
from pathlib import Path
from src.ingestion.text_cleaner import clean_documents
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path) -> List[Document]:

    logger = logging.getLogger(__name__)

    if Path(file_path).exists():
        loader= PyMuPDFLoader(str(file_path))
        docs= loader.load()

        if not docs:
            raise ValueError("No pages extracted from PDF")

        cleaned_docs= clean_documents(docs)
        return cleaned_docs

    else:
        raise FileNotFoundError(f"{file_path} does not exist")

# print(load_pdf("data/raw_pdfs/trial_1.pdf")[0].page_content)
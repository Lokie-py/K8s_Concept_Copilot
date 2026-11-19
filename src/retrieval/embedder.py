import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print("Loading BGE embedding model...")

        _embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    return _embedding_model

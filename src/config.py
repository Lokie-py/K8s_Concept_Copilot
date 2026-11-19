import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEMANTIC_CHUNKER_BREAKPOINT_TYPE = "percentile"
SEMANTIC_CHUNKER_BREAKPOINT_AMOUNT = 95
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOLDER= PROJECT_ROOT / "Data" / "vector_db" / "chroma_db"
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
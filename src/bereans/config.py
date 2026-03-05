"""Configuration loaded from environment variables with sensible defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(BASE_DIR / "data" / "bereans.db")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SUPPORTED_FORMATS = {
    "pdf", "epub", "mobi", "docx", "txt", "md",
    "html", "htm", "csv", "json",
}

"""Central configuration for the Enterprise Private GPT project.

All paths and tunable parameters live here so the other scripts stay small.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
STORAGE = ROOT / "storage"
CHROMA_DIR = STORAGE / "chroma"

# --- Ollama / LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
# e5 models expect "query: " / "passage: " prefixes for best retrieval quality.
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "corporate_legal")

# --- API server (OpenAI-compatible gateway for OpenWebUI) ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_KEY = os.getenv("API_KEY", "rag-local")

# --- RuLegalNER dataset (public Google Drive files) ---
RUL_NER_FILES = {
    "train.csv": "1XJcPycpU1u8Nv8isqzYcE7ox6NNeMu17",
    "validation.csv": "1wA-S4ScPN6RwslJj9mvxtHgwyDXbWUKJ",
    "test.csv": "11I0E5M_QN3s7DuisXJuMMn949PeM7AS4",
}

# How many source documents to convert into the demo knowledge base.
MAX_DOCS = int(os.getenv("MAX_DOCS", "400"))

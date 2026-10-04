"""Multilingual E5 embeddings wrapper (CPU friendly).

E5 models are trained with "query: " / "passage: " prefixes, so we apply them
explicitly and normalise the vectors. Implements the LangChain Embeddings
interface, so it plugs directly into Chroma and retrievers.
"""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

import config


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = config.EMBEDDING_MODEL, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"[embeddings] loading {model_name} on {device} ...")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [config.EMBEDDING_PASSAGE_PREFIX + t for t in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            [config.EMBEDDING_QUERY_PREFIX + text],
            normalize_embeddings=True,
        )
        return vector[0].tolist()


def get_embeddings() -> E5Embeddings:
    return E5Embeddings()

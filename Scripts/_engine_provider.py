"""Lazy singleton provider for the RAG engine.

The engine class lives in ``04_rag.py`` (a numbered script, so not importable by
name). We load it by path once and cache the instance.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_engine_class():
    path = Path(__file__).resolve().parent / "04_rag.py"
    spec = importlib.util.spec_from_file_location("rag_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RAG engine from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RagEngine


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        print("[api] initialising RAG engine (loading embeddings + LLM client) ...")
        _engine = _load_engine_class()()
    return _engine

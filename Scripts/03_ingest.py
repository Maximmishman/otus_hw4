"""Build the local vector database (ChromaDB) from prepared documents.

Run:  python Scripts/03_ingest.py [--reset]

Reads data/processed/*.txt, splits them into overlapping chunks and stores the
embeddings in storage/chroma. Use --reset to wipe the existing collection first.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from embeddings import get_embeddings  # noqa: E402


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(config.DATA_PROCESSED.glob("*.txt")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        ))
    return docs


def main() -> None:
    reset = "--reset" in sys.argv
    if reset and config.CHROMA_DIR.exists():
        print(f"[ingest] reset: removing {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)

    docs = load_documents()
    if not docs:
        raise SystemExit("No documents in data/processed. Run 02_prepare_docs.py first.")
    print(f"[ingest] documents: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] chunks: {len(chunks)} (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"[ok] persisted to {config.CHROMA_DIR}")
    print(f"[ok] vectors in collection: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()

"""RAG core: retrieve relevant chunks from ChromaDB and answer with a local LLM.

Can be used as a library (RagEngine) or from the CLI:

    python Scripts/04_rag.py "Какое наказание назначено по делу № 5-29/2013?"
    python Scripts/04_rag.py --top-k 6 "Какой срок ареста назначен Шевченко?"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from embeddings import get_embeddings  # noqa: E402

SYSTEM_PROMPT = (
    "Ты — корпоративный ассистент, который отвечает на вопросы сотрудников "
    "исключительно по внутренней базе документов.\n"
    "Правила:\n"
    "1. Отвечай ТОЛЬКО на основе приведённого ниже контекста.\n"
    "2. Не используй собственные общие знания и не додумывай факты.\n"
    "3. Если в контексте нет ответа — прямо скажи: "
    "«В предоставленных документах нет информации по этому вопросу».\n"
    "4. Отвечай по-русски, кратко и по существу.\n"
    "5. В конце ответа укажи имена файлов-источников в формате: "
    "Источники: <файл1>, <файл2>."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])


class RagEngine:
    """Loads the vector store and LLM once and answers questions."""

    def __init__(self) -> None:
        if not config.CHROMA_DIR.exists():
            raise SystemExit("Vector DB not found. Run Scripts/03_ingest.py first.")
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )
        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            base_url=config.OLLAMA_BASE_URL,
        )
        self.chain = PROMPT | self.llm | StrOutputParser()

    def retrieve(self, question: str, top_k: int | None = None):
        k = top_k or config.TOP_K
        return self.vectorstore.similarity_search(question, k=k)

    @staticmethod
    def _format_context(docs) -> str:
        blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            blocks.append(f"[{i}] ({source})\n{doc.page_content}")
        return "\n\n".join(blocks)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        started = time.perf_counter()
        docs = self.retrieve(question, top_k=top_k)
        context = self._format_context(docs)
        text = self.chain.invoke({"context": context, "question": question})
        latency = time.perf_counter() - started
        return {
            "answer": text.strip(),
            "sources": [d.metadata.get("source", "unknown") for d in docs],
            "contexts": [d.page_content for d in docs],
            "latency_sec": round(latency, 2),
            "top_k": top_k or config.TOP_K,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local RAG assistant a question.")
    parser.add_argument("question", help="Question in Russian")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    engine = RagEngine()
    result = engine.answer(args.question, top_k=args.top_k)
    print("\n=== ОТВЕТ ===")
    print(result["answer"])
    print(f"\n=== ИСТОЧНИКИ (top_k={result['top_k']}, {result['latency_sec']}s) ===")
    for s in dict.fromkeys(result["sources"]):
        print(" -", s)


if __name__ == "__main__":
    main()

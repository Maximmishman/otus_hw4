"""Evaluate the RAG system: three test types + Top-K / latency analysis.

Run:  python Scripts/06_evaluate.py

Writes notes/test_results.md and notes/test_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from _engine_provider import get_engine  # noqa: E402

TESTS = [
    {
        "category": "1. Запрос по базе",
        "question": "Какая сумма была взыскана по делу № 2-745/2013 и в чью пользу?",
        "expected": "10300 руб. в пользу ООО «Капитал»",
    },
    {
        "category": "2. Запрос вне базы",
        "question": "Дай пошаговый рецепт шоколадного пирога.",
        "expected": "Отказ: информации в документах нет",
    },
    {
        "category": "3. Конфликт фактов",
        "question": (
            "Какое наказание назначили Кулагину В.Н. по делу №1-12/13 за организацию "
            "незаконного пребывания иностранного гражданина — лишение свободы или штраф?"
        ),
        "expected": "Штраф 5 000 руб. (приоритет документа над общими знаниями)",
    },
]

TOP_K_VALUES = [2, 4, 8]


def main() -> None:
    engine = get_engine()
    results = []

    for test in TESTS:
        for k in TOP_K_VALUES:
            print(f"[eval] {test['category']} | top_k={k} ...")
            res = engine.answer(test["question"], top_k=k)
            record = {
                **test,
                "top_k": k,
                "latency_sec": res["latency_sec"],
                "answer": res["answer"],
                "sources": list(dict.fromkeys(res["sources"])),
            }
            results.append(record)
            print(f"        {res['latency_sec']}s | {res['answer'][:90]!r}")

    out_json = config.ROOT / "notes" / "test_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Результаты тестирования RAG-системы",
        "",
        f"- LLM: `{config.LLM_MODEL}` (Ollama, {config.OLLAMA_BASE_URL})",
        f"- Эмбеддинги: `{config.EMBEDDING_MODEL}`",
        f"- Векторная БД: ChromaDB (`{config.CHROMA_DIR.name}`), top-K: {TOP_K_VALUES}",
        f"- chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}",
        "",
        "## Сводная таблица",
        "",
        "| Тип теста | top-K | Latency, с | Ответ (сокращённо) |",
        "|---|---:|---:|---|",
    ]
    for r in results:
        short = r["answer"].replace("\n", " ").replace("|", "/")
        if len(short) > 160:
            short = short[:157] + "..."
        lines.append(f"| {r['category']} | {r['top_k']} | {r['latency_sec']} | {short} |")

    lines += ["", "## Полные ответы", ""]
    for r in results:
        lines += [
            f"### {r['category']} — top_k={r['top_k']} ({r['latency_sec']} с)",
            "",
            f"**Вопрос:** {r['question']}",
            "",
            f"**Ожидалось:** {r['expected']}",
            "",
            f"**Ответ:**",
            "",
            r["answer"],
            "",
            f"**Источники:** {', '.join(r['sources'])}",
            "",
        ]

    out_md = config.ROOT / "notes" / "test_results.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()

"""Convert RuLegalNER CSV rows into plain-text documents for the RAG knowledge base.

Run:  python Scripts/02_prepare_docs.py

Input : data/raw/{train,validation,test}.csv  (no header; col0 = document text,
        col1 = JSON NER annotations, which we drop).
Output: data/processed/*.txt  and  data/processed/manifest.jsonl
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MIN_CHARS = 300


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


def title_of(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 3:
            return line[:120]
    return ""


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest_path = config.DATA_PROCESSED / "manifest.jsonl"

    # newest split first so the demo is deterministic
    splits = ["train.csv", "validation.csv", "test.csv"]
    written = 0
    per_split: dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for split in splits:
            csv_path = config.DATA_RAW / split
            if not csv_path.exists():
                print(f"[warn] {csv_path} not found, skipping")
                continue
            split_name = csv_path.stem
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader):
                    if written >= config.MAX_DOCS:
                        break
                    if not row:
                        continue
                    text = clean_text(row[0])
                    if len(text) < MIN_CHARS:
                        continue
                    doc_id = f"{split_name}_{row_idx:06d}"
                    (config.DATA_PROCESSED / f"{doc_id}.txt").write_text(text, encoding="utf-8")
                    manifest.write(json.dumps({
                        "doc_id": doc_id,
                        "source": f"{split}#row{row_idx}",
                        "title": title_of(text),
                        "n_chars": len(text),
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    per_split[split_name] = per_split.get(split_name, 0) + 1
            if written >= config.MAX_DOCS:
                break

    print(f"[ok] wrote {written} documents to {config.DATA_PROCESSED} (limit={config.MAX_DOCS})")
    for name, cnt in per_split.items():
        print(f"     {name}: {cnt}")


if __name__ == "__main__":
    main()

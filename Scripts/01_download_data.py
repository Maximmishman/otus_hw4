"""Download the RuLegalNER dataset CSVs from the public Google Drive folder.

Run:  python Scripts/01_download_data.py

Files land in data/raw/. Re-running skips files that already exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, file_id in config.RUL_NER_FILES.items():
        target = config.DATA_RAW / name
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {name} already present ({target.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"[download] {name} (id={file_id}) ...")
        result = gdown.download(id=file_id, output=str(target), quiet=False)
        if not result or not target.exists():
            raise RuntimeError(f"Failed to download {name}")
        print(f"[ok] {name}: {target.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

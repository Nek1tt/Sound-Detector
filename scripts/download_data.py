#!/usr/bin/env python3
"""
Скачивает датасет ESC-50 в папку data/.

Использование:
    python scripts/download_data.py
    python scripts/download_data.py --dir /path/to/data
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def download_esc50(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = target_dir / "ESC-50-master"

    if out_dir.exists():
        print(f"[download] ESC-50 уже скачан: {out_dir}")
        return

    zip_path = target_dir / "esc50.zip"
    print(f"[download] Скачиваем ESC-50...")

    def _progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            print(f"\r  {pct}%  {done // 1024 // 1024} MB / {total // 1024 // 1024} MB", end="", flush=True)

    urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=_progress)
    print()

    print(f"[download] Распаковываем в {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    zip_path.unlink()

    print(f"[download] ✓ Готово: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Скачать датасет ESC-50")
    parser.add_argument("--dir", default="data", help="Папка назначения")
    args = parser.parse_args()
    download_esc50(Path(args.dir))


if __name__ == "__main__":
    main()

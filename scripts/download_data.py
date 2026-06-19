"""
download_data.py — fetch the Kaggle 'Give Me Some Credit' dataset into Data/.

Downloads the competition files via kagglehub and copies them into the
repository's Data/ directory (resolved relative to this file, so it works
from any working directory).

Requires Kaggle API credentials configured for kagglehub
(see https://github.com/Kaggle/kagglehub#authenticate).

Usage:
    python scripts/download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

# Data/ lives one level up from scripts/ (repo root / Data)
DEST = Path(__file__).resolve().parent.parent / "Data"


def download() -> None:
    print("Downloading GiveMeSomeCredit dataset from Kaggle...")
    path = Path(kagglehub.competition_download("GiveMeSomeCredit"))
    print(f"Cached at: {path}")

    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in path.iterdir():
        if src.is_file():                      # skip any nested directories
            shutil.copy2(src, DEST / src.name)
            print(f"  -> {src.name}")
            copied += 1

    print(f"Done. Copied {copied} file(s) to: {DEST}")


if __name__ == "__main__":
    download()

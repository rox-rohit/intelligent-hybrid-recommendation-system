"""Download the MovieLens 25M dataset into data/raw/ with checksum verification.

Usage:
    python scripts/download_data.py

The raw archive is never modified in place; the pipeline reads from
data/raw/ and writes cleaned outputs to data/processed/.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

# --- Source of truth: GroupLens research datasets -------------------------
# https://grouplens.org/datasets/movielens/25m/
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
EXPECTED_MD5 = "3c07bb3b7e24af3a3d9a744dbb118848"  # verify against GroupLens page
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_FILES = [
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
    "genome-scores.csv",
    "genome-tags.csv",
    "README.txt",
]


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 hash of a file, streaming to bound memory use."""
    digest = hashlib.md5()  # noqa: S324 — integrity check, not security
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-25m.zip"

    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        with requests.get(DATASET_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        size_mb = zip_path.stat().st_size / (1 << 20)
        print(f"Downloaded {size_mb:.1f} MB -> {zip_path}")
    else:
        print(f"Archive already present: {zip_path}")

    checksum = md5sum(zip_path)
    if checksum != EXPECTED_MD5:
        print(f"ERROR: checksum mismatch for {zip_path}")
        print(f"  expected {EXPECTED_MD5}")
        print(f"  got      {checksum}")
        print("Delete the archive and re-run, or update EXPECTED_MD5 after")
        print("verifying the hash on the GroupLens website.")
        return 1
    print(f"Checksum OK ({checksum})")

    extract_dir = RAW_DIR / "ml-25m"
    if not extract_dir.exists():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        print(f"Extracted -> {extract_dir}")
    else:
        print(f"Already extracted: {extract_dir}")

    missing = [f for f in EXPECTED_FILES if not (extract_dir / f).exists()]
    if missing:
        print(f"ERROR: expected files missing after extraction: {missing}")
        return 1

    print("MovieLens 25M dataset ready:")
    for f in EXPECTED_FILES:
        p = extract_dir / f
        print(f"  {f:20s} {p.stat().st_size / (1 << 20):8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

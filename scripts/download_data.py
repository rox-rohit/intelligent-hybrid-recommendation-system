"""Download the MovieLens 25M dataset into data/raw/ with checksum verification.

Usage:
    python scripts/download_data.py

The raw archive is never modified in place; the pipeline reads from
data/raw/ and writes cleaned outputs to data/processed/.
"""

from __future__ import annotations

import hashlib
import sys
import time
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


def download_with_resume(url: str, dest: Path, max_retries: int = 8) -> None:
    """Stream `url` to `dest`, resuming from partial file on retry."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        offset = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 90), headers=headers) as resp:
                if offset and resp.status_code == 416:
                    return  # file already complete (server says range unsatisfiable)
                resp.raise_for_status()
                if offset and resp.status_code != 206:
                    # server ignored Range: restart from scratch
                    offset = 0
                    mode = "wb"
                else:
                    mode = "ab" if offset else "wb"
                with dest.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return
        except (requests.ConnectionError, requests.Timeout) as err:
            last_err = err
            wait = min(2**attempt, 60)
            print(f"  attempt {attempt}: {err.__class__.__name__} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {max_retries} attempts: {last_err}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-25m.zip"

    if not zip_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        download_with_resume(DATASET_URL, zip_path)
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

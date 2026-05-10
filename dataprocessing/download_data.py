#!/usr/bin/env python3
"""Fetch the UCI Heart Disease zip and install required raw files into data/.

Idempotent and SHA-256 verified. Stdlib-only on purpose so this can run before
any project dependencies are installed (CI bootstrap, fresh clone, etc.).

CLI:
    python dataprocessing/download_data.py            # idempotent: no-op when files match
    python dataprocessing/download_data.py --force    # re-download even if data is valid

Library:
    from dataprocessing.download_data import ensure_data
    ensure_data()                              # call from any flow that needs data
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ZIP_SHA256 = "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc"

# Only files actually consumed by the pipeline. Everything else in the zip
# (other regions, costs/, raw cleveland.data, etc.) is intentionally ignored.
REQUIRED_FILES: dict[str, str] = {
    "processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "heart-disease.names":      "ec7ed5aa4ed8321097808f47b836301c2a0c36f3bb09c40b8b43b622363e6b61",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HTTP_TIMEOUT_SEC = 30
CHUNK_SIZE = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_match() -> bool:
    for name, want in REQUIRED_FILES.items():
        target = DATA_DIR / name
        if not target.is_file() or _sha256(target) != want:
            return False
    return True


def _download(url: str, dst: Path) -> None:
    print(f"Downloading: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "heart-disease-mlops/1.0"},
    )
    # Default urlopen validates the server certificate; HTTPS pinned URL above.
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out, length=CHUNK_SIZE)


def ensure_data(force: bool = False) -> bool:
    """Guarantee `data/` contains the required raw files.

    Returns True if no work was done (already valid), False if a fresh
    download/extract happened. Raises RuntimeError on integrity failure.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _files_match():
        print("data/ is already up to date (matching SHA-256). Use --force to re-download.")
        return True

    with tempfile.TemporaryDirectory(prefix="heart-disease-download-") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "heart-disease.zip"

        _download(ZIP_URL, zip_path)

        print("Verifying zip checksum...")
        got = _sha256(zip_path)
        if got != ZIP_SHA256:
            raise RuntimeError(
                "Zip SHA-256 mismatch.\n"
                f"  expected: {ZIP_SHA256}\n"
                f"  got:      {got}"
            )
        print(f"  OK ({got})")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            members = set(zf.namelist())
            missing = [n for n in REQUIRED_FILES if n not in members]
            if missing:
                raise RuntimeError(f"Required entries not found in zip: {missing}")
            # Selective extract -- never let zip choose its own paths.
            for name in REQUIRED_FILES:
                zf.extract(name, extract_dir)

        print("Verifying file checksums and installing into data/...")
        for name, want in REQUIRED_FILES.items():
            src = extract_dir / name
            got = _sha256(src)
            if got != want:
                raise RuntimeError(
                    f"SHA-256 mismatch for {name}.\n"
                    f"  expected: {want}\n"
                    f"  got:      {got}"
                )
            dst = DATA_DIR / name
            shutil.copyfile(src, dst)
            dst.chmod(0o644)
            print(f"  {name}  OK ({got})")

    print(f"Done. Files written to: {DATA_DIR}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if data is already present and valid",
    )
    args = parser.parse_args(argv)

    try:
        ensure_data(force=args.force)
    except (urllib.error.URLError, RuntimeError, zipfile.BadZipFile, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

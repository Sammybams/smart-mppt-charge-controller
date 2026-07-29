#!/usr/bin/env python3
"""Download and extract the pinned UCP photovoltaic dataset summaries."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://data.mendeley.com/public-files/datasets/z93gzbptf7/files/"
    "88796d89-2328-4440-b381-508585150ac1/file_downloaded"
)
EXPECTED_SHA256 = (
    "22e39cc0b074d9ffd09459851c34898a54652ae9113661a118e2cc6270a08ae8"
)
ARCHIVE_NAME = "PV_Dataset.zip"
MEMBERS = {
    (
        "PV_Dataset/Partial_Shading_Data/"
        "new_inferred_file_7line_PS.csv"
    ): "partial_shading_summary.csv",
    (
        "PV_Dataset/Uniform_Irradiance_Data/"
        "new_inferred_file_4line_FS.csv"
    ): "uniform_irradiance_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "smart-mppt-dataset-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify(archive: Path) -> None:
    actual = sha256(archive)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum mismatch for {archive}: expected {EXPECTED_SHA256}, "
            f"received {actual}"
        )


def extract(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member, output_name in MEMBERS.items():
            output_path = destination / output_name
            with bundle.open(member) as source, output_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(output_path)
    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even when the archive already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.root / "data" / "raw" / ARCHIVE_NAME
    destination = args.root / "data" / "source"

    try:
        download(archive, force=args.force_download)
        verify(archive)
        extracted = extract(archive, destination)
    except Exception as exc:
        print(f"Dataset setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Verified {archive} ({EXPECTED_SHA256})")
    for path in extracted:
        print(f"Extracted {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

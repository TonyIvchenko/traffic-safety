from __future__ import annotations

import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from common import US_ACCIDENTS_PATH, US_ACCIDENTS_RAW_DIR, ensure_dirs


DATASET = "sobhanmoosavi/us-accidents"
CSV_NAME = "US_Accidents_March23.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    target = US_ACCIDENTS_PATH
    if target.exists() and not args.force:
        print(f"exists {target}")
        return

    api = KaggleApi()
    api.authenticate()
    print(f"download {DATASET}")
    api.dataset_download_files(
        DATASET,
        path=str(US_ACCIDENTS_RAW_DIR),
        unzip=True,
        quiet=False,
        force=args.force,
    )

    source = US_ACCIDENTS_RAW_DIR / CSV_NAME
    if not source.exists():
        candidates = sorted(US_ACCIDENTS_RAW_DIR.glob("US_Accidents*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"expected {CSV_NAME} after download into {US_ACCIDENTS_RAW_DIR}"
            )
        source = candidates[0]

    if source != target:
        if target.exists():
            target.unlink()
        source.rename(target)

    zip_path = US_ACCIDENTS_RAW_DIR / "us-accidents.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"wrote {target}")
    print(f"size_bytes={target.stat().st_size}")


if __name__ == "__main__":
    main()

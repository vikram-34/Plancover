"""Run GMC policy extraction for a folder of PDFs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-extract GMC policy PDFs into JSON files.")
    parser.add_argument(
        "pdf_folder",
        nargs="?",
        type=Path,
        default=Path("sample_docs"),
        help="Folder containing policy PDFs (default: sample_docs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Destination directory for extracted JSON (default: output)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    batch = process_folder(args.pdf_folder, args.output_dir)
    print(
        f"Completed: {batch.succeeded_count} succeeded, "
        f"{batch.failed_count} failed, {len(batch.files)} total."
    )
    print("Alias mappings used:")
    if batch.alias_mappings:
        for source, targets in sorted(batch.alias_mappings.items(), key=lambda item: item[0].casefold()):
            for target, count in sorted(targets.items()):
                print(f"  {source!r} -> {target} ({count}x)")
    else:
        print("  none")


if __name__ == "__main__":
    main()

"""CLI for ingesting and normalizing every raw source."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.pipeline import RawIngestionPipeline
from src.utils.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingère un fichier supporté ou tous les fichiers d'un dossier")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("src/raw"),
        help="Fichier unique ou dossier contenant les sources CSV/JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cache/ingested"))
    parser.add_argument("--max-rows", type=int, help="Limite utile pour un test")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    manifest = RawIngestionPipeline(args.raw_dir, args.output_dir, args.max_rows).run()
    for item in manifest["files"]:
        print(f"{item['source']} -> {item['output']} ({item['rows']:,} lignes)")


if __name__ == "__main__":
    main()

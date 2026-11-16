"""CLI for the full semantic profiling pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embeddings.embedding_cache import DEFAULT_EMBEDDING_CACHE_PATH
from src.embeddings.model import DEFAULT_TEI_ENDPOINT
from src.semantic.semantic_profiler import SemanticProfiler
from src.utils.logging_config import configure_logging, log_stage
import logging


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profilage classique et sémantique du cache OSIRIS")
    parser.add_argument("--cache", type=Path, default=Path("cache/profile.parquet"))
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--fields", help="Champs textuels pour le matching, séparés par des virgules")
    parser.add_argument("--embed", action="store_true", help="Génère les embeddings et lance le k-NN")
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_TEI_ENDPOINT, help="URL de Text Embeddings Inference")
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    log_stage(logger, "SEMANTIC_PROFILE_START", cache=args.cache, sample_size=args.sample_size, embeddings=args.embed)

    fields = [field.strip() for field in args.fields.split(",") if field.strip()] if args.fields else None
    profiler = SemanticProfiler(
        args.cache,
        sample_size=args.sample_size,
        embedding_model=args.model,
        embedding_cache=args.embedding_cache,
        endpoint=args.endpoint,
    )
    report = profiler.profile(
        fuzzy_fields=fields,
        generate_embeddings=args.embed,
    )
    log_stage(logger, "SEMANTIC_PROFILE_DONE", columns=len(report.get("classical", {}).get("column_profiles", [])))
    destination = args.json_out
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Rapport écrit : {destination}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

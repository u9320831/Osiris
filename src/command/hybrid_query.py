"""Build the no-embedding index and run fast hybrid semantic queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embeddings.embedding_cache import DEFAULT_EMBEDDING_CACHE_PATH
from src.embeddings.model import DEFAULT_MODEL, DEFAULT_TEI_ENDPOINT
from src.semantic.hybrid_search import HybridSearch
from src.utils.logging_config import configure_logging
from src.vector_search.lexical_index import LexicalIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Recherche hybride rapide OSIRIS")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-index", help="Construit l'index lexical sans embeddings")
    build.add_argument("--shards-dir", type=Path, default=Path("cache/ingested"))
    build.add_argument("--index", type=Path, default=Path("cache/lexical.sqlite"))
    build.add_argument(
        "--index-shards",
        type=int,
        default=4,
        help="Nombre de bases SQLite à alimenter en parallèle",
    )
    build.add_argument("--batch-size", type=int, default=4_096)
    build.add_argument("--commit-every", type=int, default=100_000)
    build.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    search = sub.add_parser("search", help="Recherche lexicale puis reranking par embedding")
    search.add_argument("text")
    search.add_argument("--index", type=Path, default=Path("cache/lexical.sqlite"))
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--candidates", type=int, default=200)
    search.add_argument("--model", default=None)
    search.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    search.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Nombre maximal de textes par requête TEI",
    )
    search.add_argument(
        "--embedding-concurrency",
        type=int,
        default=4,
        help="Nombre maximal de requêtes TEI simultanées pour le reclassement",
    )
    search.add_argument("--endpoint", default=DEFAULT_TEI_ENDPOINT, help="URL de Text Embeddings Inference")
    search.add_argument("--json-out", type=Path)
    search.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    if args.command == "build-index":
        index = LexicalIndex(args.index, index_shards=args.index_shards)
        try:
            print(index.build(args.shards_dir, args.batch_size, args.commit_every))
        finally:
            index.close()
        return
    engine = HybridSearch(
        args.index,
        args.model or DEFAULT_MODEL,
        args.embedding_cache,
        args.endpoint,
        args.embedding_batch_size,
        args.embedding_concurrency,
    )
    try:
        results = engine.search(args.text, args.limit, args.candidates)
    finally:
        engine.close()
    rendered = json.dumps(results, ensure_ascii=False, indent=2, default=str)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
        print(f"Résultats écrits : {args.json_out} ({len(results)} résultats)")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

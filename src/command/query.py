"""Search every normalized ingestion shard with a free-text query."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import time

import polars as pl

from src.embeddings.embedding_cache import DEFAULT_EMBEDDING_CACHE_PATH
from src.embeddings.model import DEFAULT_MODEL, DEFAULT_TEI_ENDPOINT
from src.ingestion.cache import search_text_expression
from src.ingestion.pipeline import normalize_column_name
from src.similarity.text_similarity import normalize_text
from src.utils.logging_config import configure_logging, log_stage


TOKEN_RE = re.compile(r"[\w@.-]+", re.UNICODE)
logger = logging.getLogger(__name__)


class UnifiedQuery:
    def __init__(self, shards_dir: Path | str = Path("cache/ingested")) -> None:
        self.shards_dir = Path(shards_dir)
        self.pattern = str(self.shards_dir / "*.parquet")
        shards = list(self.shards_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"Aucun shard parquet dans {self.shards_dir}. "
                "Lancez d'abord : python -m src.command.ingest_raw"
            )
        names: list[str] = []
        for shard in shards:
            for name in pl.scan_parquet(shard).collect_schema().names():
                if name not in names:
                    names.append(name)
        # Data fields are strings; row ids retain their integer type. Supplying
        # the union schema prevents Polars from dropping columns found later.
        self.schema = {
            name: pl.UInt64 if name == "_row_id" else pl.String
            for name in names
        }
        log_stage(logger, "QUERY_READY", shards=len(shards), columns=len(self.schema), directory=self.shards_dir)

    def _field_columns(self, field: str) -> list[str]:
        normalized = normalize_column_name(field)
        aliases = {
            "name": ("name", "nom"),
            "first_name": ("first_name", "prenom"),
            "firstname": ("firstname", "prenom"),
            "address": ("address", "adresse"),
            "phone": ("phone", "telephone", "tel", "mobile"),
        }.get(normalized, (normalized,))
        matches = []
        for alias in aliases:
            matches.extend(
                name for name in self.schema
                if name == alias or name.startswith(f"{alias}_")
            )
        matches = list(dict.fromkeys(matches))
        if not matches:
            raise ValueError(
                f"Colonne inconnue : {field}. Colonnes disponibles : "
                + ", ".join(self.schema)
            )
        return matches

    def search(
        self,
        text: str = "",
        limit: int = 100,
        match_all: bool = False,
        filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        if not text.strip() and not filters:
            raise ValueError("Indiquez un texte ou au moins un filtre de colonne")
        if limit <= 0:
            raise ValueError("limit doit être strictement positif")

        normalized_query = normalize_text(text, remove_punctuation=False)
        terms = list(dict.fromkeys(token for token in TOKEN_RE.findall(normalized_query) if token.strip()))
        if not terms and text.strip():
            terms = [normalized_query]
        query = pl.scan_parquet(
            self.pattern,
            schema=self.schema,
            missing_columns="insert",
            extra_columns="ignore",
        )
        query = query.with_columns(search_text_expression(self.schema).alias("_search_text"))
        text_matches = [pl.col("_search_text").str.contains(term, literal=True) for term in terms]
        predicates = []
        score_parts = [match.cast(pl.UInt8) for match in text_matches]
        if text_matches:
            predicates.append(pl.all_horizontal(text_matches) if match_all else pl.any_horizontal(text_matches))

        for field, value in (filters or {}).items():
            normalized_value = normalize_text(value, remove_punctuation=False)
            field_matches = [
                pl.col(column).str.to_lowercase().str.contains(normalized_value, literal=True)
                for column in self._field_columns(field)
            ]
            predicates.append(pl.any_horizontal(field_matches))
            score_parts.append(pl.any_horizontal(field_matches).cast(pl.UInt8))

        predicate = pl.all_horizontal(predicates)
        score = pl.sum_horizontal(score_parts).alias("_match_score")
        return (
            query.with_columns(score)
            .filter(predicate)
            .sort("_match_score", descending=True)
            .head(limit)
            .collect()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recherche dans tous les fichiers raw normalisés")
    parser.add_argument("query", nargs="?", default="", help="Texte libre à rechercher")
    parser.add_argument("--shards-dir", type=Path, default=Path("cache/ingested"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all-terms", action="store_true", help="Exige que tous les termes apparaissent")
    parser.add_argument("--name", "--nom", dest="name", help="Filtre sur le nom")
    parser.add_argument("--prenom", help="Filtre sur le prénom")
    parser.add_argument("--ville", help="Filtre sur la ville")
    parser.add_argument("--email", help="Filtre sur l'adresse email")
    parser.add_argument("--phone", "--telephone", dest="phone", help="Filtre sur le téléphone")
    parser.add_argument("--adresse", "--address", dest="adresse", help="Filtre sur l'adresse")
    parser.add_argument("--where", action="append", metavar="COLONNE=VALEUR", help="Filtre générique répétable")
    parser.add_argument("--json", type=Path, help="Exporte les résultats en JSON")
    parser.add_argument("--hybrid", action="store_true", help="Recherche lexicale puis reranking par embedding")
    parser.add_argument("--lexical-index", type=Path, default=Path("cache/lexical.sqlite"))
    parser.add_argument("--candidates", type=int, default=200, help="Nombre de candidats avant embedding")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Nombre maximal de textes par requête TEI",
    )
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=4,
        help="Nombre maximal de requêtes TEI simultanées pour le reclassement",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modèle chargé par TEI")
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--endpoint", default=DEFAULT_TEI_ENDPOINT, help="URL de Text Embeddings Inference")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    started = time.perf_counter()
    if args.hybrid:
        if not args.query:
            parser.error("Le mode --hybrid nécessite une requête texte")
        from src.semantic.hybrid_search import HybridSearch

        engine = HybridSearch(
            args.lexical_index,
            args.model,
            args.embedding_cache,
            args.endpoint,
            args.embedding_batch_size,
            args.embedding_concurrency,
        )
        try:
            results = engine.search(args.query, args.limit, args.candidates)
        finally:
            engine.close()
        log_stage(logger, "QUERY_DONE", rows=len(results), mode="hybrid", seconds=f"{time.perf_counter() - started:.2f}")
        rendered = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(rendered, encoding="utf-8")
            print(f"Résultats écrits : {args.json} ({len(results)} lignes)")
        else:
            print(rendered)
        return
    filters = {
        field: value for field, value in {
            "name": args.name,
            "prenom": args.prenom,
            "ville": args.ville,
            "email": args.email,
            "phone": args.phone,
            "adresse": args.adresse,
        }.items() if value
    }
    for condition in args.where or []:
        if "=" not in condition:
            parser.error(f"--where doit respecter COLONNE=VALEUR : {condition}")
        field, value = condition.split("=", 1)
        if not field.strip() or not value.strip():
            parser.error(f"--where doit contenir une colonne et une valeur : {condition}")
        filters[field.strip()] = value.strip()
    log_stage(logger, "QUERY_START", text=args.query or "<filters>", filters=filters, limit=args.limit)
    results = UnifiedQuery(args.shards_dir).search(args.query, args.limit, args.all_terms, filters)
    log_stage(logger, "QUERY_DONE", rows=results.height, seconds=f"{time.perf_counter() - started:.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(results.write_json(), encoding="utf-8")
        print(f"Résultats écrits : {args.json} ({results.height} lignes)")
    else:
        print(results)


if __name__ == "__main__":
    main()

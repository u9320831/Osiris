"""Load CSV/NDJSON files into the local parquet cache."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
import logging
from pathlib import Path
import tempfile

import polars as pl

from src.command.detector import detect_csv
from src.ingestion.cache import scan_ndjson_as_strings
from src.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)


@dataclass
class Content:
    """A source file and the parquet cache generated from it."""

    file: Path | str
    chunk: int = 10_000
    cache_dir: Path | str = field(default_factory=lambda: Path("cache"))
    cache_name: str = "profile.parquet"
    min_confidence: float = 0.70

    def __post_init__(self) -> None:
        self.file = Path(self.file)
        self.cache_dir = Path(self.cache_dir)

        if not self.file.exists():
            raise FileNotFoundError(f"Fichier source introuvable : {self.file}")
        if not self.file.is_file():
            raise ValueError(f"Le chemin source ne désigne pas un fichier : {self.file}")
        if self.chunk <= 0:
            raise ValueError("chunk doit être strictement positif")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence doit être compris entre 0 et 1")

        logger.info("[+] Fichier source : %s", self.file)

    @property
    def cache_path(self) -> Path:
        """Path used consistently by both writing and reading the cache."""

        return self.cache_dir / self.cache_name

    def _prepare_cache_path(self, output: Path | str | None = None) -> Path:
        target = Path(output) if output is not None else self.cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _utf8_source(self, encoding: str, directory: Path) -> tuple[Path, Path | None]:
        """Return a Polars-compatible UTF-8 source and optional temp path."""

        normalized = encoding.lower().replace("-", "_")
        if normalized in {"utf8", "utf_8", "ascii"}:
            return self.file, None

        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".osiris-",
            suffix=".utf8.csv",
            dir=directory,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with self.file.open("rb") as source, temporary:
                while block := source.read(max(self.chunk, 64 * 1024)):
                    temporary.write(decoder.decode(block).encode("utf-8"))
                temporary.write(decoder.decode(b"", final=True).encode("utf-8"))
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path, temporary_path

    def read_csv(self, output: Path | str | None = None) -> Path:
        """Detect and stream the CSV source into a parquet cache."""

        config = detect_csv(self.file)
        confidence = config.get("confidence", 0.0)
        if confidence < self.min_confidence:
            raise ValueError(
                "Détection CSV trop incertaine "
                f"({confidence:.1%} < {self.min_confidence:.1%})"
            )

        target = self._prepare_cache_path(output)
        logger.info(
            "[+] CSV détecté : encodage=%s, séparateur=%r, guillemet=%r, "
            "colonnes=%s, confiance=%.1f%%",
            config["encoding"],
            config["separator"],
            config["quote_char"],
            config.get("columns", "?"),
            confidence * 100,
        )

        source, temporary_source = self._utf8_source(config["encoding"], target.parent)
        try:
            query = pl.scan_csv(
                source,
                separator=config["separator"],
                quote_char=config["quote_char"],
                encoding="utf8",
                infer_schema=False,
                has_header=True,
            )

            # Validate reader settings before writing the cache.
            logger.info("[+] Colonnes chargées : %s", query.collect_schema().names())
            query.sink_parquet(target)
            logger.info("[+] Cache parquet écrit : %s", target)
        finally:
            if temporary_source is not None:
                temporary_source.unlink(missing_ok=True)
        return target

    def read_json(self, output: Path | str | None = None) -> Path:
        """Stream an NDJSON source into the same parquet cache location."""

        target = self._prepare_cache_path(output)
        query = scan_ndjson_as_strings(self.file)
        query.sink_parquet(target)
        logger.info("[+] Cache parquet écrit : %s", target)
        return target

    def load_cache(self, cache: Path | str | None = None) -> pl.LazyFrame:
        """Return a lazy frame over the generated parquet cache."""

        target = Path(cache) if cache is not None else self.cache_path
        if not target.exists():
            raise FileNotFoundError(
                f"Cache parquet introuvable : {target}. Appelez read_csv() ou read_json()."
            )
        return pl.scan_parquet(target)


def main() -> None:
    """Small command-line entry point for the sample project input."""

    configure_logging(log_path="log/OSIRIS.log")
    profile = Content(file=Path("src/raw/Bouygues.json"))
    profile.read_json()
    print(profile.load_cache().head(10).collect())


if __name__ == "__main__":
    main()

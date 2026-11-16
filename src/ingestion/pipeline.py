"""Ingest every supported file from ``src/raw`` into normalized parquet shards."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from typing import Any

import polars as pl

from src.command.detector import detect_csv
from src.command.loader import Content
from src.ingestion.cache import scan_ndjson_as_strings
from src.utils.logging_config import log_stage


CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson"}


def normalize_column_name(name: str) -> str:
    name = CAMEL_RE.sub("_", str(name))
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = name.casefold()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "column"


def normalized_column_map(names: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: dict[str, int] = {}
    for name in names:
        base = normalize_column_name(name)
        count = used.get(base, 0)
        normalized = base if count == 0 else f"{base}_{count + 1}"
        used[base] = count + 1
        result[name] = normalized
    return result


def _compact_json(value: str) -> str:
    """Remove JSON formatting whitespace without changing string contents."""

    compact: list[str] = []
    in_string = False
    escaped = False
    for char in value:
        if in_string:
            compact.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            compact.append(char)
        elif not char.isspace():
            compact.append(char)
    return "".join(compact)


def _json_array_to_ndjson(
    source: Path,
    directory: Path,
    max_rows: int | None = None,
) -> Path:
    """Convert a top-level JSON array of objects to NDJSON using bounded memory."""

    decoder = json.JSONDecoder()
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".osiris-",
        suffix=".ndjson",
        dir=directory,
        delete=False,
    )
    temporary_path = Path(temporary.name)

    try:
        with source.open("r", encoding="utf-8-sig") as stream, temporary:
            buffer = ""
            position = 0
            eof = False

            def read_more() -> None:
                nonlocal buffer, eof
                chunk = stream.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            def skip_whitespace() -> None:
                nonlocal buffer, position
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer) or eof:
                        return
                    buffer = ""
                    position = 0
                    read_more()

            def require_character(expected: str) -> None:
                nonlocal buffer, position
                skip_whitespace()
                if position == len(buffer):
                    raise ValueError(f"Invalid JSON array in {source}: expected {expected!r}")
                if buffer[position] != expected:
                    raise ValueError(f"Invalid JSON array in {source}: expected {expected!r}")
                position += 1

            read_more()
            require_character("[")
            skip_whitespace()
            if position < len(buffer) and buffer[position] == "]":
                raise ValueError(f"JSON array in {source} contains no rows")

            rows = 0
            limited = False
            while True:
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError as error:
                        if eof:
                            raise ValueError(
                                f"Invalid JSON array in {source} at character {position + error.pos}"
                            ) from error
                        if position:
                            buffer = buffer[position:]
                            position = 0
                        read_more()

                if not isinstance(value, dict):
                    raise ValueError(f"JSON array in {source} must contain objects")
                temporary.write(_compact_json(buffer[position:end]))
                temporary.write("\n")
                rows += 1
                if max_rows is not None and rows >= max_rows:
                    limited = True
                    break
                position = end
                skip_whitespace()
                if position == len(buffer):
                    raise ValueError(f"Invalid JSON array in {source}: missing closing bracket")
                if buffer[position] == "]":
                    position += 1
                    break
                if buffer[position] != ",":
                    raise ValueError(f"Invalid JSON array in {source}: expected ',' or ']'")
                position += 1
                skip_whitespace()

            if not limited:
                while True:
                    skip_whitespace()
                    if position == len(buffer):
                        break
                    raise ValueError(f"Invalid JSON array in {source}: unexpected trailing content")

        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _split_concatenated_ndjson(
    source: Path,
    directory: Path,
    max_rows: int | None = None,
) -> tuple[Path, int]:
    """Split concatenated records and count recognizable HTTP error pages."""

    decoder = json.JSONDecoder()
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".osiris-",
        suffix=".ndjson",
        dir=directory,
        delete=False,
    )
    temporary_path = Path(temporary.name)

    try:
        rows = 0
        skipped_http_error_pages = 0
        inside_http_error_page = False
        with source.open("r", encoding="utf-8-sig") as stream, temporary:
            for line_number, line in enumerate(stream, start=1):
                lowered_line = line.casefold()
                if inside_http_error_page:
                    if "</html>" in lowered_line:
                        inside_http_error_page = False
                    continue

                position = 0
                while True:
                    while position < len(line) and line[position].isspace():
                        position += 1
                    if position == len(line):
                        break
                    remaining = line[position:]
                    lowered = remaining.casefold()
                    if (
                        lowered.startswith(("<html", "<!doctype html"))
                        and re.search(r"\b5\d{2}\b", lowered) is not None
                    ):
                        skipped_http_error_pages += 1
                        inside_http_error_page = "</html>" not in lowered
                        break
                    try:
                        value, end = decoder.raw_decode(line, position)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"JSON invalide dans {source}, ligne {line_number}, "
                            f"caractère {position + error.pos}: {error.msg}"
                        ) from error
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Enregistrement non objet dans {source}, ligne {line_number}"
                        )
                    temporary.write(_compact_json(line[position:end]))
                    temporary.write("\n")
                    rows += 1
                    if max_rows is not None and rows >= max_rows:
                        return temporary_path, skipped_http_error_pages
                    position = end
        if inside_http_error_page:
            raise ValueError(f"Page d'erreur HTTP non terminée dans {source}")
        return temporary_path, skipped_http_error_pages
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass
class RawIngestionPipeline:
    raw_dir: Path | str = Path("src/raw")
    output_dir: Path | str = Path("cache/ingested")
    max_rows: int | None = None

    logger = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.output_dir = Path(self.output_dir)
        if self.raw_dir.is_file():
            if self.raw_dir.suffix.casefold() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Format de fichier non supporté : {self.raw_dir.suffix}")
        elif not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Fichier ou dossier raw introuvable : {self.raw_dir}")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows doit être positif ou None")

    def discover(self) -> list[Path]:
        if self.raw_dir.is_file():
            files = [self.raw_dir]
            location = self.raw_dir.parent
        else:
            files = sorted(
                path for path in self.raw_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
            )
            location = self.raw_dir
        log_stage(self.logger, "DISCOVERY", directory=location, files=len(files))
        for path in files:
            self.logger.info("[DISCOVERY] source=%s size_mb=%.2f", path.name, path.stat().st_size / 1024**2)
        return files

    def _scan(self, source: Path) -> tuple[pl.LazyFrame, str | None]:
        if source.suffix.casefold() == ".csv":
            config = detect_csv(source)
            self.logger.info(
                "[DETECT] source=%s encoding=%s separator=%r quote=%r confidence=%.1f%%",
                source.name,
                config["encoding"],
                config["separator"],
                config["quote_char"],
                config.get("confidence", 0) * 100,
            )
            # Polars scan_csv only accepts UTF-8. Content performs a streamed
            # transcode for cp1252/UTF-16 and removes the temporary source.
            helper = Content(source, cache_dir=self.output_dir)
            converted, temporary = helper._utf8_source(config["encoding"], self.output_dir)
            query = pl.scan_csv(
                converted,
                separator=config["separator"],
                quote_char=config["quote_char"],
                encoding="utf8",
                infer_schema=False,
                has_header=True,
            )
            return query, str(temporary) if temporary is not None else None
        first = None
        with source.open("r", encoding="utf-8-sig") as stream:
            while chunk := stream.read(4096):
                first = next((char for char in chunk if not char.isspace()), None)
                if first is not None:
                    break
        if first == "[":
            self.logger.info("[DETECT] source=%s format=json-array", source.name)
            converted = _json_array_to_ndjson(source, self.output_dir, self.max_rows)
            return scan_ndjson_as_strings(converted), str(converted)
        self.logger.info("[DETECT] source=%s format=ndjson", source.name)
        return scan_ndjson_as_strings(source), None

    def _write_query(
        self,
        query: pl.LazyFrame,
        source: Path,
        target: Path,
        started: float,
        temporary: str | None,
        skipped_http_error_pages: int = 0,
    ) -> dict[str, Any]:
        schema = query.collect_schema()
        mapping = normalized_column_map(list(schema.names()))
        self.logger.info(
            "[NORMALIZE] source=%s columns=%d temporary=%s",
            source.name,
            len(mapping),
            temporary is not None,
        )
        selected = [
            pl.col(old_name).cast(pl.String, strict=False).alias(new_name)
            for old_name, new_name in mapping.items()
        ]
        if self.max_rows is not None:
            query = query.head(self.max_rows)
        query = query.select(selected)
        query = query.with_columns(
            pl.int_range(0, pl.len(), dtype=pl.UInt64).alias("_row_id"),
            pl.lit(source.stem).alias("_source"),
            pl.lit(source.name).alias("_source_file"),
        )
        self.logger.info("[WRITE] source=%s target=%s", source.name, target)
        query.sink_parquet(target, row_group_size=2_048, engine="streaming")
        rows = int(pl.scan_parquet(target).select(pl.len()).collect().item())
        log_stage(
            self.logger,
            "DONE",
            source=source.name,
            rows=rows,
            seconds=f"{time.perf_counter() - started:.2f}",
        )
        return {
            "source": str(source),
            "output": str(target),
            "rows": rows,
            "skipped_http_error_pages": skipped_http_error_pages,
            "columns": mapping,
        }

    def ingest_file(self, source: Path) -> dict[str, Any]:
        started = time.perf_counter()
        log_stage(self.logger, "START", source=source.name, suffix=source.suffix)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("[DETECT] source=%s", source.name)
        query, temporary = self._scan(source)
        target = self.output_dir / f"{source.stem}.{source.suffix.lstrip('.').casefold()}.parquet"
        try:
            try:
                return self._write_query(query, source, target, started, temporary)
            except pl.exceptions.ComputeError as error:
                suffix = source.suffix.casefold()
                if (
                    temporary is not None
                    or suffix not in {".json", ".jsonl", ".ndjson"}
                    or "TapeError" not in str(error)
                ):
                    raise

                self.logger.warning(
                    "[REPAIR] source=%s JSON lines were not separated correctly; retrying with streamed record splitting",
                    source.name,
                )
                repaired, skipped_http_error_pages = _split_concatenated_ndjson(
                    source,
                    self.output_dir,
                    self.max_rows,
                )
                try:
                    repaired_query = scan_ndjson_as_strings(repaired)
                    result = self._write_query(
                        repaired_query,
                        source,
                        target,
                        started,
                        str(repaired),
                        skipped_http_error_pages,
                    )
                    if skipped_http_error_pages:
                        self.logger.warning(
                            "[REPAIR] source=%s skipped_http_error_pages=%d",
                            source.name,
                            skipped_http_error_pages,
                        )
                    return result
                finally:
                    repaired.unlink(missing_ok=True)
                    self.logger.info("[CLEANUP] temporary=%s", repaired)
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
                self.logger.info("[CLEANUP] temporary=%s", temporary)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        log_stage(self.logger, "PIPELINE_START", raw_dir=self.raw_dir, output_dir=self.output_dir)
        files = self.discover()
        if not files:
            raise FileNotFoundError(f"Aucun fichier supporté dans : {self.raw_dir}")
        results = [self.ingest_file(source) for source in files]
        manifest = {
            "raw_dir": str(self.raw_dir),
            "output_dir": str(self.output_dir),
            "files": results,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_stage(
            self.logger,
            "PIPELINE_DONE",
            files=len(results),
            rows=sum(int(item["rows"]) for item in results),
            seconds=f"{time.perf_counter() - started:.2f}",
        )
        return manifest

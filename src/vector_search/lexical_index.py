"""Fast no-embedding candidate index using SQLite FTS5."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import polars as pl

from src.ingestion.cache import search_text_expression
from src.similarity.text_similarity import normalize_text
from src.utils.logging_config import log_stage


TOKEN_RE = re.compile(r"[\w@.-]+", re.UNICODE)
logger = logging.getLogger(__name__)


class LexicalIndex:
    """Persistent full-text index for candidate generation, without vectors."""

    def __init__(
        self,
        path: str | Path = "cache/lexical.sqlite",
        index_shards: int | None = None,
        *,
        _open_direct: bool = False,
        _cache_size_kib: int = 262_144,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._indexes: list[LexicalIndex] | None = None
        self._pending_writes = 0
        self._write_lock = threading.Lock()

        if index_shards is not None:
            if index_shards <= 0:
                raise ValueError("index_shards doit être strictement positif")
            paths = (
                [self.path]
                if index_shards == 1
                else [
                    self.path.with_name(
                        f"{self.path.stem}.part-{part:03d}-of-{index_shards:03d}{self.path.suffix}"
                    )
                    for part in range(index_shards)
                ]
            )
            cache_size_kib = max(16_384, 262_144 // len(paths))
            self._indexes = [
                LexicalIndex(
                    index_path,
                    _open_direct=True,
                    _cache_size_kib=cache_size_kib,
                )
                for index_path in paths
            ]
            self._index_paths = paths
            self._manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
            return

        self._manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        if not _open_direct and self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            names = manifest.get("indexes")
            if not isinstance(names, list) or not names or any(
                not isinstance(name, str) or Path(name).name != name for name in names
            ):
                raise ValueError(f"Manifeste d’index lexical invalide : {self._manifest_path}")
            paths = [self.path.parent / name for name in names]
            missing = [index_path for index_path in paths if not index_path.is_file()]
            if missing:
                raise FileNotFoundError(f"Base(s) d’index lexical manquante(s) : {missing}")
        else:
            paths = [self.path]
        self._index_paths = paths
        if len(paths) > 1:
            cache_size_kib = max(16_384, 262_144 // len(paths))
            self._indexes = [
                LexicalIndex(
                    index_path,
                    _open_direct=True,
                    _cache_size_kib=cache_size_kib,
                )
                for index_path in paths
            ]
            return
        self.path = paths[0]
        self._initialize_database(_cache_size_kib)

    def _initialize_database(self, cache_size_kib: int) -> None:
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5("
            "source_file UNINDEXED, row_id UNINDEXED, text, shard UNINDEXED, detail=none)"
        )
        definition = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()[0]
        self.detail_none = (
            re.search(r"\bdetail\s*=\s*['\"]?none\b", definition, re.IGNORECASE) is not None
        )
        if not self.detail_none:
            logger.warning(
                "Existing lexical index uses full positional detail; rebuild it to use the smaller detail=none layout"
            )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(documents)")
        }
        if "shard" not in columns:
            self.connection.execute("DROP TABLE documents")
            self.connection.execute(
                "CREATE VIRTUAL TABLE documents USING fts5("
                "source_file UNINDEXED, row_id UNINDEXED, text, shard UNINDEXED, detail=none)"
            )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS shard_state ("
            "shard TEXT PRIMARY KEY, size INTEGER NOT NULL, "
            "modified_ns INTEGER NOT NULL, rows INTEGER NOT NULL)"
        )
        # FTS5 is write-heavy; these settings reduce fsync overhead while
        # retaining recoverability through the WAL journal.
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(f"PRAGMA cache_size=-{cache_size_kib}")
        self.connection.execute("PRAGMA mmap_size=67108864")
        self.connection.commit()

    def build(
        self,
        shards_dir: str | Path = "cache/ingested",
        batch_size: int = 4_096,
        commit_every: int = 100_000,
    ) -> dict[str, int]:
        if self._indexes is not None:
            return self._build_sharded(shards_dir, batch_size, commit_every)

        started = time.perf_counter()
        shards = sorted(Path(shards_dir).glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"Aucun shard parquet dans {shards_dir}")
        if batch_size <= 0 or commit_every <= 0:
            raise ValueError("batch_size et commit_every doivent être positifs")
        shard_stats = {
            str(shard.resolve()): shard.stat()
            for shard in shards
        }
        existing_shards = {
            row[0]
            for row in self.connection.execute("SELECT shard FROM shard_state")
        }
        has_documents = self.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM documents LIMIT 1)"
        ).fetchone()[0]
        initial_build = not has_documents and not existing_shards
        removed_shards = existing_shards - shard_stats.keys()
        inserted = 0
        pending = 0
        read_seconds = 0.0
        insert_seconds = 0.0
        if initial_build:
            self.connection.execute(
                "INSERT INTO documents(documents, rank) VALUES ('automerge', 0)"
            )
            self.connection.execute(
                "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 64)"
            )
            self.connection.commit()
        try:
            self.connection.execute("BEGIN")
            for shard_path in removed_shards:
                self.connection.execute("DELETE FROM documents WHERE shard = ?", (shard_path,))
                self.connection.execute("DELETE FROM shard_state WHERE shard = ?", (shard_path,))

            for shard in shards:
                shard_path = str(shard.resolve())
                stat = shard_stats[shard_path]
                state = self.connection.execute(
                    "SELECT size, modified_ns FROM shard_state WHERE shard = ?",
                    (shard_path,),
                ).fetchone()
                if state == (stat.st_size, stat.st_mtime_ns):
                    log_stage(logger, "LEXICAL_SHARD_SKIP", shard=shard.name, reason="unchanged")
                    continue

                log_stage(logger, "LEXICAL_SHARD_START", shard=shard.name)
                self.connection.execute("DELETE FROM documents WHERE shard = ?", (shard_path,))
                shard_rows = 0
                query = pl.scan_parquet(shard)
                schema = query.collect_schema()
                source_column = pl.col("_source_file") if "_source_file" in schema else pl.lit(shard.name)
                row_column = pl.col("_row_id") if "_row_id" in schema else pl.int_range(0, pl.len())
                batch = query.select(
                    source_column.alias("source_file"),
                    row_column.alias("row_id"),
                    search_text_expression(schema.names(), normalize=False).alias("text"),
                )
                frames = iter(batch.collect_batches(chunk_size=batch_size))
                while True:
                    read_started = time.perf_counter()
                    try:
                        frame = next(frames)
                    except StopIteration:
                        break
                    read_seconds += time.perf_counter() - read_started
                    insert_started = time.perf_counter()
                    self.connection.executemany(
                        "INSERT INTO documents(source_file, row_id, text, shard) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            (*row, shard_path)
                            for row in frame.iter_rows(buffer_size=batch_size)
                        ),
                    )
                    insert_seconds += time.perf_counter() - insert_started
                    inserted += frame.height
                    shard_rows += frame.height
                    pending += frame.height
                    if pending >= commit_every:
                        self.connection.commit()
                        self.connection.execute("BEGIN")
                        pending = 0
                    if inserted % 100_000 < frame.height:
                        elapsed = time.perf_counter() - started
                        log_stage(
                            logger,
                            "LEXICAL_PROGRESS",
                            rows=inserted,
                            rows_per_second=f"{inserted / elapsed:.0f}",
                            parquet_seconds=f"{read_seconds:.2f}",
                            insert_seconds=f"{insert_seconds:.2f}",
                        )
                self.connection.execute(
                    "INSERT OR REPLACE INTO shard_state(shard, size, modified_ns, rows) "
                    "VALUES (?, ?, ?, ?)",
                    (shard_path, stat.st_size, stat.st_mtime_ns, shard_rows),
                )
            if initial_build:
                if inserted:
                    optimize_started = time.perf_counter()
                    self.connection.execute(
                        "INSERT INTO documents(documents) VALUES ('optimize')"
                    )
                    log_stage(
                        logger,
                        "LEXICAL_OPTIMIZE",
                        seconds=f"{time.perf_counter() - optimize_started:.2f}",
                    )
                self.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('automerge', 4)"
                )
                self.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 16)"
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            if initial_build:
                self.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('automerge', 4)"
                )
                self.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 16)"
                )
                self.connection.commit()
            raise
        total_rows = int(
            self.connection.execute("SELECT COALESCE(SUM(rows), 0) FROM shard_state").fetchone()[0]
        )
        log_stage(
            logger,
            "LEXICAL_DONE",
            rows=total_rows,
            updated_rows=inserted,
            shards=len(shards),
            removed_shards=len(removed_shards),
            seconds=f"{time.perf_counter() - started:.2f}",
            parquet_seconds=f"{read_seconds:.2f}",
            insert_seconds=f"{insert_seconds:.2f}",
        )
        return {"rows": total_rows, "updated_rows": inserted, "shards": len(shards)}

    def _build_sharded(
        self,
        shards_dir: str | Path,
        batch_size: int,
        commit_every: int,
    ) -> dict[str, int]:
        started = time.perf_counter()
        shards = sorted(Path(shards_dir).glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"Aucun shard parquet dans {shards_dir}")
        if batch_size <= 0 or commit_every <= 0:
            raise ValueError("batch_size et commit_every doivent être positifs")

        indexes = self._indexes or []
        index_count = len(indexes)
        shard_stats = {str(shard.resolve()): shard.stat() for shard in shards}
        changed_shards: set[str] = set()
        removed_shards: set[str] = set()
        initial_builds: list[bool] = []

        for index in indexes:
            existing_shards = {
                row[0] for row in index.connection.execute("SELECT shard FROM shard_state")
            }
            has_documents = index.connection.execute(
                "SELECT EXISTS(SELECT 1 FROM documents LIMIT 1)"
            ).fetchone()[0]
            initial_builds.append(not has_documents and not existing_shards)
            removed_shards.update(existing_shards - shard_stats.keys())
            for shard_path, stat in shard_stats.items():
                state = index.connection.execute(
                    "SELECT size, modified_ns FROM shard_state WHERE shard = ?",
                    (shard_path,),
                ).fetchone()
                if state != (stat.st_size, stat.st_mtime_ns):
                    changed_shards.add(shard_path)

        for index, initial_build in zip(indexes, initial_builds, strict=True):
            if initial_build:
                index.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('automerge', 0)"
                )
                index.connection.execute(
                    "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 64)"
                )
            for shard_path in removed_shards | changed_shards:
                index.connection.execute("DELETE FROM documents WHERE shard = ?", (shard_path,))
                index.connection.execute("DELETE FROM shard_state WHERE shard = ?", (shard_path,))
            index.connection.commit()

        inserted = 0
        inserted_by_index = [0] * index_count
        read_seconds = 0.0
        insert_seconds = 0.0
        try:
            with ThreadPoolExecutor(max_workers=index_count) as executor:
                for shard in shards:
                    shard_path = str(shard.resolve())
                    stat = shard_stats[shard_path]
                    if shard_path not in changed_shards:
                        log_stage(logger, "LEXICAL_SHARD_SKIP", shard=shard.name, reason="unchanged")
                        continue

                    log_stage(logger, "LEXICAL_SHARD_START", shard=shard.name)
                    query = pl.scan_parquet(shard)
                    schema = query.collect_schema()
                    source_column = pl.col("_source_file") if "_source_file" in schema else pl.lit(shard.name)
                    row_column = pl.col("_row_id") if "_row_id" in schema else pl.int_range(0, pl.len())
                    batch = query.select(
                        source_column.alias("source_file"),
                        row_column.alias("row_id"),
                        search_text_expression(schema.names(), normalize=False).alias("text"),
                    )
                    frames = iter(batch.collect_batches(chunk_size=batch_size))
                    shard_rows = 0
                    row_offset = 0
                    while True:
                        read_started = time.perf_counter()
                        try:
                            frame = next(frames)
                        except StopIteration:
                            break
                        read_seconds += time.perf_counter() - read_started
                        buckets: list[list[tuple[Any, ...]]] = [[] for _ in indexes]
                        for row_number, row in enumerate(
                            frame.iter_rows(buffer_size=batch_size), start=row_offset
                        ):
                            buckets[row_number % index_count].append((*row, shard_path))
                        row_offset += frame.height
                        shard_rows += frame.height

                        insert_started = time.perf_counter()
                        futures = [
                            executor.submit(index._insert_rows, rows, commit_every)
                            for index, rows in zip(indexes, buckets, strict=True)
                            if rows
                        ]
                        for future in futures:
                            future.result()
                        insert_seconds += time.perf_counter() - insert_started
                        inserted += frame.height
                        for index_number, rows in enumerate(buckets):
                            inserted_by_index[index_number] += len(rows)
                        if inserted % 100_000 < frame.height:
                            elapsed = time.perf_counter() - started
                            log_stage(
                                logger,
                                "LEXICAL_PROGRESS",
                                rows=inserted,
                                rows_per_second=f"{inserted / elapsed:.0f}",
                                parquet_seconds=f"{read_seconds:.2f}",
                                insert_seconds=f"{insert_seconds:.2f}",
                            )

                    for number, index in enumerate(indexes):
                        rows = (shard_rows + index_count - number - 1) // index_count
                        with index._write_lock:
                            index.connection.execute(
                                "INSERT OR REPLACE INTO shard_state(shard, size, modified_ns, rows) "
                                "VALUES (?, ?, ?, ?)",
                                (shard_path, stat.st_size, stat.st_mtime_ns, rows),
                            )
                            index.connection.commit()
                            index._pending_writes = 0

            for index, initial_build, index_inserted in zip(
                indexes, initial_builds, inserted_by_index, strict=True
            ):
                if initial_build:
                    if index_inserted:
                        optimize_started = time.perf_counter()
                        index.connection.execute("INSERT INTO documents(documents) VALUES ('optimize')")
                        log_stage(
                            logger,
                            "LEXICAL_OPTIMIZE",
                            index=index.path.name,
                            seconds=f"{time.perf_counter() - optimize_started:.2f}",
                        )
                    index.connection.execute(
                        "INSERT INTO documents(documents, rank) VALUES ('automerge', 4)"
                    )
                    index.connection.execute(
                        "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 16)"
                    )
                index.connection.commit()
        except Exception:
            for index, initial_build in zip(indexes, initial_builds, strict=True):
                with index._write_lock:
                    index.connection.rollback()
                    index._pending_writes = 0
                    if initial_build:
                        index.connection.execute(
                            "INSERT INTO documents(documents, rank) VALUES ('automerge', 4)"
                        )
                        index.connection.execute(
                            "INSERT INTO documents(documents, rank) VALUES ('crisismerge', 16)"
                        )
                        index.connection.commit()
            raise

        total_rows = sum(
            int(index.connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM shard_state"
            ).fetchone()[0])
            for index in indexes
        )
        self._write_manifest()
        log_stage(
            logger,
            "LEXICAL_DONE",
            rows=total_rows,
            updated_rows=inserted,
            shards=len(shards),
            indexes=index_count,
            removed_shards=len(removed_shards),
            seconds=f"{time.perf_counter() - started:.2f}",
            parquet_seconds=f"{read_seconds:.2f}",
            insert_seconds=f"{insert_seconds:.2f}",
        )
        return {
            "rows": total_rows,
            "updated_rows": inserted,
            "shards": len(shards),
            "indexes": index_count,
        }

    def _insert_rows(self, rows: list[tuple[Any, ...]], commit_every: int) -> None:
        with self._write_lock:
            self.connection.executemany(
                "INSERT INTO documents(source_file, row_id, text, shard) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._pending_writes += len(rows)
            if self._pending_writes >= commit_every:
                self.connection.commit()
                self._pending_writes = 0

    def _write_manifest(self) -> None:
        manifest = {"version": 1, "indexes": [path.name for path in self._index_paths]}
        temporary_path = self._manifest_path.with_suffix(
            self._manifest_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self._manifest_path)

    def search(
        self,
        text: str,
        limit: int = 200,
        *,
        match_any: bool = False,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit doit être strictement positif")
        if self._indexes is not None:
            with ThreadPoolExecutor(max_workers=len(self._indexes)) as executor:
                results_by_index = list(
                    executor.map(
                        lambda index: index.search(text, limit, match_any=match_any),
                        self._indexes,
                    )
                )
            fused: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
            for hits in results_by_index:
                for rank, hit in enumerate(hits, start=1):
                    key = (hit["source_file"], hit["row_id"], hit["shard"])
                    score = 1.0 / (60 + rank)
                    previous = fused.get(key)
                    if previous is None:
                        fused[key] = (score, hit)
                    else:
                        fused[key] = (previous[0] + score, previous[1])
            return [
                hit
                for _, hit in sorted(fused.values(), key=lambda item: item[0], reverse=True)[:limit]
            ]

        terms = [
            part
            for token in TOKEN_RE.findall(normalize_text(text, remove_punctuation=False))
            for part in re.findall(r"\w+", token, re.UNICODE)
            if part
        ]
        if not terms:
            return []
        # Quote terms so user punctuation cannot alter the FTS query syntax.
        operator = " OR " if match_any else " AND "
        match = operator.join(f'"{term.replace(chr(34), "")}"' for term in terms)
        rows = self.connection.execute(
            "SELECT source_file, row_id, text, shard, bm25(documents) "
            "FROM documents WHERE documents MATCH ? ORDER BY bm25(documents) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [
            {
                "source_file": row[0],
                "row_id": int(row[1]),
                "text": row[2],
                "shard": row[3],
                "lexical_score": -float(row[4]),
            }
            for row in rows
        ]

    def close(self) -> None:
        if self._indexes is not None:
            for index in self._indexes:
                index.close()
        else:
            self.connection.close()

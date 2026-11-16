"""Column-level classical profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.ingestion.cache import scan_cache
from src.profiling.statistics import column_statistics, exact_duplicate_stats
from src.profiling.type_inference import infer_semantic_type, is_sensitive_name, physical_type


class ColumnProfiler:
    def __init__(self, cache: str | Path, sample_size: int = 10_000, top_n: int = 10) -> None:
        self.cache = Path(cache)
        self.sample_size = sample_size
        self.top_n = top_n

    def profile(self) -> dict[str, Any]:
        query = scan_cache(self.cache)
        schema = query.collect_schema()
        columns = list(schema.names())
        sample = query.head(self.sample_size).collect()
        row_count = int(query.select(pl.len()).collect().item())
        profiles = []
        for name in columns:
            stats = column_statistics(query, name, schema[name], self.top_n)
            for key in ("min", "max"):
                if stats[key] is not None and not isinstance(stats[key], (str, int, float, bool)):
                    stats[key] = str(stats[key])
            values = sample.get_column(name).to_list()
            non_null = int(stats["non_null_count"] or 0)
            non_empty = int(stats["non_empty_count"] or 0)
            null_count = int(stats["null_count"] or 0)
            empty_count = int(stats["empty_count"] or 0)
            profiles.append({
                "name": name,
                "physical_type": physical_type(schema[name]),
                "semantic_type": infer_semantic_type(name, str(schema[name]), values),
                "sensitive": is_sensitive_name(name),
                "null_count": null_count,
                "null_rate": round(null_count / row_count, 6) if row_count else 0.0,
                "effective_null_count": null_count + empty_count,
                "effective_null_rate": round((null_count + empty_count) / row_count, 6) if row_count else 0.0,
                "empty_count": empty_count,
                "empty_rate": round(empty_count / row_count, 6) if row_count else 0.0,
                "non_empty_count": non_empty,
                "cardinality": int(stats["distinct_count"] or 0),
                "cardinality_rate": round((stats["distinct_count"] or 0) / non_null, 6) if non_null else 0.0,
                "duplicate_values": max(non_empty - int(stats["distinct_count"] or 0), 0),
                "min": stats["min"],
                "max": stats["max"],
                "top_values": stats["top_values"],
            })
        return {
            "cache": str(self.cache),
            "rows": row_count,
            "columns": len(columns),
            "exact_duplicates": exact_duplicate_stats(query, columns),
            "column_profiles": profiles,
        }

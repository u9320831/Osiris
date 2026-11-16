"""Reusable Polars aggregations for profiling."""

from __future__ import annotations

from typing import Any

import polars as pl


def column_statistics(query: pl.LazyFrame, name: str, dtype: pl.DataType, top_n: int = 10) -> dict[str, Any]:
    column = pl.col(name)
    cleaned = column.cast(pl.String).str.strip_chars() if dtype == pl.String else column
    empty = (
        cleaned.eq("").fill_null(False).sum()
        if dtype == pl.String
        else pl.lit(0)
    )
    non_empty = cleaned.ne("").fill_null(False) if dtype == pl.String else column.is_not_null()
    result = query.select(
        [
            pl.len().alias("row_count"),
            column.is_null().sum().alias("null_count"),
            column.is_not_null().sum().alias("non_null_count"),
            cleaned.filter(non_empty).n_unique().alias("distinct_count"),
            empty.alias("empty_count"),
            non_empty.sum().alias("non_empty_count"),
            column.min().alias("min"),
            column.max().alias("max"),
        ]
    ).collect().row(0, named=True)
    counts = (
        query.select(cleaned.alias("value"))
        .filter(pl.col("value").is_not_null() & pl.col("value").ne(""))
        .group_by("value")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
        .collect()
    )
    result["top_values"] = counts.to_dicts()
    return result


def exact_duplicate_stats(query: pl.LazyFrame, columns: list[str]) -> dict[str, int | float]:
    rows = int(query.select(pl.len()).collect().item())
    if columns:
        duplicate_rows = int(query.select(pl.struct(columns).is_duplicated().sum()).collect().item() or 0)
        duplicate_groups = int(
            query.group_by(columns)
            .agg(pl.len().alias("group_size"))
            .filter(pl.col("group_size") > 1)
            .select(pl.len())
            .collect()
            .item()
            or 0
        )
    else:
        duplicate_rows = duplicate_groups = 0
    return {
        "rows": rows,
        "unique_rows": rows - duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rows / rows, 6) if rows else 0.0,
    }

"""Format and cross-field consistency checks."""

from __future__ import annotations

from typing import Any

import polars as pl

from src.profiling.type_inference import EMAIL_RE, PHONE_RE, _parse_date


def _valid_ratio(values: list[Any], predicate: Any) -> float:
    values = [value for value in values if value is not None and str(value).strip()]
    return round(sum(bool(predicate(str(value).strip())) for value in values) / len(values), 6) if values else 1.0


def check_consistency(frame: pl.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in frame.columns:
        values = frame.get_column(name).to_list()
        lowered = name.casefold()
        if "email" in lowered or "mail" in lowered:
            report[name] = {"rule": "email", "valid_rate": _valid_ratio(values, lambda value: EMAIL_RE.fullmatch(value))}
        elif any(token in lowered for token in ("phone", "tel", "mobile")):
            report[name] = {"rule": "phone", "valid_rate": _valid_ratio(values, lambda value: PHONE_RE.fullmatch(value))}
        elif any(token in lowered for token in ("date", "naissance", "birth")):
            report[name] = {"rule": "date", "valid_rate": _valid_ratio(values, lambda value: _parse_date(value) is not None)}
    return report

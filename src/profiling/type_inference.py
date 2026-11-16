"""Physical and semantic type inference."""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Iterable

import polars as pl


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PHONE_RE = re.compile(r"^[+()\d][\d .()/-]{5,}$")
INTEGER_RE = re.compile(r"^[+-]?\d+$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$")
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
SENSITIVE_PARTS = ("email", "mail", "phone", "tel", "mobile", "iban", "bic", "password", "secret")


def is_sensitive_name(name: str) -> bool:
    lowered = name.casefold()
    return any(part in lowered for part in SENSITIVE_PARTS)


def _parse_date(value: str) -> date | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def infer_semantic_type(name: str, declared_type: str, values: Iterable[Any]) -> str:
    non_empty = [str(value).strip() for value in values if value is not None and str(value).strip()]
    if not non_empty:
        return "empty"
    lowered_name = name.casefold()
    lowered_values = {value.casefold() for value in non_empty}
    # Column semantics are stronger than a sample of values: postal codes,
    # IBANs and phone numbers are all numeric-looking strings.
    if any(token in lowered_name for token in ("iban", "bic", "code", "postal", "identifiant", "id")):
        return "identifier"
    if any(token in lowered_name for token in ("email", "mail")):
        return "email" if all(EMAIL_RE.fullmatch(value) for value in non_empty) else "text"
    if any(token in lowered_name for token in ("phone", "tel", "mobile", "telephone")):
        return "phone" if all(PHONE_RE.fullmatch(value) for value in non_empty) else "text"
    if any(token in lowered_name for token in ("date", "naissance", "birth")):
        return "date" if all(_parse_date(value) is not None for value in non_empty) else "text"
    if lowered_values <= {"true", "false", "yes", "no", "oui", "non", "0", "1"}:
        return "boolean"
    if all(EMAIL_RE.fullmatch(value) for value in non_empty):
        return "email"
    if all(_parse_date(value) is not None for value in non_empty):
        return "date"
    if all(PHONE_RE.fullmatch(value) for value in non_empty):
        return "phone"
    if all(INTEGER_RE.fullmatch(value) for value in non_empty):
        if any(token in lowered_name for token in ("code", "postal", "iban", "bic", "id")):
            return "identifier"
        return "integer"
    if all(NUMBER_RE.fullmatch(value) for value in non_empty):
        return "number"
    return declared_type if declared_type.casefold() not in {"string", "str"} else "text"


def physical_type(dtype: pl.DataType) -> str:
    return str(dtype)

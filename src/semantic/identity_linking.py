"""Confidence-weighted matching of related profile records."""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Mapping, Sequence


IDENTITY_TOKEN_FIELD = "_osiris_identity_tokens"
DOCUMENT_LOOKUP_FIELD = "_osiris_document_lookup"
_INTERNAL_FIELDS = {
    IDENTITY_TOKEN_FIELD,
    DOCUMENT_LOOKUP_FIELD,
    "_row_id",
    "_source",
    "_source_file",
    "_osiris_shard",
}
_IGNORED_FIELD_PARTS = (
    "created",
    "updated",
    "modified",
    "status",
    "situation",
    "gender",
    "genre",
    "validated",
    "a_valider",
)
_STOP_VALUES = {"", "0", "false", "none", "null", "nan", "n/a", "unknown"}
_FIELD_WEIGHTS = {
    "email": 0.99,
    "phone": 0.97,
    "birth_date": 0.94,
    "identifier": 0.98,
    "surname": 0.72,
    "first_name": 0.62,
    "full_name": 0.78,
    "address": 0.48,
    "postal_code": 0.32,
    "city": 0.20,
    "generic": 0.08,
}
_TRUSTED_IDENTIFIER_FIELDS = {
    "uuid",
    "nationalid",
    "nationalidnumber",
    "socialsecurity",
    "socialsecuritynumber",
    "passport",
    "passportnumber",
    "taxid",
    "taxidnumber",
    "ssn",
    "ssnnumber",
}
_CONFLICT_PENALTIES = {
    "email": 0.65,
    "phone": 0.60,
    "birth_date": 0.50,
    "identifier": 0.85,
    "surname": 0.42,
    "first_name": 0.25,
    "full_name": 0.38,
    "address": 0.18,
    "postal_code": 0.12,
    "city": 0.08,
    "generic": 0.02,
}
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)
_FIELD_NAME_RE = re.compile(r"[\W_]+", re.UNICODE)


def _ascii_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _field_category(field_name: str) -> str | None:
    name = _FIELD_NAME_RE.sub("", _ascii_name(field_name))
    if not name or field_name.startswith("_") or field_name in _INTERNAL_FIELDS:
        return None
    if any(part in name for part in _IGNORED_FIELD_PARTS):
        return None
    if any(part in name for part in ("email", "e_mail", "mail")):
        return "email"
    if any(part in name for part in ("phone", "telephone", "mobile", "tel", "portable")):
        return "phone"
    if any(part in name for part in ("surname", "lastname", "familyname", "nomdefamille", "nomfamille")):
        return "surname"
    if "naissance" in name and any(part in name for part in ("nom", "surname", "family")):
        return "surname"
    if any(part in name for part in ("birth", "dob", "naissance")):
        return "birth_date"
    if any(part in name for part in ("firstname", "givenname", "forename", "prenom")):
        return "first_name"
    if any(part in name for part in ("fullname", "nomcomplet")):
        return "full_name"
    if any(part in name for part in ("address", "adresse", "street", "rue")):
        return "address"
    if any(part in name for part in ("postal", "postcode", "zipcode", "codepostal")):
        return "postal_code"
    if name in {"city", "ville", "commune", "locality"}:
        return "city"
    if any(
        part in name
        for part in (
            "identifier",
            "identifiant",
            "uuid",
            "allocataire",
            "numero",
            "number",
            "reference",
            "passport",
            "nationalid",
            "socialsecurity",
            "membership",
            "dossier",
        )
    ):
        return f"identifier:{name}"
    if name.startswith("id") or name.endswith("id"):
        return f"identifier:{name}"
    if name in {"nom", "name", "family", "lastname"}:
        return "surname"
    if name in {"prenom", "firstname", "given"}:
        return "first_name"
    return f"generic:{name}"


def _base_category(category: str) -> str:
    return category.split(":", maxsplit=1)[0]


def _scalar_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool, date, datetime)):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _scalar_values(nested)]
    return []


def _normalize_value(category: str, value: object) -> str:
    text = str(value).strip()
    base = _base_category(category)
    if base == "phone":
        normalized = "".join(character for character in text if character.isdigit())
    elif base in {"email", "identifier"}:
        normalized = _ascii_name(text).replace(" ", "")
    elif base in {"surname", "first_name", "full_name", "address", "city"}:
        normalized = " ".join(_TOKEN_RE.sub(" ", _ascii_name(text)).split())
    else:
        normalized = " ".join(_ascii_name(text).split())
    return "" if normalized in _STOP_VALUES else normalized


def _profile_values(profile: Mapping[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for field_name, raw_value in profile.items():
        category = _field_category(str(field_name))
        if category is None:
            continue
        for value in _scalar_values(raw_value):
            normalized = _normalize_value(category, value)
            if normalized:
                values[category].add(normalized)
    return values


def identity_tokens(profile: Mapping[str, Any]) -> list[str]:
    """Create selective opaque tokens for exact Qdrant candidate lookup."""

    tokens = set()
    values_by_category = _profile_values(profile)

    def add_token(fields: Sequence[tuple[str, str]]) -> None:
        token_value = "\1".join(
            f"{category}\0{value}" for category, value in sorted(fields)
        )
        tokens.add(hashlib.sha256(token_value.encode("utf-8")).hexdigest())

    for category, values in values_by_category.items():
        if (
            category.startswith("identifier:")
            and category.split(":", maxsplit=1)[1] in _TRUSTED_IDENTIFIER_FIELDS
        ):
            for value in values:
                add_token([(category, value)])

    evidence_sets = (
        ("email", "full_name"),
        ("email", "first_name", "surname"),
        ("birth_date", "full_name"),
        ("birth_date", "first_name", "surname"),
        ("phone", "birth_date", "full_name"),
        ("phone", "birth_date", "first_name", "surname"),
    )
    for categories in evidence_sets:
        if not all(category in values_by_category for category in categories):
            continue
        for field_values in itertools.product(
            *(sorted(values_by_category[category]) for category in categories)
        ):
            add_token(list(zip(categories, field_values, strict=True)))
    return sorted(tokens)


def document_lookup_token(source_file: object, row_id: object, shard: object) -> str:
    shard_name = str(shard or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    key = "\0".join((str(source_file or ""), str(row_id), shard_name))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def identity_confidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return a conservative 0..1 match score; weak evidence alone never links."""

    left_values = _profile_values(left)
    right_values = _profile_values(right)
    comparable = {
        category
        for category in left_values.keys() & right_values.keys()
        if _base_category(category) != "generic"
    }
    if not comparable:
        return 0.0

    matched = [
        category
        for category in comparable
        if left_values[category] & right_values[category]
    ]
    matched_categories = {_base_category(category) for category in matched}
    exact_identity_fields = {"first_name", "surname", "full_name", "birth_date"}
    if any(
        _base_category(category) in exact_identity_fields
        and not left_values[category] & right_values[category]
        for category in comparable
    ):
        return 0.0

    matched_identifier_names = {
        category.split(":", maxsplit=1)[1]
        for category in matched
        if category.startswith("identifier:")
    }
    has_trusted_identifier = bool(
        matched_identifier_names & _TRUSTED_IDENTIFIER_FIELDS
    )
    has_full_name = "full_name" in matched_categories
    has_first_and_surname = {"first_name", "surname"} <= matched_categories

    if has_trusted_identifier:
        pass
    elif "email" in matched_categories:
        if not (has_full_name or has_first_and_surname):
            return 0.0
    elif "phone" in matched_categories:
        if not (
            (has_full_name and "birth_date" in matched_categories)
            or (has_first_and_surname and "birth_date" in matched_categories)
        ):
            return 0.0
    elif not (
        (has_full_name and "birth_date" in matched_categories)
        or (has_first_and_surname and "birth_date" in matched_categories)
    ):
        return 0.0

    evidence = 1.0
    for category in matched:
        evidence *= 1.0 - _FIELD_WEIGHTS.get(_base_category(category), 0.0)
    confidence = 1.0 - evidence
    for category in comparable - set(matched):
        confidence *= 1.0 - _CONFLICT_PENALTIES.get(_base_category(category), 0.0)
    return max(0.0, min(1.0, confidence))


def related_profile_groups(
    seed_hits: Sequence[Mapping[str, Any]],
    related_hits: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[tuple[Mapping[str, Any], list[tuple[Mapping[str, Any], float]]]]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("identity threshold doit être supérieur à 0 et inférieur ou égal à 1")

    def key(hit: Mapping[str, Any]) -> str:
        if hit.get("id") is not None:
            return str(hit["id"])
        payload = hit.get("payload") or {}
        return "|".join(
            str(payload.get(field, ""))
            for field in ("_source_file", "_row_id", "_osiris_shard")
        )

    candidates: dict[str, Mapping[str, Any]] = {}
    for hit in [*seed_hits, *related_hits]:
        candidates.setdefault(key(hit), hit)

    claimed: set[str] = set()
    groups = []
    for seed in seed_hits:
        seed_key = key(seed)
        if seed_key in claimed:
            continue
        seed_payload = seed.get("payload") or {}
        linked = []
        for candidate_key, candidate in candidates.items():
            if candidate_key == seed_key or candidate_key in claimed:
                continue
            confidence = identity_confidence(seed_payload, candidate.get("payload") or {})
            if confidence >= threshold:
                linked.append((candidate, confidence))
        linked.sort(key=lambda item: (-item[1], key(item[0])))
        claimed.add(seed_key)
        claimed.update(key(hit) for hit, _ in linked)
        groups.append((seed, linked))
    return groups


def profile_embedding_chunks(
    profiles: Sequence[Mapping[str, Any]],
    max_characters: int = 1_800,
) -> list[str]:
    """Format every distinct profile value into bounded texts for embedding."""

    if max_characters <= 0:
        raise ValueError("max_characters doit être strictement positif")
    lines = set()
    for profile in profiles:
        for field_name, raw_value in profile.items():
            if field_name in _INTERNAL_FIELDS or str(field_name).startswith("_"):
                continue
            for value in _scalar_values(raw_value):
                rendered = str(value).strip()
                if rendered and rendered.casefold() not in _STOP_VALUES:
                    lines.add(f"{field_name}: {rendered}")

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in sorted(lines):
        while len(line) > max_characters:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            chunks.append(line[:max_characters])
            line = line[max_characters:]
        line_length = len(line) + (1 if current else 0)
        if current and current_length + line_length > max_characters:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
            line_length = len(line)
        current.append(line)
        current_length += line_length
    if current:
        chunks.append("\n".join(current))
    return chunks


def combine_profiles(profiles: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    combined: dict[str, list[Any]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for profile in profiles:
        for field_name, raw_value in profile.items():
            if field_name in _INTERNAL_FIELDS or str(field_name).startswith("_"):
                continue
            for value in _scalar_values(raw_value):
                marker = repr(value)
                if marker not in seen[field_name]:
                    seen[field_name].add(marker)
                    combined[field_name].append(value)
    return dict(combined)

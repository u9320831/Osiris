"""Repair JSON Lines exports and validate the result with Polars."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from src.ingestion.cache import scan_ndjson_as_strings


LOGGER = logging.getLogger(__name__)
HTTP_ERROR_CODE_RE = re.compile(r"\b5\d{2}\b")


def _is_http_error_page(line: str) -> bool:
    lowered = line.casefold()
    return (
        lowered.lstrip().startswith(("<html", "<!doctype html"))
        and HTTP_ERROR_CODE_RE.search(lowered) is not None
    )


def _validate_with_polars(path: Path, batch_size: int) -> int:
    query = scan_ndjson_as_strings(path, infer_schema_length=1_000)
    query.collect_schema()
    rows = 0
    for frame in query.collect_batches(chunk_size=batch_size):
        rows += frame.height
    return rows


def repair_json_lines(
    source: Path,
    output: Path,
    quarantine: Path,
    report: Path,
    *,
    batch_size: int = 262_144,
    checkpoint_every: int = 1_000_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    quarantine = quarantine.expanduser().resolve()
    report = report.expanduser().resolve()
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")
    output_partial = output.with_suffix(output.suffix + ".partial")
    quarantine_partial = quarantine.with_suffix(quarantine.suffix + ".partial")
    report_partial = report.with_suffix(report.suffix + ".partial")

    if not source.is_file():
        raise FileNotFoundError(f"Fichier source introuvable : {source}")
    destinations = (
        output,
        quarantine,
        report,
        checkpoint,
        output_partial,
        quarantine_partial,
        report_partial,
    )
    if len(set(destinations)) != len(destinations):
        raise ValueError("Les chemins source, résultat, quarantaine, rapport et reprise doivent être distincts")
    if source in destinations:
        raise ValueError("Le fichier source ne peut pas être remplacé par un fichier de sortie")
    if batch_size <= 0 or checkpoint_every <= 0:
        raise ValueError("batch_size et checkpoint_every doivent être strictement positifs")

    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)

    source_stat = source.stat()
    source_identity = {
        "path": str(source),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }
    output_paths = {
        "output": str(output),
        "quarantine": str(quarantine),
        "report": str(report),
    }
    state: dict[str, Any]

    def save_checkpoint() -> None:
        checkpoint_temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        with checkpoint_temp.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(checkpoint_temp, checkpoint)

    def reset_resume_files() -> None:
        for path in (checkpoint, output_partial, quarantine_partial, report_partial):
            path.unlink(missing_ok=True)

    if checkpoint.exists() and overwrite:
        reset_resume_files()

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("source") != source_identity or state.get("destinations") != output_paths:
            raise ValueError(
                f"Le checkpoint {checkpoint} appartient à une autre source ou à d'autres sorties"
            )
        if state.get("version") != 1:
            raise ValueError(f"Version de checkpoint inconnue dans {checkpoint}")
        if state.get("phase") == "complete":
            if not report.is_file():
                raise FileNotFoundError(f"Rapport absent malgré un checkpoint terminé : {report}")
            return json.loads(report.read_text(encoding="utf-8"))
        if state.get("phase") not in {"processing", "validated"}:
            raise ValueError(f"État de reprise inconnu dans {checkpoint}")
        LOGGER.info(
            "Reprise depuis la ligne %s, enregistrement %s",
            state["input_lines"] + 1,
            state["output_records"],
        )
    else:
        existing_outputs = [path for path in (output, quarantine, report) if path.exists()]
        existing_partials = [
            path for path in (output_partial, quarantine_partial, report_partial) if path.exists()
        ]
        if (existing_outputs or existing_partials) and not overwrite:
            paths = ", ".join(str(path) for path in (*existing_outputs, *existing_partials))
            raise FileExistsError(
                f"Sorties ou fichiers partiels déjà présents ({paths}); "
                "reprenez avec le checkpoint ou utilisez --overwrite pour recommencer"
            )
        if overwrite:
            reset_resume_files()
        state = {
            "version": 1,
            "phase": "processing",
            "source": source_identity,
            "destinations": output_paths,
            "input_offset": 0,
            "input_lines": 0,
            "output_records": 0,
            "quarantined_fragments": 0,
            "skipped_http_error_pages": 0,
            "output_bytes": 0,
            "quarantine_bytes": 0,
            "inside_http_error_page": False,
        }
        for path in (output_partial, quarantine_partial):
            with path.open("wb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        save_checkpoint()

    if state["phase"] == "processing":
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
        offset = int(state["input_offset"])
        if offset < 0 or offset > source_stat.st_size:
            raise ValueError(f"Offset invalide dans le checkpoint : {offset}")
        if not output_partial.is_file() or not quarantine_partial.is_file():
            raise FileNotFoundError(
                "Fichier partiel de reprise absent; conservez les fichiers présents "
                "et utilisez --overwrite pour recommencer."
            )
        output_bytes = int(state["output_bytes"])
        quarantine_bytes = int(state["quarantine_bytes"])
        if (
            output_partial.stat().st_size < output_bytes
            or quarantine_partial.stat().st_size < quarantine_bytes
        ):
            raise RuntimeError(
                "Un fichier partiel est plus court que le checkpoint; "
                "les données ne peuvent pas être reprises de façon sûre."
            )

        decoder = json.JSONDecoder()
        since_checkpoint = 0
        with source.open("rb") as source_stream:
            source_stream.seek(offset)
            with (
                output_partial.open("r+b") as output_stream,
                quarantine_partial.open("r+b") as quarantine_stream,
            ):
                output_stream.truncate(output_bytes)
                quarantine_stream.truncate(quarantine_bytes)
                output_stream.seek(output_bytes)
                quarantine_stream.seek(quarantine_bytes)

                while raw_line := source_stream.readline():
                    line_number = int(state["input_lines"]) + 1
                    try:
                        line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            f"Encodage UTF-8 invalide dans {source}, ligne {line_number}, "
                            f"octet {error.start}"
                        ) from error

                    lowered_line = line.casefold()
                    if state["inside_http_error_page"]:
                        if "</html>" in lowered_line:
                            state["inside_http_error_page"] = False
                    else:
                        try:
                            whole_record = json.loads(line)
                        except json.JSONDecodeError:
                            whole_record = None

                        if isinstance(whole_record, dict):
                            output_stream.write(raw_line.removeprefix(b"\xef\xbb\xbf").rstrip(b"\r\n"))
                            output_stream.write(b"\n")
                            state["output_records"] += 1
                            state["input_lines"] = line_number
                            state["input_offset"] = source_stream.tell()
                            since_checkpoint += 1
                            if since_checkpoint >= checkpoint_every:
                                output_stream.flush()
                                quarantine_stream.flush()
                                os.fsync(output_stream.fileno())
                                os.fsync(quarantine_stream.fileno())
                                state["output_bytes"] = output_stream.tell()
                                state["quarantine_bytes"] = quarantine_stream.tell()
                                save_checkpoint()
                                LOGGER.info(
                                    "Checkpoint ligne=%s enregistrements=%s",
                                    state["input_lines"],
                                    state["output_records"],
                                )
                                since_checkpoint = 0
                            continue

                        position = 0
                        while True:
                            while position < len(line) and line[position].isspace():
                                position += 1
                            if position == len(line):
                                break

                            remaining = line[position:]
                            if _is_http_error_page(remaining):
                                state["skipped_http_error_pages"] += 1
                                state["inside_http_error_page"] = "</html>" not in remaining.casefold()
                                break

                            try:
                                value, end = decoder.raw_decode(line, position)
                            except json.JSONDecodeError as error:
                                quarantine_record = {
                                    "source_line": line_number,
                                    "character": position + error.pos,
                                    "error": error.msg,
                                    "raw": line[position:].rstrip("\r\n"),
                                }
                                quarantine_stream.write(
                                    (json.dumps(quarantine_record, ensure_ascii=False) + "\n").encode("utf-8")
                                )
                                state["quarantined_fragments"] += 1
                                break

                            if not isinstance(value, dict):
                                quarantine_record = {
                                    "source_line": line_number,
                                    "character": position,
                                    "error": "L'enregistrement JSON n'est pas un objet",
                                    "raw": line[position:end],
                                }
                                quarantine_stream.write(
                                    (json.dumps(quarantine_record, ensure_ascii=False) + "\n").encode("utf-8")
                                )
                                state["quarantined_fragments"] += 1
                                break

                            output_stream.write(line[position:end].encode("utf-8"))
                            output_stream.write(b"\n")
                            state["output_records"] += 1
                            position = end

                    state["input_lines"] = line_number
                    state["input_offset"] = source_stream.tell()
                    since_checkpoint += 1
                    if since_checkpoint >= checkpoint_every:
                        output_stream.flush()
                        quarantine_stream.flush()
                        os.fsync(output_stream.fileno())
                        os.fsync(quarantine_stream.fileno())
                        state["output_bytes"] = output_stream.tell()
                        state["quarantine_bytes"] = quarantine_stream.tell()
                        save_checkpoint()
                        LOGGER.info(
                            "Checkpoint ligne=%s enregistrements=%s",
                            state["input_lines"],
                            state["output_records"],
                        )
                        since_checkpoint = 0

                if state["inside_http_error_page"]:
                    output_stream.flush()
                    quarantine_stream.flush()
                    os.fsync(output_stream.fileno())
                    os.fsync(quarantine_stream.fileno())
                    state["output_bytes"] = output_stream.tell()
                    state["quarantine_bytes"] = quarantine_stream.tell()
                    save_checkpoint()
                    raise ValueError(f"Page HTML d'erreur HTTP non terminée dans {source}")
                if state["output_records"] == 0:
                    raise ValueError(f"Aucun objet JSON valide trouvé dans {source}")

                output_stream.flush()
                quarantine_stream.flush()
                os.fsync(output_stream.fileno())
                os.fsync(quarantine_stream.fileno())
                state["output_bytes"] = output_stream.tell()
                state["quarantine_bytes"] = quarantine_stream.tell()
                save_checkpoint()

        final_source_stat = source.stat()
        if (
            final_source_stat.st_size != source_identity["size"]
            or final_source_stat.st_mtime_ns != source_identity["mtime_ns"]
        ):
            raise RuntimeError(f"Le fichier source a changé pendant son traitement : {source}")
        validated_rows = _validate_with_polars(output_partial, batch_size)
        if validated_rows != state["output_records"]:
            raise RuntimeError(
                f"Validation Polars incohérente : {validated_rows} lignes lues, "
                f"{state['output_records']} objets écrits"
            )
        state["validated_rows"] = validated_rows
        state["phase"] = "validated"
        save_checkpoint()
    else:
        validated_rows = int(state["validated_rows"])

    result = {
        "source": str(source),
        "output": str(output),
        "quarantine": str(quarantine),
        "report": str(report),
        "checkpoint": str(checkpoint),
        "input_lines": int(state["input_lines"]),
        "valid_records": int(state["output_records"]),
        "skipped_http_error_pages": int(state["skipped_http_error_pages"]),
        "quarantined_fragments": int(state["quarantined_fragments"]),
        "polars_validated_rows": validated_rows,
    }
    with report_partial.open("w", encoding="utf-8") as report_stream:
        json.dump(result, report_stream, ensure_ascii=False, indent=2)
        report_stream.flush()
        os.fsync(report_stream.fileno())

    if output_partial.exists():
        os.replace(output_partial, output)
    elif not output.is_file():
        raise FileNotFoundError(f"Résultat partiel ou publié absent : {output_partial}")
    if quarantine_partial.exists():
        os.replace(quarantine_partial, quarantine)
    elif not quarantine.is_file():
        raise FileNotFoundError(f"Quarantaine partielle ou publiée absente : {quarantine_partial}")
    os.replace(report_partial, report)

    state["phase"] = "complete"
    save_checkpoint()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Nettoie un export JSON Lines volumineux sans modifier la source : "
            "sépare les objets collés, ignore les pages HTTP 5xx et met les fragments invalides en quarantaine."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Fichier JSON/NDJSON original")
    parser.add_argument("--output", type=Path, required=True, help="Nouveau fichier NDJSON nettoyé")
    parser.add_argument(
        "--quarantine",
        type=Path,
        help="Fichier JSONL contenant les fragments invalides (défaut : <output>.invalid.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Rapport de nettoyage (défaut : <output>.report.json)",
    )
    parser.add_argument("--batch-size", type=int, default=262_144, help="Taille des lots de validation Polars")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        help="Nombre de lignes source entre deux checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recommencer et remplacer les sorties existantes (efface le checkpoint OSIRIS)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    output = args.output
    quarantine = args.quarantine or output.with_suffix(output.suffix + ".invalid.jsonl")
    report = args.report or output.with_suffix(output.suffix + ".report.json")
    result = repair_json_lines(
        args.source,
        output,
        quarantine,
        report,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
    )

    print(f"Fichier nettoyé : {result['output']}")
    print(f"Enregistrements JSON valides : {result['valid_records']:,}")
    print(f"Pages HTTP 5xx ignorées : {result['skipped_http_error_pages']:,}")
    print(f"Fragments mis en quarantaine : {result['quarantined_fragments']:,}")
    print(f"Lignes validées par Polars : {result['polars_validated_rows']:,}")
    print(f"Quarantaine : {result['quarantine']}")
    print(f"Rapport : {result['report']}")
    print(f"Checkpoint conservé : {result['checkpoint']}")


if __name__ == "__main__":
    main()

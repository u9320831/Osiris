"""Stream parquet cache rows through Kafka efficiently.

Install project dependencies with::

    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson
import polars as pl


def _kafka_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from confluent_kafka import Consumer, KafkaError, Producer
        from confluent_kafka.admin import AdminClient, NewTopic
    except ImportError as error:
        raise RuntimeError(
            "Le client Kafka est requis : pip install -r requirements.txt"
        ) from error
    return Producer, Consumer, KafkaError, AdminClient, NewTopic


def _json_default(value: Any) -> str:
    return str(value)


def ensure_topic(
    brokers: str,
    topic: str,
    partitions: int = 3,
    replication_factor: int = 1,
) -> bool:
    """Create a topic if it does not exist; return whether it was created."""

    _, _, KafkaError, AdminClient, NewTopic = _kafka_imports()
    if partitions <= 0 or replication_factor <= 0:
        raise ValueError("partitions et replication_factor doivent être positifs")

    admin = AdminClient({"bootstrap.servers": brokers})
    metadata = admin.list_topics(topic=topic, timeout=10)
    if topic in metadata.topics:
        return False

    future = admin.create_topics(
        [NewTopic(topic, num_partitions=partitions, replication_factor=replication_factor)]
    )[topic]
    try:
        future.result(10)
    except Exception as error:
        # A concurrent producer may have created it between metadata and create.
        if hasattr(error, "code") and error.code() == KafkaError._TOPIC_ALREADY_EXISTS:
            return False
        raise
    return True


class KafkaTransfer:
    """Produce cache rows and consume them with bounded memory."""

    def __init__(
        self,
        brokers: str = "localhost:9092",
        topic: str = "osiris.profile",
    ) -> None:
        self.brokers = brokers
        self.topic = topic

    def produce_cache(
        self,
        cache: Path | str,
        *,
        batch_size: int = 1_000,
        max_rows: int | None = None,
        key_column: str | None = None,
        partitions: int = 3,
    ) -> dict[str, int]:
        """Publish rows from a parquet cache without loading it all in memory."""

        Producer, _, _, _, _ = _kafka_imports()
        cache = Path(cache)
        if not cache.is_file():
            raise FileNotFoundError(f"Cache parquet introuvable : {cache}")
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows doit être positif ou None")

        ensure_topic(self.brokers, self.topic, partitions=partitions)
        query = pl.scan_parquet(cache)
        schema = query.collect_schema()
        if key_column is not None and key_column not in schema:
            raise ValueError(f"Colonne de clé absente du cache : {key_column}")

        delivery_errors: list[str] = []

        def on_delivery(error: Any, message: Any) -> None:
            if error is not None:
                delivery_errors.append(str(error))

        producer = Producer(
            {
                "bootstrap.servers": self.brokers,
                "client.id": "osiris-cache-producer",
                "compression.type": "lz4",
                "linger.ms": 10,
                "batch.num.messages": batch_size,
                "queue.buffering.max.messages": max(batch_size * 10, 10_000),
            }
        )
        rows_sent = 0
        bytes_sent = 0
        try:
            for frame in query.collect_batches(chunk_size=batch_size):
                if max_rows is not None:
                    remaining = max_rows - rows_sent
                    if remaining <= 0:
                        break
                    frame = frame.head(remaining)

                for row in frame.iter_rows(named=True):
                    row_id = rows_sent
                    payload = orjson.dumps(row, default=_json_default)
                    key_value = row.get(key_column) if key_column else row_id
                    key = None if key_value is None else str(key_value).encode("utf-8")

                    while True:
                        try:
                            producer.produce(
                                self.topic,
                                key=key,
                                value=payload,
                                on_delivery=on_delivery,
                            )
                            break
                        except BufferError:
                            producer.poll(0.1)

                    rows_sent += 1
                    bytes_sent += len(payload)
                    producer.poll(0)

                if max_rows is not None and rows_sent >= max_rows:
                    break
        finally:
            remaining = producer.flush(60)
            if remaining:
                raise TimeoutError(f"{remaining} messages Kafka non livrés")

        if delivery_errors:
            raise RuntimeError(f"Erreur de livraison Kafka : {delivery_errors[0]}")
        return {"rows": rows_sent, "bytes": bytes_sent}

    def consume_to_ndjson(
        self,
        output: Path | str,
        *,
        group: str = "osiris-cache-consumer",
        max_messages: int | None = None,
        idle_timeout: float = 10.0,
        offset: str = "earliest",
    ) -> dict[str, int]:
        """Consume Kafka messages and write compact JSON lines to disk."""

        _, Consumer, KafkaError, _, _ = _kafka_imports()
        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages doit être positif ou None")

        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        consumer = Consumer(
            {
                "bootstrap.servers": self.brokers,
                "group.id": group,
                "auto.offset.reset": offset,
                "enable.auto.commit": False,
                "client.id": "osiris-cache-consumer",
            }
        )
        messages = 0
        bytes_received = 0
        consumer.subscribe([self.topic])
        try:
            with destination.open("wb") as stream:
                while max_messages is None or messages < max_messages:
                    message = consumer.poll(idle_timeout)
                    if message is None:
                        if idle_timeout > 0:
                            break
                        continue
                    if message.error():
                        if message.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        raise RuntimeError(str(message.error()))

                    payload = message.value()
                    # Validate the payload while keeping the original compact
                    # bytes, so the consumer does not re-serialize every row.
                    orjson.loads(payload)
                    stream.write(payload + b"\n")
                    messages += 1
                    bytes_received += len(payload)
                    if messages % 1_000 == 0:
                        consumer.commit(asynchronous=False)
                consumer.commit(asynchronous=False)
        finally:
            consumer.close()
        return {"messages": messages, "bytes": bytes_received}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfert du cache OSIRIS via Kafka")
    subparsers = parser.add_subparsers(dest="command", required=True)

    topic_parser = subparsers.add_parser("topic", help="Crée le topic s'il n'existe pas")
    topic_parser.add_argument("--brokers", default="localhost:9092")
    topic_parser.add_argument("--topic", default="osiris.profile")
    topic_parser.add_argument("--partitions", type=int, default=3)

    produce_parser = subparsers.add_parser("produce", help="Publie le cache")
    produce_parser.add_argument("--cache", type=Path, default=Path("cache/profile.parquet"))
    produce_parser.add_argument("--brokers", default="localhost:9092")
    produce_parser.add_argument("--topic", default="osiris.profile")
    produce_parser.add_argument("--batch-size", type=int, default=1_000)
    produce_parser.add_argument("--max-rows", type=int)
    produce_parser.add_argument("--key-column")
    produce_parser.add_argument("--partitions", type=int, default=3)

    consume_parser = subparsers.add_parser("consume", help="Consomme vers un fichier NDJSON")
    consume_parser.add_argument("--output", type=Path, default=Path("cache/received.ndjson"))
    consume_parser.add_argument("--brokers", default="localhost:9092")
    consume_parser.add_argument("--topic", default="osiris.profile")
    consume_parser.add_argument("--group", default="osiris-cache-consumer")
    consume_parser.add_argument("--max-messages", type=int)
    consume_parser.add_argument("--idle-timeout", type=float, default=10.0)
    consume_parser.add_argument("--offset", choices=("earliest", "latest"), default="earliest")

    args = parser.parse_args()
    if args.command == "topic":
        created = ensure_topic(args.brokers, args.topic, args.partitions)
        print(f"Topic {args.topic}: {'créé' if created else 'déjà présent'}")
    elif args.command == "produce":
        transfer = KafkaTransfer(args.brokers, args.topic)
        print(transfer.produce_cache(
            args.cache,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            key_column=args.key_column,
            partitions=args.partitions,
        ))
    else:
        transfer = KafkaTransfer(args.brokers, args.topic)
        print(transfer.consume_to_ndjson(
            args.output,
            group=args.group,
            max_messages=args.max_messages,
            idle_timeout=args.idle_timeout,
            offset=args.offset,
        ))


if __name__ == "__main__":
    main()

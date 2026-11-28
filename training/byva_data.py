"""Deterministic corpus cleaning, dedup, and sharding for BYVA pretraining data.

All operations are pure and order-independent: given the same set of documents,
the pipeline produces byte-identical shards and manifest regardless of input
ordering, filesystem timestamps, or machine.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MIN_CHARS = 40
DEFAULT_MAX_CHARS = 100_000


class DataError(ValueError):
    """Raised for malformed input documents."""


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShardConfig:
    docs_per_shard: int = 1_000
    min_chars: int = DEFAULT_MIN_CHARS
    max_chars: int = DEFAULT_MAX_CHARS


def load_documents(path: Path, progress_every: int | None = None) -> list[str]:
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise DataError(f"{path}:{line_number}: record must be an object with a text field")
            documents.append(record["text"])
            if progress_every is not None and len(documents) % progress_every == 0:
                print(f"  {path.name}: {len(documents)} docs loaded", file=sys.stderr, flush=True)
    return documents


def prepare_documents(
    raw_texts: list[str], config: ShardConfig, progress_every: int | None = None
) -> tuple[list[dict], dict]:
    """Normalize, filter, and deduplicate. Returns (docs, stats).

    Docs are sorted by content hash so output depends only on the document set.
    """
    seen: set[str] = set()
    stats = {
        "input": len(raw_texts),
        "dropped_short": 0,
        "dropped_long": 0,
        "dropped_duplicates": 0,
    }
    kept: list[dict] = []
    for index, text in enumerate(raw_texts):
        normalized = normalize_text(text)
        if len(normalized) < config.min_chars:
            stats["dropped_short"] += 1
        elif len(normalized) > config.max_chars:
            stats["dropped_long"] += 1
        else:
            digest = doc_hash(normalized)
            if digest in seen:
                stats["dropped_duplicates"] += 1
            else:
                seen.add(digest)
                kept.append({"id": digest[:16], "text": normalized})
        if progress_every is not None and (index + 1) % progress_every == 0:
            print(
                f"  prepare: {index + 1}/{len(raw_texts)} docs "
                f"(kept {len(kept)}, dup {stats['dropped_duplicates']})",
                file=sys.stderr,
                flush=True,
            )
    kept.sort(key=lambda doc: doc["id"])
    stats["kept"] = len(kept)
    return kept, stats


def shard_documents(docs: list[dict], config: ShardConfig) -> list[list[dict]]:
    return [docs[start : start + config.docs_per_shard] for start in range(0, len(docs), config.docs_per_shard)]


def build_manifest(
    source_files: list[Path],
    shards: list[tuple[str, bytes]],
    config: ShardConfig,
) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sort_key": "id",
        "normalization": "NFC",
        "config": {
            "docs_per_shard": config.docs_per_shard,
            "min_chars": config.min_chars,
            "max_chars": config.max_chars,
        },
        "sources": [
            {"name": source.name, "sha256": file_sha256(source)}
            for source in sorted(source_files, key=lambda source: source.name)
        ],
        "shards": [
            {"name": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in shards
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_bytes(docs: list[dict]) -> bytes:
    lines = [canonical_json(doc) for doc in docs]
    return ("\n".join(lines) + "\n").encode("utf-8")

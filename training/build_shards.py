"""Build reproducible BYVA pretraining shards from JSONL input.

Usage:
    python3 training/build_shards.py --input data/raw.jsonl [--input more.jsonl ...]
        --output-dir shards/ [--docs-per-shard 1000] [--min-chars 40] [--max-chars 100000]

Writes shard-00000.jsonl ... plus manifest.json into --output-dir.
Corpora and shards stay out of Git (see D-0007); only this script and tiny
fixtures are tracked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from byva_data import (
    ShardConfig,
    build_manifest,
    load_documents,
    prepare_documents,
    shard_bytes,
    shard_documents,
)


PROGRESS_EVERY = 50_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--docs-per-shard", type=int, default=1_000)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=100_000)
    args = parser.parse_args()

    for source in args.input:
        if not source.is_file():
            parser.error(f"input file not found: {source}")

    config = ShardConfig(
        docs_per_shard=args.docs_per_shard,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    raw_texts: list[str] = []
    for source in args.input:
        print(f"loading {source.name} ...", file=sys.stderr, flush=True)
        loaded_before = len(raw_texts)
        raw_texts.extend(load_documents(source, progress_every=PROGRESS_EVERY))
        print(
            f"loaded {len(raw_texts) - loaded_before} docs from {source.name}",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"normalizing, filtering, deduplicating {len(raw_texts)} docs ...",
        file=sys.stderr,
        flush=True,
    )
    docs, stats = prepare_documents(raw_texts, config, progress_every=PROGRESS_EVERY)
    shards = shard_documents(docs, config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, bytes]] = []
    for index, shard in enumerate(shards):
        name = f"shard-{index:05d}.jsonl"
        payload = shard_bytes(shard)
        (output_dir / name).write_bytes(payload)
        written.append((name, payload))
        if (index + 1) % 100 == 0:
            print(
                f"  writing shards: {index + 1}/{len(shards)}",
                file=sys.stderr,
                flush=True,
            )

    manifest = build_manifest(args.input, written, config)
    manifest["stats"] = stats
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        __import__("byva_data").canonical_json(manifest) + "\n", encoding="utf-8"
    )

    print(f"docs kept: {stats['kept']} / {stats['input']}")
    print(f"shards written: {len(shards)} -> {output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

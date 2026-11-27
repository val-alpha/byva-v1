# BYVA Data Pipeline

Deterministic preprocessing for BYVA pretraining data: JSONL input → NFC
normalization → length filter → deduplication → hash-sorted shards + manifest.
Pure Python 3 standard library, no dependencies.

## Usage

```bash
python3 training/build_shards.py \
  --input data/raw-part1.jsonl --input data/raw-part2.jsonl \
  --output-dir shards/ \
  [--docs-per-shard 1000] [--min-chars 40] [--max-chars 100000]
```

Input is JSONL, one `{"text": "..."}` object per line (any extra keys are ignored).
Output is `shard-00000.jsonl ...` plus `manifest.json` in `--output-dir`.

## Determinism guarantees

- Documents are NFC-normalized and stripped, then filtered by normalized length
  (`min_chars`..`max_chars`), deduplicated by SHA-256 of the normalized text, and
  sorted by `id` (first 16 hex chars of the digest).
- Output depends **only on the set of documents**, never on input order or file
  grouping: two runs with permuted inputs produce byte-identical shards and
  manifests (verified in `training/tests/test_pipeline.py`).
- All JSON is canonical (`sort_keys`, compact separators, `ensure_ascii=False`).
- The manifest records `sources[{name, sha256}]` (file name, not absolute path, so
  manifests are portable), per-shard sha256 and byte counts, filter config,
  normalization, sort key, and run stats.

## Tests

```bash
python3 -m unittest discover -s training/tests -v
```

Six tests cover filtering/dedup, order independence, exact-once sharding,
manifest-to-shard checksum agreement, malformed-record rejection, and a frozen
fixture regression (`training/fixtures/smoke/`).

## Policy

- Corpora and shards stay out of Git (D-0007); commit manifests, scripts, and the
  tiny smoke fixture only.
- Dataset admission rules and the verified source table live in `docs/DATA.md`.
- Shard/manifest determinism rules are recorded in D-0012.

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

## Tokenizer training (BPE, D-0015/D-0017)

```bash
python3 training/train_bpe.py --shards shards/ \
  --target-vocab 32768 --version 1 \
  --output tokenizer/byva-tokenizer-32k-v1.json [--stats stats.json]
```

Trains a ranked byte-level BPE merge table directly on shard documents with no
pre-tokenization boundary (matching `Tokenizer::encode_bytes`). Deterministic:
input order never matters; each step merges the most frequent pair with ties
broken by (left, right) ascending. Output is a versioned artifact
`{version, vocab_size, merges}` loadable by `Tokenizer::from_artifact`.
`training/byva_tokenizer.py` is the Python mirror of the Rust tokenizer used
for training-time encoding; a golden fixture
(`crates/tokenizer/tests/golden/bpe_smoke_v1.json`) pins cross-language
equivalence. The exact trainer is memory-bounded; very large corpora need a
scaled trainer (D-0017).

## Colab T4 notebook

`training/colab_t4.ipynb` runs the Phase 3 hardware gate end to end on a Colab
T4: build shards, train the 32,768 BPE table, then a SMALL-config FP16 run
with gradient accumulation and Drive-backed resumable checkpoints
(`train.py --config small --tokenizer ... --device cuda --fp16 --grad-accum`).

## Tests

```bash
python3 -m unittest discover -s training/tests -v
```

Pipeline tests cover filtering/dedup, order independence, exact-once sharding,
manifest-to-shard checksum agreement, malformed-record rejection, and a frozen
fixture regression (`training/fixtures/smoke/`). BPE trainer tests (no PyTorch
required) cover artifact validity, naive-BPE equivalence, permutation
determinism, and the golden fixture regression. `test_train.py` requires
PyTorch and skips cleanly when it is absent.

## Policy

- Corpora and shards stay out of Git (D-0007); commit manifests, scripts, and the
  tiny smoke fixture only.
- Dataset admission rules and the verified source table live in `docs/DATA.md`.
- Shard/manifest determinism rules are recorded in D-0012.

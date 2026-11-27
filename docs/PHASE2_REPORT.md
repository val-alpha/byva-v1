# Phase 2 Gate Report — Verified Data Research and Pipeline

Status: PASSED (with one verification-method limitation recorded below)

## Environment

- Python: python3.12, standard library only (no new dependencies)
- Platform: x86_64-unknown-linux-gnu, Intel Xeon 6985P-C, CPU only
- Date: 2026-11-26

## Gate: license review

Verification method limitation: the Exa connector was not available in this
environment. Licenses were therefore verified against the **primary source**
(Hugging Face dataset API, `https://huggingface.co/api/datasets/{id}`) on
2026-11-26, with the dataset sha captured at retrieval time. Recorded per D-0008
as "Exa verification not run; primary-API verification run instead".

| Dataset | License verified | Dataset sha | Notes |
|---|---|---|---|
| roneneldan/TinyStories | CDLA-Sharing-1.0 | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | Synthetic text (GPT-3.5/4) — capability experiments only |
| allenai/c4 | ODC-BY 1.0 | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | Attribution required; subsets `id`/`en` |
| wikimedia/wikipedia | CC BY-SA 3.0 + GFDL | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | Share-alike; config `20231101.id` |

Full policy and table: `docs/DATA.md`. Decisions: D-0011, D-0012.

## Gate: reproducible shards

### Pipeline implementation

- `training/byva_data.py` — normalization (NFC + strip), length filter, SHA-256
  dedup, hash-sorted docs, sharding, canonical-JSON shard bytes, portable
  manifest builder (`sources` use file names, not absolute paths).
- `training/build_shards.py` — CLI wrapper (`--input`, `--output-dir`,
  `--docs-per-shard`, `--min-chars`, `--max-chars`).

### Checks actually run

| Check | Command | Result |
|---|---|---|
| Unit + fixture tests | `python3 -m unittest discover -s training/tests -v` | PASS — 6 tests, 0 failed |
| CLI reproducibility (two independent runs) | `python3 training/build_shards.py --input training/fixtures/smoke/sample.jsonl --output-dir /tmp/shards-{a,b}` then `diff -r` | PASS — `diff -r` empty; both manifests sha256 `179af5aa69e5c00f0a5361e294585849760fb24662cd8c5f9fcf9797d2c2431e` |
| CLI vs frozen fixture manifest | `--docs-per-shard 4` run compared to `training/fixtures/smoke/expected_manifest.json` | PASS — deep-equal |

### Fixture

`training/fixtures/smoke/` (sha256 of files at commit time):

- `sample.jsonl` — `c8160a69a132a17b8ea1141efb9f61cf59628f6aed5cc610a560eaab392b7983`
  (12 documents: 10 valid id/en, 1 duplicate, 1 below `min_chars`)
- `expected_manifest.json` — `1a14037f11c574624699b7575d2f0824f936d5ca8169643691fb5f9cfc143900`
  (docs_per_shard=4 → 3 shards; stats: kept=10, dropped_short=1, dropped_duplicates=1)

The fixture test rebuilds the pipeline output and compares deep-equality with the
frozen manifest, so any determinism regression fails CI.

### Real-data smoke (not committed, run in /tmp)

Source: 20 rows from TinyStories via HF datasets-server `rows` endpoint
(`offset=0&length=20`, all rows ≥398 chars). Pipeline result: 20/20 kept,
1 shard. Manifest sha256:
`74668927c4c733358c1042b545c56449ad87f0742fa40311df77d0352d2972a3`.
Shards and the sample stay out of Git per D-0007; only the checksum above is
recorded here.

## Not run

- **Exa source verification** — Exa connector unavailable; replaced by
  primary-API verification (see above). Should be revisited if the connector
  becomes available.
- **Full-scale shard builds** for c4/wikipedia — needs corpus download and
  storage outside this environment; deferred to Phase 3 planning.

## CI

`.github/workflows/ci.yml` gains a `python3 -m unittest discover -s
training/tests -v` step (stdlib only, no extra setup) so the determinism tests
run on every push/PR.

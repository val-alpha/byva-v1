"""Determinism and correctness tests for the BYVA data pipeline.

Run with: python3 -m unittest discover -s training/tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from byva_data import (  # noqa: E402
    DataError,
    ShardConfig,
    build_manifest,
    canonical_json,
    doc_hash,
    load_documents,
    prepare_documents,
    shard_bytes,
    shard_documents,
)


def write_jsonl(path: Path, texts: list[str]) -> None:
    lines = [canonical_json({"text": text}) for text in texts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.config = ShardConfig(docs_per_shard=3, min_chars=10, max_chars=200)
        self.texts = [
            "Teks bahasa Indonesia nomor satu untuk pengujian pipeline.",
            "Second English document used to exercise the deterministic order.",
            "Teks bahasa Indonesia nomor satu untuk pengujian pipeline.",
            "Pendek",
            "Third document with enough length to pass the configured filter.",
        ]

    def test_prepare_documents_filters_and_dedups(self):
        docs, stats = prepare_documents(self.texts, self.config)
        self.assertEqual(stats["kept"], 3)
        self.assertEqual(stats["dropped_short"], 1)
        self.assertEqual(stats["dropped_duplicates"], 1)
        self.assertEqual([doc["id"] for doc in docs], sorted(doc["id"] for doc in docs))
        for doc in docs:
            self.assertEqual(doc["id"], doc_hash(doc["text"])[:16])

    def test_output_is_order_independent(self):
        docs_a, _ = prepare_documents(self.texts, self.config)
        docs_b, _ = prepare_documents(list(reversed(self.texts)), self.config)
        self.assertEqual(docs_a, docs_b)
        self.assertEqual(shard_bytes(docs_a), shard_bytes(docs_b))

    def test_sharding_partitions_every_document_exactly_once(self):
        docs, _ = prepare_documents(self.texts, self.config)
        shards = shard_documents(docs, self.config)
        flattened = [doc for shard in shards for doc in shard]
        self.assertEqual(flattened, docs)
        self.assertTrue(all(len(shard) <= self.config.docs_per_shard for shard in shards))

    def test_manifest_matches_written_shards(self):
        docs, _ = prepare_documents(self.texts, self.config)
        shards = shard_documents(docs, self.config)
        written = [(f"shard-{index:05d}.jsonl", shard_bytes(shard)) for index, shard in enumerate(shards)]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw.jsonl"
            write_jsonl(source, self.texts)
            manifest = build_manifest([source], written, self.config)
        self.assertEqual(len(manifest["shards"]), len(written))
        for entry, (name, payload) in zip(manifest["shards"], written):
            self.assertEqual(entry["name"], name)
            import hashlib

            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(entry["bytes"], len(payload))
        self.assertEqual(manifest["schema_version"], 1)

    def test_load_documents_rejects_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"text": "ok"}\n{"no_text_field": true}\n', encoding="utf-8")
            with self.assertRaises(DataError):
                load_documents(path)

    def test_fixture_pipeline_is_reproducible(self):
        fixture = TRAINING_DIR / "fixtures" / "smoke" / "sample.jsonl"
        expected = TRAINING_DIR / "fixtures" / "smoke" / "expected_manifest.json"
        if not fixture.is_file() or not expected.is_file():
            self.fail("smoke fixture files are missing")
        docs, stats = prepare_documents(load_documents(fixture), ShardConfig(docs_per_shard=4))
        shards = shard_documents(docs, ShardConfig(docs_per_shard=4))
        written = [(f"shard-{index:05d}.jsonl", shard_bytes(shard)) for index, shard in enumerate(shards)]
        manifest = build_manifest([fixture], written, ShardConfig(docs_per_shard=4))
        manifest["stats"] = stats
        expected_manifest = __import__("json").loads(expected.read_text(encoding="utf-8"))
        actual = canonical_json(manifest)
        self.assertEqual(actual, canonical_json(expected_manifest))


if __name__ == "__main__":
    unittest.main()

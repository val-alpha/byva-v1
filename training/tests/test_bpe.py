"""Tests for the deterministic BPE trainer.

Run with: python3 -m unittest discover -s training/tests
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from train_bpe import (  # noqa: E402
    BpeEncoder,
    BpeError,
    BpeTrainer,
    BYTE_TOKEN_BASE,
    FIRST_MERGE_ID,
    canonical_artifact_json,
    target_merge_count,
    validate_vocab_size,
)


FIXTURE_CORPUS = [
    "kuda hitam kecil berlari di padang rumput.",
    "kuda hitam berlari cepat dan kuda kecil berlari pelan.",
    "the small black horse runs across the green field.",
    "the black horse runs fast and the small horse runs far.",
    "kuda kecil melihat kuda hitam dan kuda berlari ke padang.",
]
FIXTURE_TARGET_MERGES = 24


def train_fixture() -> BpeTrainer:
    trainer = BpeTrainer(target_merges=FIXTURE_TARGET_MERGES)
    for text in FIXTURE_CORPUS:
        trainer.add_text(text)
    trainer.prepare()
    trainer.train()
    return trainer


class BpeTrainerTest(unittest.TestCase):
    def test_artifact_shape_matches_rust_tokenizer_artifact(self):
        trainer = train_fixture()
        vocab_size = FIRST_MERGE_ID + FIXTURE_TARGET_MERGES
        artifact = trainer.artifact(vocab_size)
        self.assertEqual(sorted(artifact), ["merges", "version", "vocab_size"])
        self.assertEqual(artifact["version"], 1)
        self.assertEqual(artifact["vocab_size"], vocab_size)
        self.assertEqual(len(artifact["merges"]), FIXTURE_TARGET_MERGES)
        for index, merge in enumerate(artifact["merges"]):
            self.assertEqual(sorted(merge), ["left", "right"])
            output_id = FIRST_MERGE_ID + index
            self.assertGreaterEqual(merge["left"], BYTE_TOKEN_BASE)
            self.assertGreaterEqual(merge["right"], BYTE_TOKEN_BASE)
            self.assertLess(merge["left"], output_id)
            self.assertLess(merge["right"], output_id)

    def test_training_is_deterministic_and_order_independent(self):
        first = train_fixture().artifact(FIRST_MERGE_ID + FIXTURE_TARGET_MERGES)
        trainer = BpeTrainer(target_merges=FIXTURE_TARGET_MERGES)
        for text in reversed(FIXTURE_CORPUS):
            trainer.add_text(text)
        trainer.prepare()
        trainer.train()
        second = trainer.artifact(FIRST_MERGE_ID + FIXTURE_TARGET_MERGES)
        self.assertEqual(canonical_artifact_json(first), canonical_artifact_json(second))

    def test_merge_selection_prefers_higher_count_then_lower_ids(self):
        trainer = BpeTrainer(target_merges=2)
        for text in ["ab ab cd cd", "ab cd"]:
            trainer.add_text(text)
        trainer.prepare()
        trainer.train()
        self.assertEqual(trainer.merges[0], (BYTE_TOKEN_BASE + ord("a"), BYTE_TOKEN_BASE + ord("b")))
        self.assertEqual(trainer.merges[1], (BYTE_TOKEN_BASE + ord("c"), BYTE_TOKEN_BASE + ord("d")))

    def test_unreachable_target_raises(self):
        trainer = BpeTrainer(target_merges=10_000)
        trainer.add_text("ab ab")
        trainer.prepare()
        with self.assertRaises(BpeError):
            trainer.train()

    def test_empty_corpus_raises(self):
        trainer = BpeTrainer(target_merges=1)
        with self.assertRaises(BpeError):
            trainer.prepare()

    def test_frozen_fixture_regression(self):
        trainer = train_fixture()
        artifact = trainer.artifact(FIRST_MERGE_ID + FIXTURE_TARGET_MERGES)
        expected_path = TRAINING_DIR / "fixtures" / "bpe" / "expected_mini_artifact.json"
        self.assertTrue(expected_path.is_file(), "frozen BPE fixture is missing")
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(canonical_artifact_json(artifact), json.dumps(expected, sort_keys=True, separators=(",", ":")))


class BpeEncoderTest(unittest.TestCase):
    def setUp(self):
        trainer = train_fixture()
        self.encoder = BpeEncoder(trainer.artifact(FIRST_MERGE_ID + FIXTURE_TARGET_MERGES))

    def test_round_trip_on_multilingual_text(self):
        samples = [
            "kuda hitam kecil berlari.",
            "the small black horse runs fast, far away!",
            "Kécap かな mix 🌏 emoji",
            "angka 12345 dan tanda #?! bersebelahan",
            "literal <|call|> stays content",
        ]
        for text in samples:
            self.assertEqual(self.encoder.decode(self.encoder.encode(text)), text)

    def test_repetitive_text_compresses(self):
        text = "kuda kuda kuda kuda berlari " * 10
        encoded = self.encoder.encode(text)
        self.assertLess(len(encoded), len(text.encode("utf-8")))

    def test_unknown_symbols_fall_back_to_bytes(self):
        text = "異体字"
        encoded = self.encoder.encode(text)
        self.assertTrue(all(token >= BYTE_TOKEN_BASE for token in encoded))
        self.assertEqual(self.encoder.decode(encoded), text)


class CommittedArtifactTest(unittest.TestCase):
    def test_committed_32k_artifact_round_trips(self):
        artifact_path = TRAINING_DIR.parent / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json"
        manifest_path = artifact_path.with_name("byva-tokenizer-v1.manifest.json")
        self.assertTrue(artifact_path.is_file(), "committed BPE artifact is missing")
        self.assertTrue(manifest_path.is_file(), "committed BPE manifest is missing")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["version"], 1)
        self.assertEqual(artifact["vocab_size"], 32_768)
        self.assertEqual(len(artifact["merges"]), 32_768 - FIRST_MERGE_ID)
        encoder = BpeEncoder(artifact)
        for text in [
            "kuda hitam kecil itu berlari.",
            "The small black horse runs across the field.",
            "emoji \U0001F30F dan aksara \u304B\u306A",
        ]:
            self.assertEqual(encoder.decode(encoder.encode(text)), text)


class VocabValidationTest(unittest.TestCase):
    def test_supported_range(self):
        self.assertEqual(validate_vocab_size(32_768), 32_768)
        self.assertEqual(validate_vocab_size(49_152), 49_152)
        with self.assertRaises(BpeError):
            validate_vocab_size(32_000)
        with self.assertRaises(BpeError):
            validate_vocab_size(50_000)

    def test_target_merge_count(self):
        self.assertEqual(target_merge_count(32_768), 32_768 - FIRST_MERGE_ID)


if __name__ == "__main__":
    unittest.main()

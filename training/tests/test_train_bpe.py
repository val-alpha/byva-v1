"""Tests for the deterministic BPE trainer and the tokenizer Python mirror.

Pure standard library; no PyTorch required.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from byva_tokenizer import (  # noqa: E402
    BYTE_TOKEN_BASE,
    FIRST_MERGE_ID,
    TokenizerError,
    decode_content,
    encode_bytes,
    encode_content,
    load_artifact,
    merge_ranks,
    token_bytes,
)
from train_bpe import artifact_stats, load_shard_texts, train_bpe  # noqa: E402


def build_shard_dir(dir_path: Path, texts: list[str]) -> Path:
    from byva_data import (
        ShardConfig,
        build_manifest,
        prepare_documents,
        shard_bytes,
        shard_documents,
    )

    dir_path.mkdir(parents=True, exist_ok=True)
    source = dir_path / "fixture.jsonl"
    source.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n",
        encoding="utf-8",
    )
    config = ShardConfig(docs_per_shard=4, min_chars=1, max_chars=10_000)
    docs, _ = prepare_documents(texts, config)
    shards = []
    for index, group in enumerate(shard_documents(docs, config)):
        name = f"shard-{index:05d}.jsonl"
        payload = shard_bytes(group)
        (dir_path / name).write_bytes(payload)
        shards.append((name, payload))
    manifest = build_manifest([source], shards, config)
    (dir_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return dir_path


def naive_encode(bytes_data: bytes, merges: list[dict[str, int]]) -> list[int]:
    """Brute-force reference: apply the lowest-rank pair one occurrence at a time."""
    symbols = [BYTE_TOKEN_BASE + byte for byte in bytes_data]
    ranks = merge_ranks(merges)
    while True:
        best_rank = None
        best_index = None
        for index in range(len(symbols) - 1):
            rank = ranks.get((symbols[index], symbols[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = index
        if best_rank is None:
            return symbols
        symbols[best_index : best_index + 2] = [best_rank]


CORPUS = [
    "Teks bahasa Indonesia nomor satu untuk pengujian pipeline data BYVA.",
    "Second English document used to exercise the deterministic ordering.",
    "Cerita pendek berbahasa Indonesia tentang kucing yang penasaran.",
    "A tiny story about a brave rabbit who crossed the long wooden bridge.",
    "the quick brown fox jumps over the lazy dog again and again",
    "data deterministik menghasilkan shard yang dapat direproduksi",
] * 8


class TrainBpeTest(unittest.TestCase):
    def test_artifact_shape_and_validation(self):
        artifact = train_bpe(CORPUS, target_vocab=FIRST_MERGE_ID + 40)
        self.assertEqual(artifact["vocab_size"], FIRST_MERGE_ID + 40)
        self.assertEqual(len(artifact["merges"]), 40)
        self.assertEqual(artifact["version"], 1)
        seen = set()
        for index, merge in enumerate(artifact["merges"]):
            output = FIRST_MERGE_ID + index
            self.assertGreaterEqual(merge["left"], BYTE_TOKEN_BASE)
            self.assertGreaterEqual(merge["right"], BYTE_TOKEN_BASE)
            self.assertLess(merge["left"], output)
            self.assertLess(merge["right"], output)
            self.assertNotIn((merge["left"], merge["right"]), seen)
            seen.add((merge["left"], merge["right"]))

    def test_round_trip_matches_naive_bpe(self):
        artifact = train_bpe(CORPUS, target_vocab=FIRST_MERGE_ID + 60)
        ranks = merge_ranks(artifact["merges"])
        for text in CORPUS[:6]:
            expected = naive_encode(text.encode("utf-8"), artifact["merges"])
            self.assertEqual(encode_content(text, ranks), expected)
            decoded = decode_content(expected, artifact["merges"])
            self.assertEqual(decoded, text)

    def test_merge_order_respects_counts(self):
        artifact = train_bpe(["abab" * 8], target_vocab=FIRST_MERGE_ID + 3)
        merges = artifact["merges"]
        byte_a = BYTE_TOKEN_BASE + ord("a")
        byte_b = BYTE_TOKEN_BASE + ord("b")
        self.assertEqual(merges[0], {"left": byte_a, "right": byte_b})
        merged_ab = FIRST_MERGE_ID
        self.assertEqual(merges[1], {"left": merged_ab, "right": merged_ab})
        merged_cc = FIRST_MERGE_ID + 1
        self.assertEqual(merges[2], {"left": merged_cc, "right": merged_cc})
        self.assertEqual(decode_content([FIRST_MERGE_ID + 2], merges), "abababab")

    def test_determinism_under_input_permutation(self):
        first = train_bpe(CORPUS, target_vocab=FIRST_MERGE_ID + 50)
        permuted = list(reversed(CORPUS))
        second = train_bpe(permuted, target_vocab=FIRST_MERGE_ID + 50)
        self.assertEqual(first, second)

    def test_incremental_counts_match_brute_force(self):
        artifact = train_bpe(CORPUS, target_vocab=FIRST_MERGE_ID + 64)
        for text in CORPUS[:4]:
            self.assertEqual(
                encode_bytes(text.encode("utf-8"), merge_ranks(artifact["merges"])),
                naive_encode(text.encode("utf-8"), artifact["merges"]),
            )

    def test_corpus_exhaustion_raises(self):
        with self.assertRaises(TokenizerError):
            train_bpe(["abab"], target_vocab=FIRST_MERGE_ID + 10)

    def test_target_vocab_bounds(self):
        with self.assertRaises(TokenizerError):
            train_bpe(CORPUS, target_vocab=100)
        with self.assertRaises(TokenizerError):
            train_bpe(CORPUS, target_vocab=99_999)

    def test_shard_loader_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TokenizerError):
                load_shard_texts(Path(tmp))

    def test_end_to_end_on_built_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = build_shard_dir(
                Path(tmp) / "shards",
                [
                    "bola bola bola voli voli voli data data data",
                    "mata mata mata litera litera litera lunar lunar",
                ],
            )
            texts = load_shard_texts(shard_dir)
            self.assertEqual(len(texts), 2)
            artifact = train_bpe(texts, target_vocab=FIRST_MERGE_ID + 12)
            ranks = merge_ranks(artifact["merges"])
            for text in texts:
                self.assertEqual(
                    decode_content(encode_content(text, ranks), artifact["merges"]),
                    text,
                )
            stats = artifact_stats(texts, artifact)
            self.assertEqual(stats["documents"], 2)
            self.assertGreater(stats["bytes_per_token"], 1.0)

    def test_load_artifact_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"version": 1, "vocab_size": 999, "merges": []}))
            with self.assertRaises(TokenizerError):
                load_artifact(path)

    def test_token_bytes_rejects_control_ids(self):
        with self.assertRaises(TokenizerError):
            token_bytes(0, {})

    def test_golden_fixture_regression(self):
        fixture_dir = REPO_ROOT / "crates" / "tokenizer" / "tests" / "golden"
        golden_path = fixture_dir / "bpe_smoke_v1.json"
        sample = TRAINING_DIR / "fixtures" / "smoke" / "sample.jsonl"
        texts = [
            json.loads(line)["text"]
            for line in sample.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifact = train_bpe(texts, target_vocab=FIRST_MERGE_ID + 64, version=1)
        regenerated = json.dumps(
            artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.assertEqual(regenerated, golden_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

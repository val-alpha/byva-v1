"""Tests for the Phase 3 training reference implementation.

These tests require PyTorch; they skip (rather than fail) when it is not
installed, and any skipped run is recorded as "not run" in the phase report.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from train import (
    SMOKE_PATTERN_PERIOD,
    DecoderOnlyTransformer,
    ModelConfig,
    TrainState,
    byte_token_stream,
    load_checkpoint,
    load_shard_texts,
    make_sequences,
    save_checkpoint,
    smoke_token_stream,
    train,
)

TINY = ModelConfig(d_model=256, n_layers=4, n_heads=8, ffn_hidden=1024, max_seq_len=512)
SMALL = ModelConfig(
    d_model=512, n_layers=8, n_heads=8, ffn_hidden=2048, max_seq_len=1024
)


def smoke_train_x_y():
    stream = smoke_token_stream(6_000)
    return make_sequences(stream, 64)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TrainingTest(unittest.TestCase):
    def test_python_param_count_matches_rust_golden_values(self):
        self.assertEqual(TINY.parameter_count(), 11_670_016)
        self.assertEqual(SMALL.parameter_count(), 42_484_736)
        untied = dataclasses.replace(SMALL, tie_embeddings=False)
        self.assertEqual(untied.parameter_count(), 59_261_952)

    def test_instantiated_model_numel_matches_formula(self):
        model = DecoderOnlyTransformer(TINY)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11_670_016)

    def test_config_validation_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=100).validate()
        with self.assertRaises(ValueError):
            ModelConfig(d_model=100).validate()
        with self.assertRaises(ValueError):
            ModelConfig(n_layers=0).validate()

    def test_loss_falls_on_structured_smoke_stream(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        _, state = train(
            TINY, train_x, train_y, val_x, val_y, steps=40, log_every=0, batch_size=4
        )
        self.assertLess(state.loss_history[-1], state.loss_history[0])

    def test_checkpoint_resume_matches_uninterrupted_run(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        _, reference_state = train(
            TINY, train_x, train_y, val_x, val_y, steps=24, log_every=0, batch_size=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            _, interrupted = train(
                TINY,
                train_x,
                train_y,
                val_x,
                val_y,
                steps=12,
                log_every=0,
                batch_size=4,
                checkpoint_path=path,
                checkpoint_every=6,
            )
            _, resumed_state = train(
                TINY,
                train_x,
                train_y,
                val_x,
                val_y,
                steps=12,
                log_every=0,
                batch_size=4,
                checkpoint_path=path,
                checkpoint_every=6,
                resume=True,
            )
        self.assertEqual(resumed_state.step, 24)
        self.assertEqual(
            resumed_state.loss_history,
            reference_state.loss_history,
            "resumed run must reproduce the uninterrupted run exactly",
        )

    def test_greedy_generation_follows_learned_pattern(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        model, _ = train(
            TINY, train_x, train_y, val_x, val_y, steps=60, log_every=0, batch_size=4
        )
        start = 64 + SMOKE_PATTERN_PERIOD - 3
        prompt = torch.tensor([[start, start + 1, start + 2]])
        generated = model.generate(prompt, max_new_tokens=3, temperature=0.0)
        expected = [
            (start - 64 + i) % SMOKE_PATTERN_PERIOD + 64 for i in range(3, 6)
        ]
        self.assertEqual(generated[0, 3:].tolist(), expected)

    def test_shard_loader_and_byte_stream_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            shard_dir.mkdir()
            docs = [{"text": "Once upon a time"}, {"text": "Cerita bahasa Indonesia."}]
            with (shard_dir / "shard-00000.jsonl").open("w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            (shard_dir / "manifest.json").write_text("{}", encoding="utf-8")
            texts = load_shard_texts(shard_dir)
            self.assertEqual(texts, [doc["text"] for doc in docs])
            stream = byte_token_stream(texts)
            self.assertTrue(all(0 <= token < 256 for token in stream))
            self.assertEqual(
                bytes(stream).decode("utf-8"), "Once upon a timeCerita bahasa Indonesia."
            )
            with self.assertRaises(ValueError):
                load_shard_texts(Path(tmp))

    def test_byte_level_training_loss_falls(self):
        texts = [
            "Once upon a time there was a little car named Beep who loved to drive.",
            "Sebuah cerita berbahasa Indonesia tentang kucing yang penasaran.",
        ] * 30
        stream = byte_token_stream(texts)
        train_x, train_y, val_x, val_y = make_sequences(stream, 64)
        config = ModelConfig(d_model=64, n_layers=2, n_heads=8, ffn_hidden=128, max_seq_len=64)
        _, state = train(
            config, train_x, train_y, val_x, val_y, steps=30, log_every=0, batch_size=8
        )
        self.assertLess(state.loss_history[-1], state.loss_history[0])

    def test_checkpoint_rejects_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save({"format": "unknown"}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()

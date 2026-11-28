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

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    TINY = ModelConfig(
        d_model=256, n_layers=4, n_heads=8, ffn_hidden=1024, max_seq_len=512
    )
    SMALL = ModelConfig(
        d_model=512, n_layers=8, n_heads=8, ffn_hidden=2048, max_seq_len=1024
    )
else:
    TINY = None
    SMALL = None


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

    def test_v1_checkpoint_still_loads(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.pt"
            payload = {
                "format": "byva-train-checkpoint-v1",
                "step": 4,
                "config": dataclasses.asdict(TINY),
                "model": DecoderOnlyTransformer(TINY).state_dict(),
                "optimizer": torch.optim.AdamW(
                    DecoderOnlyTransformer(TINY).parameters(), lr=1e-3
                ).state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_python": __import__("random").getstate(),
                "loss_history": [1.0, 0.9],
            }
            torch.save(payload, path)
            model, _, state, scaler = load_checkpoint(path)
            self.assertIsNone(scaler)
            self.assertEqual(state.step, 4)

    def test_grad_accum_resume_matches_uninterrupted_run(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        _, reference_state = train(
            TINY, train_x, train_y, val_x, val_y,
            steps=24, log_every=0, batch_size=4, grad_accum=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            _, interrupted = train(
                TINY, train_x, train_y, val_x, val_y,
                steps=12, log_every=0, batch_size=4, grad_accum=2,
                checkpoint_path=path, checkpoint_every=6,
            )
            _, resumed_state = train(
                TINY, train_x, train_y, val_x, val_y,
                steps=12, log_every=0, batch_size=4, grad_accum=2,
                checkpoint_path=path, checkpoint_every=6, resume=True,
            )
        self.assertEqual(resumed_state.step, 24)
        self.assertEqual(resumed_state.loss_history, reference_state.loss_history)

    def test_fp16_requires_cuda(self):
        train_x, train_y, val_x, val_y = smoke_train_x_y()
        with self.assertRaises(ValueError):
            train(
                TINY, train_x, train_y, val_x, val_y,
                steps=1, log_every=0, batch_size=4, use_fp16=True,
            )

    def test_bpe_token_stream_trains_and_decodes(self):
        from byva_tokenizer import FIRST_MERGE_ID, decode_content, merge_ranks
        from byva_tokenizer import encode_content
        from train import bpe_token_stream
        from train_bpe import train_bpe

        texts = [
            "bola bola bola voli voli voli data data data",
            "mata mata mata litera litera litera lunar lunar",
        ] * 40
        artifact = train_bpe(texts[:2], target_vocab=FIRST_MERGE_ID + 16)
        ranks = merge_ranks(artifact["merges"])
        for text in texts[:2]:
            self.assertEqual(
                decode_content(encode_content(text, ranks), artifact["merges"]), text
            )
        stream = bpe_token_stream(texts, artifact)
        self.assertTrue(all(0 <= token < 32_768 for token in stream))
        train_x, train_y, val_x, val_y = make_sequences(stream, 64)
        config = ModelConfig(d_model=64, n_layers=2, n_heads=8, ffn_hidden=128, max_seq_len=64)
        _, state = train(
            config, train_x, train_y, val_x, val_y, steps=20, log_every=0, batch_size=8
        )
        self.assertLess(state.loss_history[-1], state.loss_history[0])


if __name__ == "__main__":
    unittest.main()

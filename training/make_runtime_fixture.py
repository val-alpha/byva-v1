#!/usr/bin/env python3
"""Generate the Phase 5 runtime numerical-parity fixture (D-0022).

Builds synthetic weights from the shared SplitMix64 generator (seeded with
FIXTURE_SEED), runs the PyTorch reference forward pass from
training/train.py, and writes crates/runtime/tests/fixtures/runtime_golden_v1.json
with the exact f32 logits (as raw bit patterns) at the final prompt position.

The Rust side reproduces the identical weight sequence in
crates/runtime/src/lcg.rs (`fixture_weights`); the two must stay in lockstep.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DecoderOnlyTransformer, ModelConfig  # noqa: E402

FIXTURE_SEED = 0x2026_1128_5EED_0001
FIXTURE_WEIGHT_SCALE = 0.02
MASK64 = (1 << 64) - 1

PROMPT_TOKENS = [7, 42, 1024, 3, 29999, 8, 777, 12, 32123, 256]
FIXTURE_CONFIG = ModelConfig(
    vocab_size=32768,
    d_model=16,
    n_layers=2,
    n_heads=2,
    ffn_hidden=32,
    max_seq_len=32,
    tie_embeddings=True,
)
TOLERANCE = 1e-3


class SplitMix64:
    """Bit-exact mirror of crates/runtime/src/lcg.rs `SplitMix64`."""

    GAMMA = 0x9E37_79B9_7F4A_7C15

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + self.GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_uniform_f32(self) -> float:
        top24 = (self.next_u64() >> 40) & 0xFF_FFFF
        return top24 / 16_777_216.0 * 2.0 - 1.0

    def fill_scaled_f32(self, count: int) -> torch.Tensor:
        values = [self.next_uniform_f32() * FIXTURE_WEIGHT_SCALE for _ in range(count)]
        packed = struct.pack(f"<{count}f", *values)
        return torch.frombuffer(bytearray(packed), dtype=torch.float32).clone()


def build_state_dict(config: ModelConfig) -> dict[str, torch.Tensor]:
    generator = SplitMix64(FIXTURE_SEED)
    state: dict[str, torch.Tensor] = {}
    state["token_embedding.weight"] = generator.fill_scaled_f32(
        config.vocab_size * config.d_model
    ).reshape(config.vocab_size, config.d_model)
    state["position_embedding.weight"] = generator.fill_scaled_f32(
        config.max_seq_len * config.d_model
    ).reshape(config.max_seq_len, config.d_model)
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}."
        d = config.d_model
        for suffix, elements in (
            ("norm_attn.weight", d),
            ("norm_attn.bias", d),
            ("attention.qkv.weight", 3 * d * d),
            ("attention.out.weight", d * d),
            ("norm_mlp.weight", d),
            ("norm_mlp.bias", d),
            ("mlp.0.weight", d * config.ffn_hidden),
            ("mlp.2.weight", config.ffn_hidden * d),
        ):
            shape = _shape_for(suffix, config)
            state[f"{prefix}{suffix}"] = generator.fill_scaled_f32(elements).reshape(shape)
    state["final_norm.weight"] = generator.fill_scaled_f32(config.d_model)
    state["final_norm.bias"] = generator.fill_scaled_f32(config.d_model)
    return state


def _shape_for(suffix: str, config: ModelConfig) -> tuple[int, ...]:
    d = config.d_model
    if suffix in ("attention.qkv.weight",):
        return (3 * d, d)
    if suffix in ("attention.out.weight",):
        return (d, d)
    if suffix == "mlp.0.weight":
        return (config.ffn_hidden, d)
    if suffix == "mlp.2.weight":
        return (d, config.ffn_hidden)
    return (d,)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crates/runtime/tests/fixtures/runtime_golden_v1.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)

    config = FIXTURE_CONFIG
    config.validate()
    model = DecoderOnlyTransformer(config)
    state = build_state_dict(config)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys

    tokens = torch.tensor([PROMPT_TOKENS], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens)
    final_logits = logits[0, -1].to(torch.float32)
    bits = [f32_bits(float(value)) for value in final_logits.tolist()]
    argmax = int(final_logits.argmax().item())

    fixture = {
        "format": "byva-runtime-parity-v1",
        "generator": "training/make_runtime_fixture.py",
        "fixture_seed": FIXTURE_SEED,
        "weight_scale": FIXTURE_WEIGHT_SCALE,
        "config": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "ffn_hidden": config.ffn_hidden,
            "max_seq_len": config.max_seq_len,
            "tie_embeddings": config.tie_embeddings,
        },
        "prompt": PROMPT_TOKENS,
        "expected_argmax": argmax,
        "tolerance": TOLERANCE,
        "expected_logits_bits": bits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture))
    print(f"wrote {args.output} ({len(bits)} logits, argmax={argmax})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

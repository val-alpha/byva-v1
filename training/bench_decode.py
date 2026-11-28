"""CPU greedy-decode benchmark for the reference TINY model (D-0019 proxy).

Measures single-thread greedy-decode throughput of the PyTorch reference
implementation so the Phase 5/7 Rust runtime numbers have a recorded
training-side baseline. This is a proxy only: the D-0019 budget (TINY >= 200
tokens/s on CPU) may be claimed met exclusively with the Rust harness.

Prompt ids are drawn from a fixed-seed generator; decode cost per token is
independent of token values because the reference generate() re-runs the full
forward pass without a KV cache.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from train import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, _, state, _ = load_checkpoint(args.checkpoint)
    model.eval()
    prompt = torch.randint(
        0, model.config.vocab_size, (1, args.prompt_tokens), generator=torch.manual_seed(0)
    )
    model.generate(prompt, 8, temperature=0.0)

    rates = []
    for _ in range(args.runs):
        start = time.perf_counter()
        model.generate(prompt, args.new_tokens, temperature=0.0)
        rates.append(args.new_tokens / (time.perf_counter() - start))

    print(
        f"step {state.step} | threads {torch.get_num_threads()} | "
        f"prompt {args.prompt_tokens} + {args.new_tokens} greedy tokens | "
        f"median {statistics.median(rates):.1f} tok/s | "
        f"runs {[round(rate, 1) for rate in rates]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

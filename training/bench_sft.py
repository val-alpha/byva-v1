"""Phase 6 identity-recall gate (smoke-scale SFT through the released CLI).

Protocol (runtime-valid small model, CPU fp32, end to end via `byva-cli`):

1. Base run:  `byva-cli train` on the synthetic bilingual corpus at small
   dims, so the SFT stage starts from a real pretraining checkpoint.
2. SFT run:   `byva-cli sft --identity` (the built-in bilingual byva/BYV
   identity set, upsampled by `--repeat`), with loss masked to the response
   tokens only.
3. Convert:   `byva-cli convert` from the Rust SFT checkpoint to a
   `byva-weights-v1` runtime container.
4. Recall:    `byva-cli generate --chat` on the two canonical identity
   prompts, one per language.

Gate: the greedy continuations of "Siapa namamu?" and "What is your name?"
must both name `byva` and attribute the creator `BYV`. This is a smoke-scale
check of the identity path only; the full Phase 6 scope (tool behaviour,
schema, tool-selection, multi-step evals per PHASES.md) remains open.

Recorded run (D-0025): base 10 steps + SFT 1000 steps (repeat 32) reaches
held-out 0.0644 and both prompts decode to the exact identity sentences.

Usage (from the repository root):

    python3 training/bench_sft.py [--base-steps N] [--sft-steps N] [--workdir DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from parity_rust_ne2410 import synthetic_corpus  # noqa: E402

TOKENIZER = REPO_ROOT / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json"

BASE_FLAGS = [
    "--d-model", "32", "--layers", "2", "--heads", "2", "--ffn", "64",
    "--vocab", "32768", "--seq-len", "64",
    "--token-budget", "512",
    "--lr", "1e-3", "--warmup", "0", "--floor-ratio", "1.0",
    "--weight-decay", "0.01", "--eval-every", "5", "--log-every", "5",
    "--seed", "1",
]

IDENTITY_PROMPTS = ["Siapa namamu?", "What is your name?"]


def cli_binary() -> Path:
    binary = REPO_ROOT / "target" / "release" / "byva-cli"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "byva-cli"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
    return binary


def run_cli(command: list[str]) -> str:
    result = subprocess.run(
        [str(cli_binary()), *command],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-steps", type=int, default=10)
    parser.add_argument("--sft-steps", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp()) if args.workdir is None else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    docs = synthetic_corpus(40)
    corpus_path = workdir / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs), encoding="utf-8"
    )
    shards_dir = workdir / "shards"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "training" / "build_shards.py"),
         "--input", str(corpus_path), "--output-dir", str(shards_dir),
         "--docs-per-shard", "20"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    print(f"workdir: {workdir}")

    base_ckpt = workdir / "base.ckpt"
    run_cli([
        "train", "--shards", str(shards_dir), "--tokenizer", str(TOKENIZER),
        "--out", str(base_ckpt), "--steps", str(args.base_steps), *BASE_FLAGS,
    ])
    print(f"base checkpoint: {base_ckpt}")

    sft_ckpt = workdir / "sft.ckpt"
    sft_out = run_cli([
        "sft", "--base", str(base_ckpt), "--tokenizer", str(TOKENIZER),
        "--identity", "--repeat", str(args.repeat),
        "--steps", str(args.sft_steps), "--seq-len", "64",
        "--token-budget", "512", "--lr", "1e-3",
        "--eval-every", "50", "--log-every", "25", "--seed", "7",
        "--out", str(sft_ckpt),
    ])
    print(sft_out.strip())

    weights_path = workdir / "sft.weights"
    convert_out = run_cli([
        "convert", "--checkpoint", str(sft_ckpt), "--out", str(weights_path),
    ])
    print(convert_out.strip())

    failures: list[str] = []
    for prompt in IDENTITY_PROMPTS:
        text = run_cli([
            "generate", "--weights", str(weights_path), "--tokenizer", str(TOKENIZER),
            "--chat", "--prompt", prompt, "--max-new", "48",
        ]).strip()
        print(f"\n{prompt!r} -> {text!r}")
        if "byva" not in text.lower():
            failures.append(f"{prompt!r}: reply does not name byva")
        if "BYV" not in text:
            failures.append(f"{prompt!r}: reply does not attribute BYV")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("\nSFT GATE: PASS (identity recalled in Indonesian and English)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R5 performance gate: ne2410 Rust trainer throughput vs the PyTorch
reference, plus the SignTrust optimizer gate (D-0022, K6).

Protocol (all runs TINY, 1024 tokens/step, CPU fp32, same synthetic shards,
same seed so both optimizers see identical init and identical batches):

1. AdamW run:     `byva-cli train --log-every 1` for STEPS steps; per-step
   wall times parsed from the log for mean/p50/p99 (eval steps excluded).
2. SignTrust run: identical settings; gate = held-out loss must be at least
   as low as AdamW's at (approximately) equal wall-clock.
3. PyTorch run:   the reference trainer on the same batches for the same
   step count, timed per step (the fresh baseline in the same session).

Usage (from the repository root, torch required; the release CLI binary is
built first):

    python3 training/bench_ne2410.py [--steps N] [--workdir DIR]

Artifacts stay under a temp workdir; nothing is committed (D-0007).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from parity_rust_ne2410 import (  # noqa: E402
    encode_shards,
    load_rust_checkpoint,
    make_windows,
    rust_config_to_model_config,
    synthetic_corpus,
)
from train import DecoderOnlyTransformer  # noqa: E402

SIGNTRUST_GATE_EPS = 1e-6
THROUGHPUT_FACTOR = 2.0
STEP_LINE = re.compile(r"step (\d+)/\d+ loss ([\d.]+) \(([\d.]+)(ns|µs|ms|s)/step\)")
UNIT_SECONDS = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}

MODEL_FLAGS = [
    "--d-model", "256", "--layers", "4", "--heads", "8", "--ffn", "1024",
    "--vocab", "32768", "--seq-len", "128",
    "--token-budget", "1024",
    "--lr", "1e-3", "--warmup", "0", "--floor-ratio", "1.0",
    "--weight-decay", "0.01", "--eval-every", "10", "--log-every", "1",
    "--seed", "1",
]


def cli_binary() -> Path:
    binary = REPO_ROOT / "target" / "release" / "byva-cli"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "byva-cli"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
    return binary


def run_cli(shards_dir: Path, workdir: Path, optimizer: str, steps: int) -> dict:
    out_path = workdir / f"{optimizer}.ckpt"
    started = time.perf_counter()
    process = subprocess.run(
        [
            str(cli_binary()), "train",
            "--shards", str(shards_dir),
            "--tokenizer", str(REPO_ROOT / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json"),
            "--out", str(out_path),
            "--steps", str(steps),
            "--optimizer", optimizer,
            *MODEL_FLAGS,
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    wall = time.perf_counter() - started
    step_times: dict[int, float] = {}
    for line in process.stderr.splitlines():
        match = STEP_LINE.search(line)
        if match:
            step = int(match.group(1))
            value = float(match.group(3)) * UNIT_SECONDS[match.group(4)]
            step_times[step] = value
    meta = load_rust_checkpoint(out_path)
    return {
        "wall": wall,
        "step_times": step_times,
        "loss_history": meta["loss_history"],
        "eval_history": meta["eval_history"],
        "config": meta["config"],
    }


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no samples")
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def summarize_run(
    label: str, run: dict, steps: int, eval_every: int
) -> tuple[float, float, float]:
    samples = [
        run["step_times"][step]
        for step in sorted(run["step_times"])
        if step > 2 and step % eval_every != 0
    ]
    mean = statistics.fmean(samples)
    p50 = percentile(samples, 0.50)
    p99 = percentile(samples, 0.99)
    last_eval = run["eval_history"][-1] if run["eval_history"] else None
    print(
        f"{label}: mean {mean:.3f} s/step, p50 {p50:.3f}, p99 {p99:.3f} "
        f"({len(samples)} samples, {steps} steps, wall {run['wall']:.1f}s), "
        f"final train loss {run['loss_history'][-1]:.4f}"
        + (f", final held-out loss {last_eval['loss']:.4f}" if last_eval else "")
    )
    return mean, p99, (last_eval["loss"] if last_eval else float("nan"))


def run_reference(shards_dir: Path, workdir: Path, steps: int) -> float:
    init_meta = load_rust_checkpoint(workdir / "adamw.ckpt")
    config = rust_config_to_model_config(init_meta["config"])
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(init_meta["_params"], strict=True)
    model.train()
    tokens = encode_shards(shards_dir)
    seq_len = config.max_seq_len
    train_windows, _ = make_windows(tokens, seq_len)
    batch_size = init_meta["config"]["token_budget"] // seq_len
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cursor, count = 0, len(train_windows)
    durations: list[float] = []
    for _ in range(steps):
        rows = [(cursor + offset) % count for offset in range(batch_size)]
        batch_x = torch.tensor([train_windows[row][:seq_len] for row in rows], dtype=torch.long)
        batch_y = torch.tensor([train_windows[row][1:] for row in rows], dtype=torch.long)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        durations.append(time.perf_counter() - started)
        cursor = (cursor + batch_size) % count
    mean = statistics.fmean(durations[2:])
    print(
        f"pytorch: mean {mean:.3f} s/step, p50 {percentile(durations[2:], 0.50):.3f}, "
        f"p99 {percentile(durations[2:], 0.99):.3f} ({steps} steps, torch {torch.__version__})"
    )
    return mean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp()) if args.workdir is None else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    docs = synthetic_corpus(120)
    corpus_path = workdir / "corpus.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs), encoding="utf-8"
    )
    shards_dir = workdir / "shards"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "training" / "build_shards.py"),
         "--input", str(corpus_path), "--output-dir", str(shards_dir),
         "--docs-per-shard", "40"],
        cwd=REPO_ROOT / "training", check=True, capture_output=True, text=True,
    )
    print(f"shards: {shards_dir}")

    eval_every = 10
    adamw = run_cli(shards_dir, workdir, "adamw", args.steps)
    adamw_mean, adamw_p99, adamw_eval = summarize_run(
        "ne2410 adamw    ", adamw, args.steps, eval_every
    )
    signtrust = run_cli(shards_dir, workdir, "signtrust", args.steps)
    signtrust_mean, signtrust_p99, signtrust_eval = summarize_run(
        "ne2410 signtrust", signtrust, args.steps, eval_every
    )
    torch_mean = run_reference(shards_dir, workdir, args.steps)

    failures: list[str] = []
    speedup = torch_mean / adamw_mean
    print(f"\nthroughput vs pytorch: {speedup:.2f}x (gate >= {THROUGHPUT_FACTOR:.0f}x)")
    if speedup < THROUGHPUT_FACTOR:
        failures.append(
            f"throughput {speedup:.2f}x below gate {THROUGHPUT_FACTOR:.0f}x"
        )
    wall_ratio = signtrust["wall"] / adamw["wall"]
    print(
        f"signtrust gate: held-out {signtrust_eval:.4f} vs adamw {adamw_eval:.4f} "
        f"(wall ratio {wall_ratio:.2f})"
    )
    if signtrust_eval > adamw_eval + SIGNTRUST_GATE_EPS:
        failures.append(
            f"signtrust held-out loss {signtrust_eval:.4f} above adamw {adamw_eval:.4f}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("R5 GATES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

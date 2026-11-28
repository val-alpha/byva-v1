"""R6 mixture gate (K5, D-0016): the inline loss-aware id:en mixture.

Protocol (TINY, 1024 tokens/step, CPU fp32, same synthetic bilingual shards,
same seed, AdamW for both runs):

1. Baseline run:   natural corpus order (no mixture flags).
2. Mixture run:    `--lang-share-id 0.6 --lang-band 0.15`; windows are tagged
   inline by the deterministic document-language detector, batches are drawn
   from per-language pools at the current Indonesian share, and the share
   adapts at every evaluation within the band.

Gate (K6): every adjustment is recorded in the checkpoint's mixture history
and stays inside the configured band; the quality delta (held-out loss,
baseline vs mixture, plus the per-language eval losses) is reported.

Usage (from the repository root, the release CLI binary is built first):

    python3 training/bench_mixture.py [--steps N] [--workdir DIR]
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

from parity_rust_ne2410 import (  # noqa: E402
    load_rust_checkpoint,
    rust_config_to_model_config,
    synthetic_corpus,
)

MODEL_FLAGS = [
    "--d-model", "256", "--layers", "4", "--heads", "8", "--ffn", "1024",
    "--vocab", "32768", "--seq-len", "128",
    "--token-budget", "1024",
    "--lr", "1e-3", "--warmup", "0", "--floor-ratio", "1.0",
    "--weight-decay", "0.01", "--eval-every", "10", "--log-every", "10",
    "--seed", "1",
]
MIXTURE_FLAGS = ["--lang-share-id", "0.6", "--lang-band", "0.15"]


def cli_binary() -> Path:
    binary = REPO_ROOT / "target" / "release" / "byva-cli"
    if not binary.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "byva-cli"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        )
    return binary


def run_cli(shards_dir: Path, workdir: Path, label: str, steps: int, mixture: bool) -> dict:
    out_path = workdir / f"{label}.ckpt"
    command = [
        str(cli_binary()), "train",
        "--shards", str(shards_dir),
        "--tokenizer", str(REPO_ROOT / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json"),
        "--out", str(out_path),
        "--steps", str(steps),
        *MODEL_FLAGS,
        *(MIXTURE_FLAGS if mixture else []),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    meta = load_rust_checkpoint(out_path)
    return {
        "loss_history": meta["loss_history"],
        "eval_history": meta["eval_history"],
        "mixture_history": meta.get("mixture_history", []),
        "config": meta["config"],
    }


def describe(label: str, run: dict) -> float:
    train_loss = run["loss_history"][-1]
    held_out = run["eval_history"][-1]["loss"] if run["eval_history"] else float("nan")
    line = f"{label}: train {train_loss:.4f}, held-out {held_out:.4f}"
    last = run["eval_history"][-1] if run["eval_history"] else {}
    if last.get("loss_id") is not None:
        line += f" (id {last['loss_id']:.4f}, en {last['loss_en']:.4f})"
    print(line)
    return held_out


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
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    print(f"shards: {shards_dir}")

    baseline = run_cli(shards_dir, workdir, "baseline", args.steps, mixture=False)
    mixture = run_cli(shards_dir, workdir, "mixture", args.steps, mixture=True)

    baseline_held_out = describe("ne2410 baseline (natural order)", baseline)
    mixture_held_out = describe("ne2410 mixture   (id share 0.6)", mixture)

    history = mixture["mixture_history"]
    target = mixture["config"]["mixture_target_id"]
    band = mixture["config"]["mixture_band"]
    failures: list[str] = []
    if not history:
        failures.append("mixture history is empty: no adjustment was recorded")
    outside = [r for r in history if not (target - band <= r["id_share"] <= target + band)]
    if outside:
        failures.append(
            f"{len(outside)} adjustment(s) outside the band [{target - band}, {target + band}]"
        )
    if history:
        print("\nmixture adjustments (recorded in the checkpoint manifest):")
        for record in history:
            print(
                f"  step {record['step']}: id_share {record['id_share']:.3f} "
                f"(id {record['id_loss']:.4f} vs en {record['en_loss']:.4f})"
            )
    delta = mixture_held_out - baseline_held_out
    print(f"\nquality delta (mixture - baseline held-out): {delta:+.4f}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("MIXTURE GATE: PASS (adjustments recorded and bounded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

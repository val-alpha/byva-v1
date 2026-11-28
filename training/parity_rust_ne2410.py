"""R4 parity gate: ne2410 Rust trainer vs the PyTorch reference (D-0022).

Runs both trainers on the SAME shards (built by the real `build_shards.py`
pipeline) and the SAME batches, starting from the SAME initial weights, then
compares:

1. per-step training loss (forward + backward + AdamW over N steps),
2. the final parameter tensors (max abs elementwise difference),

and records the PyTorch wall-clock per step (the R5 baseline).

Batch sampling policy is deliberately NOT compared: the reference samples
windows with `torch.randint`, while ne2410/K4 uses a deterministic cycling
batch cursor. The parity script feeds the reference the same explicit
sequential-cyclic batches the Rust trainer consumes, so the comparison
isolates the math (forward, backward, optimizer) from the batching policy.

Usage (from the repository root, torch required):

    python3 training/parity_rust_ne2410.py [--steps N] [--workdir DIR]

Artifacts stay under a temp workdir; nothing is committed (D-0007).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from byva_tokenizer import load_artifact, merge_ranks, encode_content  # noqa: E402
from train import ModelConfig, DecoderOnlyTransformer  # noqa: E402

MAGIC = b"BYVATRCK"
DIGEST_LEN = 64

GATE_LOSS_ABS = 5e-3
GATE_WEIGHT_ABS = 1e-3

ID_WORDS = ["anak", "kota", "sungai", "buku", "hujan", "pagi", "sekolah", "pasar",
            "keluarga", "perjalanan", "cerita", "pekerjaan", "makanan", "musik"]
EN_WORDS = ["child", "city", "river", "book", "rain", "morning", "school", "market",
            "family", "journey", "story", "work", "food", "music"]


def synthetic_corpus(doc_count: int) -> list[dict]:
    """Deterministic bilingual docs; no RNG so both pipelines see identical bytes."""
    docs = []
    for index in range(doc_count):
        lang = index % 2
        words = ID_WORDS if lang == 0 else EN_WORDS
        opener = "Pada hari" if lang == 0 else "On the day"
        sentences = []
        for s in range(6):
            w1 = words[(index + s) % len(words)]
            w2 = words[(index * 3 + s * 5) % len(words)]
            w3 = words[(index * 7 + s * 11) % len(words)]
            if lang == 0:
                sentences.append(f"{opener} {index} ini, {w1} melihat {w2} di dekat {w3} dengan tenang.")
            else:
                sentences.append(f"{opener} {index}, the {w1} watched the {w2} near the {w3} quietly.")
        docs.append({"text": " ".join(sentences)})
    return docs


def encode_shards(shards_dir: Path) -> list[int]:
    """The Rust CLI's exact data path: manifest order, docs in shard order."""
    artifact = load_artifact(REPO_ROOT / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json")
    ranks = merge_ranks(artifact["merges"])
    manifest = json.loads((shards_dir / "manifest.json").read_text(encoding="utf-8"))
    stream: list[int] = []
    for entry in manifest["shards"]:
        for line in (shards_dir / entry["name"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                stream.extend(encode_content(json.loads(line)["text"], ranks))
    return stream


def make_windows(tokens: list[int], seq_len: int) -> tuple[list[list[int]], list[list[int]]]:
    window_len = seq_len + 1
    usable = len(tokens) - (len(tokens) % window_len)
    if usable < 2 * window_len:
        raise ValueError(f"not enough tokens: {usable} < {2 * window_len}")
    windows = [tokens[start:start + window_len] for start in range(0, usable, window_len)]
    split = (len(windows) // 10) * 9
    return windows[:split], windows[split:]


def load_rust_checkpoint(path: Path) -> dict:
    raw = path.read_bytes()
    payload, digest_bytes = raw[:-DIGEST_LEN], raw[-DIGEST_LEN:]
    if hashlib.sha256(payload).hexdigest() != digest_bytes.decode("ascii"):
        raise ValueError(f"sha256 mismatch in {path}")
    if payload[:8] != MAGIC:
        raise ValueError(f"bad magic in {path}")
    version = int.from_bytes(payload[8:12], "little")
    if version != 1:
        raise ValueError(f"bad version {version}")
    meta_len = int.from_bytes(payload[12:20], "little")
    meta = json.loads(payload[20:20 + meta_len])
    body = payload[20 + meta_len:]
    floats = len(body) // 4
    values = torch.frombuffer(bytearray(body), dtype=torch.float32, count=floats)
    cursor = 0
    params: dict[str, torch.Tensor] = {}
    for desc in meta["params"]:
        take = desc["len"]
        params[desc["name"]] = values[cursor:cursor + take].clone().view(desc["shape"])
        cursor += take
    meta["_params"] = params
    return meta


def rust_config_to_model_config(config: dict) -> ModelConfig:
    return ModelConfig(
        vocab_size=config["model"]["vocab"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        ffn_hidden=config["model"]["ffn"],
        max_seq_len=config["model"]["seq_len"],
        tie_embeddings=True,
    )


def run_rust(shards_dir: Path, workdir: Path, steps: int) -> tuple[dict, dict, float]:
    init_path = workdir / "rust-init.ckpt"
    final_path = workdir / "rust-final.ckpt"
    started = time.perf_counter()
    subprocess.run(
        [
            "cargo", "run", "--release", "-p", "byva-cli", "--", "train",
            "--shards", str(shards_dir),
            "--tokenizer", str(REPO_ROOT / "artifacts" / "tokenizer" / "byva-tokenizer-v1.json"),
            "--out", str(final_path),
            "--save-init", str(init_path),
            "--steps", str(steps),
            "--optimizer", "adamw",
            "--d-model", "256", "--layers", "4", "--heads", "8", "--ffn", "1024",
            "--vocab", "32768", "--seq-len", "128",
            "--token-budget", "1024",
            "--lr", "1e-3", "--warmup", "0", "--floor-ratio", "1.0",
            "--weight-decay", "0.01", "--eval-every", "0", "--log-every", "0",
            "--seed", "1",
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    rust_seconds = time.perf_counter() - started
    return load_rust_checkpoint(init_path), load_rust_checkpoint(final_path), rust_seconds


def run_reference(init_meta: dict, final_meta: dict, tokens: list[int], steps: int) -> tuple[list[float], float, float]:
    config = rust_config_to_model_config(init_meta["config"])
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(init_meta["_params"], strict=True)
    model.train()

    seq_len = config.max_seq_len
    train_windows, _ = make_windows(tokens, seq_len)
    batch_size = init_meta["config"]["token_budget"] // seq_len

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    cursor, count = 0, len(train_windows)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        rows = [(cursor + offset) % count for offset in range(batch_size)]
        batch_x = torch.tensor([train_windows[row][:seq_len] for row in rows], dtype=torch.long)
        batch_y = torch.tensor([train_windows[row][1:] for row in rows], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        loss.backward()
        optimizer.step()
        cursor = (cursor + batch_size) % count
        losses.append(loss.item())
    reference_seconds = (time.perf_counter() - started) / steps

    final_max = 0.0
    reference_params = dict(model.state_dict())
    for name, tensor in final_meta["_params"].items():
        diff = (reference_params[name] - tensor).abs().max().item()
        final_max = max(final_max, diff)
    return losses, reference_seconds, final_max


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
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
    tokens = encode_shards(shards_dir)
    print(f"corpus: {len(docs)} docs, {len(tokens)} tokens -> {shards_dir}")

    init_meta, final_meta, rust_seconds = run_rust(shards_dir, workdir, args.steps)
    rust_losses = final_meta["loss_history"]
    if len(rust_losses) != args.steps:
        raise ValueError(f"rust loss history has {len(rust_losses)} entries, expected {args.steps}")

    reference_losses, reference_step_seconds, final_max = run_reference(
        init_meta, final_meta, tokens, args.steps
    )

    print("\nstep | rust loss | torch loss | abs diff")
    worst = 0.0
    for index, (rust, reference) in enumerate(zip(rust_losses, reference_losses), start=1):
        diff = abs(rust - reference)
        worst = max(worst, diff)
        print(f"{index:4d} | {rust:.6f} | {reference:.6f} | {diff:.3e}")
    print(f"\nmax abs loss diff:      {worst:.3e} (gate {GATE_LOSS_ABS:.0e})")
    print(f"max abs final wt diff:  {final_max:.3e} (gate {GATE_WEIGHT_ABS:.0e})")
    print(f"torch step baseline:    {reference_step_seconds:.3f} s/step "
          f"(TINY, {init_meta['config']['token_budget']} tokens/step, cpu fp32)")
    print(f"rust total (build+5-model run): {rust_seconds:.1f}s incl. cargo")

    failures = []
    if worst > GATE_LOSS_ABS:
        failures.append(f"loss diff {worst:.3e} exceeds gate {GATE_LOSS_ABS:.0e}")
    if final_max > GATE_WEIGHT_ABS:
        failures.append(f"weight diff {final_max:.3e} exceeds gate {GATE_WEIGHT_ABS:.0e}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PARITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

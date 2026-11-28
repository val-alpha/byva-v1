"""BYVA decoder-only pretraining reference implementation (PyTorch, training-only).

Architecture is authored in this repository (D-0013): pre-norm decoder-only
transformer, causal self-attention, GELU MLP, learned absolute positions, tied
token embedding and output projection by default. PyTorch is used only for
tensor operations and autograd (CLAUDE.md constraint).

Checkpoint policy (D-0014, extended by D-0018): every checkpoint stores step,
model state, optimizer state, both RNG states, config, and the loss history, so
a resumed run continues bit-for-bit on the same hardware. Format
`byva-train-checkpoint-v2` additionally stores the FP16 GradScaler state; v1
checkpoints (no scaler) still load.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from byva_tokenizer import load_artifact, merge_ranks, encode_content


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Mirrors crates/model/src/lib.rs `ModelConfig`; must stay in sync."""

    vocab_size: int = 32768
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_hidden: int = 1024
    max_seq_len: int = 512
    tie_embeddings: bool = True

    def validate(self) -> None:
        if not (32768 <= self.vocab_size <= 49152):
            raise ValueError(f"vocab_size {self.vocab_size} outside 32768..=49152")
        for name, value in (
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("ffn_hidden", self.ffn_hidden),
            ("max_seq_len", self.max_seq_len),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}"
            )

    def parameter_count(self) -> int:
        self.validate()
        d = self.d_model
        embedding = self.vocab_size * d
        positions = self.max_seq_len * d
        per_layer = 4 * d * d + 2 * d * self.ffn_hidden
        norms = 4 * d * self.n_layers + 2 * d
        total = embedding + positions + per_layer * self.n_layers + norms
        if not self.tie_embeddings:
            total += embedding
        return total


TINY_CONFIG = ModelConfig(
    d_model=256, n_layers=4, n_heads=8, ffn_hidden=1024, max_seq_len=512
)
SMALL_CONFIG = ModelConfig(
    d_model=512, n_layers=8, n_heads=8, ffn_hidden=2048, max_seq_len=1024
)


def config_by_name(name: str) -> ModelConfig:
    if name == "tiny":
        return TINY_CONFIG
    if name == "small":
        return SMALL_CONFIG
    raise ValueError(f"unknown config {name!r}; expected 'tiny' or 'small'")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        weights = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        weights = weights.masked_fill(mask, float("-inf"))
        attended = F.softmax(weights, dim=-1) @ v
        out = attended.transpose(1, 2).contiguous().view(batch, seq, d)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(config.ffn_hidden, config.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attn(x))
        return x + self.mlp(self.norm_mlp(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        if config.tie_embeddings:
            self.lm_head: nn.Linear | None = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        if seq > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq} exceeds max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if self.lm_head is None:
            return x @ self.token_embedding.weight.t()
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        tokens = prompt
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.max_seq_len :]
            logits = self.forward(context)[:, -1, :]
            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def make_sequences(
    token_ids: list[int], seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs one long id stream into overlapping-free fixed-length windows.

    Returns (train_x, train_y, val_x, val_y) with a deterministic 90/10 split.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    usable = len(token_ids) - (len(token_ids) % (seq_len + 1))
    if usable < 2 * (seq_len + 1):
        raise ValueError("not enough tokens for a held-out split")
    windows = [
        token_ids[start : start + seq_len + 1]
        for start in range(0, usable, seq_len + 1)
    ]
    split = (len(windows) // 10) * 9
    train_windows = windows[:split]
    val_windows = windows[split:]
    train = torch.tensor(train_windows, dtype=torch.long)
    val = torch.tensor(val_windows, dtype=torch.long)
    return train[:, :-1], train[:, 1:], val[:, :-1], val[:, 1:]


@dataclasses.dataclass
class TrainState:
    step: int
    loss_history: list[float]


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "byva-train-checkpoint-v2",
        "step": state.step,
        "config": dataclasses.asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_python": random.getstate(),
        "loss_history": state.loss_history,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer, TrainState, torch.amp.GradScaler | None]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt not in ("byva-train-checkpoint-v1", "byva-train-checkpoint-v2"):
        raise ValueError(f"unsupported checkpoint format: {fmt!r}")
    if fmt == "byva-train-checkpoint-v1" and payload.get("scaler") is not None:
        raise ValueError("v1 checkpoints must not carry scaler state")
    config = ModelConfig(**payload["config"])
    model = DecoderOnlyTransformer(config)
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng_torch"])
    random.setstate(payload["rng_python"])
    state = TrainState(
        step=payload["step"], loss_history=list(payload["loss_history"])
    )
    scaler = None
    if fmt == "byva-train-checkpoint-v2" and payload.get("scaler") is not None:
        scaler = torch.amp.GradScaler()
        scaler.load_state_dict(payload["scaler"])
    return model, optimizer, state, scaler


def evaluate(
    model: DecoderOnlyTransformer, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y[start : start + batch_size].reshape(-1)
            )
            total += loss.item() * len(x[start : start + batch_size])
            count += len(x[start : start + batch_size])
    model.train()
    return total / count


def train(
    config: ModelConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    steps: int,
    lr: float = 1e-3,
    batch_size: int = 8,
    log_every: int = 50,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 200,
    resume: bool = False,
    generator_seed: int = 0,
    device: torch.device | None = None,
    use_fp16: bool = False,
    grad_accum: int = 1,
) -> tuple[DecoderOnlyTransformer, TrainState]:
    """Runs the training loop; resumable when `checkpoint_path` is provided."""
    config.validate()
    if grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")
    device = device or torch.device("cpu")
    if use_fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires a CUDA device")
    torch.manual_seed(generator_seed)
    model = DecoderOnlyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_fp16)
    state = TrainState(step=0, loss_history=[])
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        model, optimizer, state, scaler = load_checkpoint(checkpoint_path, device)
        if model.config != config:
            raise ValueError("checkpoint config differs from requested config")
        if use_fp16 and scaler is None:
            raise ValueError("--fp16 resume needs a v2 checkpoint with scaler state")
        model.train()
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    start = state.step
    for step in range(start, start + steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[torch.Tensor] = []
        for _ in range(grad_accum):
            indices = torch.randint(0, len(train_x), (batch_size,))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(train_x[indices])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    train_y[indices].reshape(-1),
                ) / grad_accum
            scaler.scale(loss).backward()
            micro_losses.append(loss.detach())
        scaler.step(optimizer)
        scaler.update()
        state.step = step + 1
        state.loss_history.append(
            sum(item.item() for item in micro_losses)
        )
        if log_every and (step + 1) % log_every == 0:
            val_loss = evaluate(model, val_x.to(device), val_y.to(device))
            print(
                f"step {state.step}: train_loss {state.loss_history[-1]:.4f} "
                f"val_loss {val_loss:.4f}"
            )
        if checkpoint_path is not None and checkpoint_every and state.step % checkpoint_every == 0:
            save_checkpoint(Path(checkpoint_path), model, optimizer, state, scaler)
    if checkpoint_path is not None:
        save_checkpoint(Path(checkpoint_path), model, optimizer, state, scaler)
    return model, state


SMOKE_PATTERN_PERIOD = 97


def smoke_token_stream(length: int) -> list[int]:
    """Structured synthetic stream (counting with a reset period) for CPU smokes.

    Real pretraining tokens come from the Phase 2 shard pipeline; this stream
    only exists so the training loop, checkpointing, and generation can be
    exercised without a GPU. The structure (x[i+1] = (x[i] + 1) % period with
    rising sawtooth) is learnable, so measured loss must fall.
    """
    stream = [0] * length
    value = 0
    for index in range(length):
        stream[index] = value + 64
        value = (value + 1) % SMOKE_PATTERN_PERIOD
    return stream


def load_shard_texts(shard_dir: Path) -> list[str]:
    """Loads documents from a Phase 2 shard directory (manifest-verified)."""
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no manifest.json in {shard_dir}")
    texts: list[str] = []
    for shard in sorted(shard_dir.glob("shard-*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                texts.append(json.loads(line)["text"])
    return texts


def byte_token_stream(texts: list[str]) -> list[int]:
    """UTF-8 byte ids (0..255) inside the full vocabulary.

    The 32k BPE vocabulary is not trained yet (D-0015), so dataset smokes train
    on raw byte ids; the model config still validates at the production vocab.
    """
    stream: list[int] = []
    for text in texts:
        stream.extend(text.encode("utf-8"))
    return stream


def bpe_token_stream(texts: list[str], artifact: dict) -> list[int]:
    """Encodes shard texts with a trained BPE artifact (production path, D-0015)."""
    ranks = merge_ranks(artifact["merges"])
    stream: list[int] = []
    for text in texts:
        stream.extend(encode_content(text, ranks))
    return stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--smoke",
        action="store_true",
        help="run the tiny-config CPU smoke training on the synthetic stream",
    )
    source.add_argument(
        "--shards",
        type=Path,
        help="train on a shard directory built by training/build_shards.py",
    )
    parser.add_argument(
        "--config",
        choices=("tiny", "small"),
        default="tiny",
        help="frozen model configuration (D-0013)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="BPE artifact from training/train_bpe.py; omit for byte-level (D-0015)",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--device", default="cpu", help="torch device, e.g. cuda on a T4"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA autocast fp16 with GradScaler (requires --device cuda)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = config_by_name(args.config)
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        raise SystemExit("--fp16 requires --device cuda")
    artifact = None
    if args.tokenizer is not None:
        artifact = load_artifact(args.tokenizer)
        if artifact["vocab_size"] != config.vocab_size:
            raise SystemExit(
                f"tokenizer vocab_size {artifact['vocab_size']} != model vocab_size "
                f"{config.vocab_size}"
            )
    if args.smoke:
        if artifact is not None:
            raise SystemExit("--tokenizer is not valid with --smoke")
        stream = smoke_token_stream(40_000)
        seq_len = config.max_seq_len // 4
        prompt_text = None
    else:
        texts = load_shard_texts(args.shards)
        if artifact is not None:
            stream = bpe_token_stream(texts, artifact)
            prompt_text = "Once upon a time"
        else:
            stream = byte_token_stream(texts)
            prompt_text = "Once upon a time"
        seq_len = args.seq_len
    train_x, train_y, val_x, val_y = make_sequences(stream, seq_len)
    model, state = train(
        config,
        train_x,
        train_y,
        val_x[:200],
        val_y[:200],
        steps=args.steps,
        batch_size=args.batch_size,
        log_every=50,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        device=device,
        use_fp16=args.fp16,
        grad_accum=args.grad_accum,
    )
    first, last = state.loss_history[0], state.loss_history[-1]
    print(f"loss {first:.4f} -> {last:.4f} over {state.step} steps")
    if prompt_text is None:
        prompt = torch.tensor([[64, 65, 66]], device=device)
    elif artifact is not None:
        prompt_ids = encode_content(prompt_text, merge_ranks(artifact["merges"]))
        prompt = torch.tensor([prompt_ids], device=device)
    else:
        prompt = torch.tensor([list(prompt_text.encode("utf-8"))], device=device)
    generated = model.generate(prompt, max_new_tokens=48, temperature=0.0)
    continuation = generated[0, prompt.shape[1] :].tolist()
    if prompt_text is None:
        print("greedy continuation of [64,65,66]:", continuation)
    elif artifact is not None:
        print("greedy continuation ids:", continuation)
    else:
        print("greedy continuation of", repr(prompt_text), "->",
              repr(bytes(continuation).decode("utf-8", errors="replace")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

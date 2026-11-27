"""Deterministic byte-level BPE trainer for BYVA tokenizer artifacts (D-0015).

Trains a ranked merge table directly on Phase 2 shard documents with **no
pre-tokenization boundary**, matching `Tokenizer::encode_bytes`, so training
and serving segment text identically. Produces a versioned artifact
`{version, vocab_size, merges}` loadable by `Tokenizer::from_artifact`
(`training/byva_tokenizer.py` mirrors the validation for tests).

Determinism contract (D-0017):
- The corpus is treated as a set of documents; pair counts are aggregated over
  all documents, so input order never affects the result.
- At each step the merged pair is chosen by (count desc, left asc, right asc).
- Occurrences are replaced left-to-right without overlap, exactly like the
  Rust encoder's linked-list walk.

The implementation is pure Python standard library and is exact (incremental
pair counts, no sampling). Memory scales with corpus size because merge
occurrences are indexed; corpora beyond a few tens of MB need a scaled
trainer (recorded as an open item for the Phase 3/8 hardware runs).
"""

from __future__ import annotations

import argparse
import heapq
import json
import unicodedata
from pathlib import Path

from byva_tokenizer import (
    BYTE_TOKEN_BASE,
    FIRST_MERGE_ID,
    MAX_VOCAB_SIZE,
    MIN_VOCAB_SIZE,
    TokenizerError,
)


def load_shard_texts(shard_dir: Path) -> list[str]:
    """Loads documents from a Phase 2 shard directory (manifest-verified)."""
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        raise TokenizerError(f"no manifest.json in {shard_dir}")
    texts: list[str] = []
    for shard in sorted(shard_dir.glob("shard-*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                texts.append(unicodedata.normalize("NFC", json.loads(line)["text"]))
    return texts


def _initial_sequences(texts: list[str]) -> list[list[int]]:
    return [
        [BYTE_TOKEN_BASE + byte for byte in text.encode("utf-8")]
        for text in texts
        if text
    ]


def train_bpe(
    texts: list[str], target_vocab: int, version: int = 1
) -> dict:
    """Trains a ranked merge table and returns a validated artifact dict.

    Production bounds (32768..49152) are enforced by `validate_merges` on
    load; the trainer itself only requires at least one merge.
    """
    if target_vocab <= FIRST_MERGE_ID:
        raise TokenizerError(
            f"target vocabulary {target_vocab} leaves no room for merges "
            f"(FIRST_MERGE_ID is {FIRST_MERGE_ID})"
        )
    merge_target = target_vocab - FIRST_MERGE_ID
    docs = _initial_sequences(texts)
    if not docs:
        raise TokenizerError("corpus is empty")

    counts: dict[tuple[int, int], int] = {}
    positions: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for doc_index, symbols in enumerate(docs):
        for index in range(len(symbols) - 1):
            pair = (symbols[index], symbols[index + 1])
            counts[pair] = counts.get(pair, 0) + 1
            positions.setdefault(pair, set()).add((doc_index, index))

    heap: list[tuple[int, int, int]] = [
        (-count, left, right) for (left, right), count in counts.items()
    ]
    heapq.heapify(heap)

    merges: list[dict[str, int]] = []
    while len(merges) < merge_target:
        best = _pop_best(heap, counts)
        if best is None:
            raise TokenizerError(
                "corpus exhausted: no pair occurs twice; "
                f"only {len(merges)} merges trainable, "
                f"{merge_target} requested"
            )
        left, right = best
        output = FIRST_MERGE_ID + len(merges)
        merges.append({"left": left, "right": right})
        _apply_merge(docs, counts, positions, heap, left, right, output)

    return {
        "version": version,
        "vocab_size": FIRST_MERGE_ID + len(merges),
        "merges": merges,
    }


def _pop_best(
    heap: list[tuple[int, int, int]], counts: dict[tuple[int, int], int]
) -> tuple[int, int] | None:
    """Pops the (count desc, left asc, right asc) pair with current count >= 2."""
    while heap:
        neg_count, left, right = heapq.heappop(heap)
        current = counts.get((left, right), 0)
        if current < 2:
            counts.pop((left, right), None)
            continue
        if current != -neg_count:
            heapq.heappush(heap, (-current, left, right))
            continue
        return left, right
    return None


def _next_alive(symbols: list[int | None], start: int) -> int:
    index = start
    while index < len(symbols) and symbols[index] is None:
        index += 1
    return index


def _prev_alive(symbols: list[int | None], start: int) -> int:
    index = start
    while index >= 0 and symbols[index] is None:
        index -= 1
    return index


def _bump(
    counts: dict[tuple[int, int], int],
    positions: dict[tuple[int, int], set[tuple[int, int]]],
    heap: list[tuple[int, int, int]],
    pair: tuple[int, int],
    delta: int,
    doc_index: int,
    index: int,
) -> None:
    new_count = counts.get(pair, 0) + delta
    if new_count > 0:
        counts[pair] = new_count
        if delta > 0:
            positions.setdefault(pair, set()).add((doc_index, index))
        else:
            positions.get(pair, set()).discard((doc_index, index))
        heapq.heappush(heap, (-new_count, pair[0], pair[1]))
    else:
        counts.pop(pair, None)
        bucket = positions.get(pair)
        if bucket is not None:
            bucket.discard((doc_index, index))


def _apply_merge(
    docs: list[list[int | None]],
    counts: dict[tuple[int, int], int],
    positions: dict[tuple[int, int], set[tuple[int, int]]],
    heap: list[tuple[int, int, int]],
    left: int,
    right: int,
    output: int,
) -> None:
    """Replaces occurrences of (left, right) left-to-right, updating counts."""
    pair = (left, right)
    for doc_index, index in sorted(positions.get(pair, set())):
        symbols = docs[doc_index]
        if index >= len(symbols) or symbols[index] != left:
            continue
        next_index = _next_alive(symbols, index + 1)
        if next_index >= len(symbols) or symbols[next_index] != right:
            continue
        prev_index = _prev_alive(symbols, index - 1)
        after_index = _next_alive(symbols, next_index + 1)

        symbols[index] = output
        symbols[next_index] = None

        _bump(counts, positions, heap, pair, -1, doc_index, index)
        if prev_index >= 0:
            _bump(
                counts,
                positions,
                heap,
                (symbols[prev_index], left),
                -1,
                doc_index,
                prev_index,
            )
            _bump(
                counts,
                positions,
                heap,
                (symbols[prev_index], output),
                1,
                doc_index,
                prev_index,
            )
        if after_index < len(symbols):
            _bump(
                counts,
                positions,
                heap,
                (right, symbols[after_index]),
                -1,
                doc_index,
                next_index,
            )
            _bump(
                counts,
                positions,
                heap,
                (output, symbols[after_index]),
                1,
                doc_index,
                index,
            )


def artifact_stats(texts: list[str], artifact: dict) -> dict:
    """Byte/token compression stats over the training corpus."""
    from byva_tokenizer import merge_ranks, encode_content

    ranks = merge_ranks(artifact["merges"])
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    total_tokens = sum(len(encode_content(text, ranks)) for text in texts)
    return {
        "documents": len(texts),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else 0.0,
        "merges": len(artifact["merges"]),
        "vocab_size": artifact["vocab_size"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True,
                        help="shard directory built by training/build_shards.py")
    parser.add_argument("--target-vocab", type=int, default=32_768)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    if not MIN_VOCAB_SIZE <= args.target_vocab <= MAX_VOCAB_SIZE:
        raise SystemExit(
            f"--target-vocab must be within {MIN_VOCAB_SIZE}..{MAX_VOCAB_SIZE}"
        )
    texts = load_shard_texts(args.shards)
    artifact = train_bpe(texts, args.target_vocab, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    stats = artifact_stats(texts, artifact)
    print(json.dumps(stats, sort_keys=True))
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(
            json.dumps(stats, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Python mirror of the crates/tokenizer BPE runtime (training-time only).

Must stay semantically identical to the Rust implementation: NFC normalization,
byte ids at BYTE_TOKEN_BASE..BYTE_TOKEN_BASE+255, ranked merges applied through
a lazily invalidated priority queue without any pre-tokenization boundary
(matches `Tokenizer::encode_bytes`), and decode via merge-child expansion
(matches `Tokenizer::token_bytes`). Used by train_bpe.py to build artifacts and
by train.py to encode shards.
"""

from __future__ import annotations

import heapq
import json
import unicodedata
from pathlib import Path

SPECIAL_TOKEN_COUNT = 15
RESERVED_TOKEN_COUNT = 32
BYTE_TOKEN_BASE = SPECIAL_TOKEN_COUNT + RESERVED_TOKEN_COUNT
BYTE_TOKEN_COUNT = 256
FIRST_MERGE_ID = BYTE_TOKEN_BASE + BYTE_TOKEN_COUNT
MIN_VOCAB_SIZE = 32_768
MAX_VOCAB_SIZE = 49_152


class TokenizerError(ValueError):
    """Mirrors the typed Rust TokenizerError variants used here."""


def validate_merges(merges: list[dict[str, int]], vocab_size: int) -> None:
    """Applies the same validation as `Tokenizer::build` in the Rust crate."""
    if not MIN_VOCAB_SIZE <= vocab_size <= MAX_VOCAB_SIZE:
        raise TokenizerError(
            f"tokenizer vocabulary size {vocab_size} is outside the supported range"
        )
    if vocab_size != FIRST_MERGE_ID + len(merges):
        raise TokenizerError(
            "artifact vocabulary size does not match the number of merge entries"
        )
    seen: set[tuple[int, int]] = set()
    for index, merge in enumerate(merges):
        left, right = merge["left"], merge["right"]
        output_id = FIRST_MERGE_ID + index
        if (
            left < BYTE_TOKEN_BASE
            or right < BYTE_TOKEN_BASE
            or left >= output_id
            or right >= output_id
        ):
            raise TokenizerError(
                f"merge at index {index} refers to a non-existent or non-prior token"
            )
        if (left, right) in seen:
            raise TokenizerError(f"duplicate BPE merge pair at index {index}")
        seen.add((left, right))


def load_artifact(path: Path) -> dict:
    """Loads a `{version, vocab_size, merges}` artifact JSON with full validation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("version", "vocab_size", "merges"):
        if key not in payload:
            raise TokenizerError(f"artifact is missing {key!r}")
    validate_merges(payload["merges"], payload["vocab_size"])
    return payload


def merge_ranks(merges: list[dict[str, int]]) -> dict[tuple[int, int], int]:
    return {
        (merge["left"], merge["right"]): FIRST_MERGE_ID + index
        for index, merge in enumerate(merges)
    }


def _merge_children(merges: list[dict[str, int]]) -> dict[int, tuple[int, int]]:
    return {
        FIRST_MERGE_ID + index: (merge["left"], merge["right"])
        for index, merge in enumerate(merges)
    }


def encode_bytes(bytes_data: bytes, ranks: dict[tuple[int, int], int]) -> list[int]:
    """Ranked BPE over the raw byte stream, identical to the Rust encoder.

    Mirrors `Tokenizer::encode_bytes`: a doubly linked list of nodes plus a
    priority queue of (rank, left, right, left_generation, right_generation)
    entries, popped lowest-rank first and skipped when stale.
    """
    if not bytes_data:
        return []
    tokens = [BYTE_TOKEN_BASE + byte for byte in bytes_data]
    previous = [index - 1 for index in range(len(tokens))]
    next = [index + 1 if index + 1 < len(tokens) else -1 for index in range(len(tokens))]
    alive = [True] * len(tokens)
    generation = [0] * len(tokens)

    queue: list[tuple[int, int, int, int, int]] = []

    def queue_pair(left: int) -> None:
        right = next[left]
        if right < 0 or not alive[left] or not alive[right]:
            return
        rank = ranks.get((tokens[left], tokens[right]))
        if rank is not None:
            heapq.heappush(
                queue, (rank, left, right, generation[left], generation[right])
            )

    for left in range(len(tokens) - 1):
        queue_pair(left)

    while queue:
        rank, left, right, left_generation, right_generation = heapq.heappop(queue)
        if not alive[left] or not alive[right]:
            continue
        if next[left] != right:
            continue
        if generation[left] != left_generation or generation[right] != right_generation:
            continue
        current = ranks.get((tokens[left], tokens[right]))
        if current != rank:
            continue

        after = next[right]
        tokens[left] = rank
        next[left] = after
        generation[left] += 1
        alive[right] = False
        generation[right] += 1
        if after >= 0:
            previous[after] = left
        if previous[left] >= 0:
            queue_pair(previous[left])
        queue_pair(left)

    return [tokens[index] for index in range(len(tokens)) if alive[index]]


def encode_content(text: str, ranks: dict[tuple[int, int], int]) -> list[int]:
    """NFC-normalizes, then encodes with byte tokens and ranked merges."""
    normalized = unicodedata.normalize("NFC", text)
    return encode_bytes(normalized.encode("utf-8"), ranks)


def token_bytes(token: int, children: dict[int, tuple[int, int]]) -> bytes:
    """Expands a content token to its original bytes, mirroring `token_bytes`."""
    if token >= FIRST_MERGE_ID:
        pair = children.get(token)
        if pair is None:
            raise TokenizerError(f"token ID {token} is outside this tokenizer vocabulary")
        return token_bytes(pair[0], children) + token_bytes(pair[1], children)
    if token >= BYTE_TOKEN_BASE:
        byte = token - BYTE_TOKEN_BASE
        if byte >= BYTE_TOKEN_COUNT:
            raise TokenizerError(f"token ID {token} is outside this tokenizer vocabulary")
        return bytes([byte])
    raise TokenizerError(
        f"control/reserved token ID {token} cannot be decoded as text content"
    )


def decode_content(ids: list[int], merges: list[dict[str, int]]) -> str:
    """Decode content tokens back to text, rejecting control/reserved ids."""
    children = _merge_children(merges)
    payload = bytearray()
    for token in ids:
        payload.extend(token_bytes(token, children))
    return payload.decode("utf-8")

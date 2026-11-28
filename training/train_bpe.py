"""Deterministic BPE trainer for the BYVA tokenizer.

Trains a byte-level BPE table on Phase 2 shards and emits a JSON artifact that
loads directly into the Rust `byva_tokenizer::Tokenizer` via
`TokenizerArtifact` (fields: version, vocab_size, merges[{left, right}]).

Token ID layout matches crates/tokenizer/src/lib.rs:
15 special + 32 reserved + 256 byte tokens, merges start at FIRST_MERGE_ID.

Determinism: the corpus is reduced to sorted word-type frequencies before
training, so the artifact depends only on the document set. Merge selection
uses (count desc, left id asc, right id asc) tie-breaking, making every merge
step deterministic. Output JSON is canonical (sorted keys, compact separators).
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from byva_data import canonical_json, file_sha256

SPECIAL_TOKEN_COUNT = 15
RESERVED_TOKEN_COUNT = 32
BYTE_TOKEN_BASE = SPECIAL_TOKEN_COUNT + RESERVED_TOKEN_COUNT
BYTE_TOKEN_COUNT = 256
FIRST_MERGE_ID = BYTE_TOKEN_BASE + BYTE_TOKEN_COUNT
MIN_VOCAB_SIZE = 32_768
MAX_VOCAB_SIZE = 49_152
ARTIFACT_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MIN_PAIR_COUNT = 2

PRETOKENIZER = re.compile(r"[^\W\d_]+|\d+|\s+|.", re.DOTALL)

PRETOKENIZER_DESCRIPTION = "regex:[^\\W\\d_]+|\\d+|\\s+|. (Unicode letter runs, digit runs, whitespace runs, single other characters)"
TIE_BREAK_DESCRIPTION = "highest pair count, then lowest left token id, then lowest right token id"


class BpeError(ValueError):
    """Raised when BPE training cannot satisfy its configuration."""


def pretokenize(text: str) -> list[str]:
    return PRETOKENIZER.findall(text)


def canonical_artifact_json(artifact: dict) -> str:
    return canonical_json(artifact)


def validate_vocab_size(vocab_size: int) -> int:
    if not MIN_VOCAB_SIZE <= vocab_size <= MAX_VOCAB_SIZE:
        raise BpeError(
            f"vocab_size {vocab_size} outside the supported range "
            f"{MIN_VOCAB_SIZE}..={MAX_VOCAB_SIZE}"
        )
    return vocab_size


def target_merge_count(vocab_size: int) -> int:
    return vocab_size - FIRST_MERGE_ID


class BpeTrainer:
    """Counts word types, then learns merges with a lazy-selection heap."""

    def __init__(self, target_merges: int):
        if target_merges < 1:
            raise BpeError(f"target merge count must be positive, got {target_merges}")
        self.target_merges = target_merges
        self.word_types: Counter[bytes] = Counter()
        self.word_tokens: list[list[int]] = []
        self.word_freqs: list[int] = []
        self.pair_counts: dict[tuple[int, int], int] = {}
        self.pair_words: dict[tuple[int, int], set[int]] = {}
        self.heap: list[tuple[int, int, int]] = []
        self.merges: list[tuple[int, int]] = []

    def add_text(self, text: str) -> None:
        for piece in pretokenize(text):
            self.word_types[piece.encode("utf-8")] += 1

    def prepare(self) -> dict:
        if not self.word_types:
            raise BpeError("corpus is empty; no word types were collected")
        self.word_tokens = []
        self.word_freqs = []
        for word_bytes in sorted(self.word_types):
            self.word_tokens.append([BYTE_TOKEN_BASE + byte for byte in word_bytes])
            self.word_freqs.append(self.word_types[word_bytes])
        self.word_types = Counter()

        self.pair_counts = {}
        self.pair_words = {}
        for index, (tokens, freq) in enumerate(zip(self.word_tokens, self.word_freqs)):
            for pair in zip(tokens, tokens[1:]):
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + freq
                self.pair_words.setdefault(pair, set()).add(index)
        for pair, count in self.pair_counts.items():
            if count >= MIN_PAIR_COUNT:
                heapq.heappush(self.heap, (-count, pair[0], pair[1]))

        word_instances = sum(self.word_freqs)
        token_instances = sum(
            len(tokens) * freq for tokens, freq in zip(self.word_tokens, self.word_freqs)
        )
        return {
            "word_types": len(self.word_tokens),
            "word_instances": word_instances,
            "byte_instances": token_instances,
        }

    def train(self) -> None:
        while len(self.merges) < self.target_merges:
            pair = self._select_pair()
            if pair is None:
                raise BpeError(
                    "corpus exhausted after "
                    f"{len(self.merges)} merges; cannot reach {self.target_merges} "
                    "(no pair occurs at least MIN_PAIR_COUNT times)"
                )
            self._apply_merge(pair)

    def _select_pair(self) -> tuple[int, int] | None:
        while self.heap:
            neg_count, left, right = heapq.heappop(self.heap)
            pair = (left, right)
            if self.pair_counts.get(pair) == -neg_count and self.pair_words.get(pair):
                return pair
        return None

    def _apply_merge(self, pair: tuple[int, int]) -> None:
        left, right = pair
        new_id = FIRST_MERGE_ID + len(self.merges)
        for word_index in sorted(self.pair_words.get(pair, ())):
            self._merge_in_word(word_index, left, right, new_id)
        self.pair_counts.pop(pair, None)
        self.pair_words.pop(pair, None)
        self.merges.append(pair)

    def _merge_in_word(self, index: int, left: int, right: int, new_id: int) -> None:
        freq = self.word_freqs[index]
        tokens = self.word_tokens[index]
        merged = []
        position = 0
        length = len(tokens)
        while position < length:
            if (
                position + 1 < length
                and tokens[position] == left
                and tokens[position + 1] == right
            ):
                merged.append(new_id)
                position += 2
            else:
                merged.append(tokens[position])
                position += 1
        self.word_tokens[index] = merged

        old_pairs = Counter(zip(tokens, tokens[1:]))
        new_pairs = Counter(zip(merged, merged[1:]))
        for pair in (old_pairs.keys() | new_pairs.keys()):
            delta = (new_pairs.get(pair, 0) - old_pairs.get(pair, 0)) * freq
            if delta:
                count = self.pair_counts.get(pair, 0) + delta
                if count >= MIN_PAIR_COUNT:
                    self.pair_counts[pair] = count
                    heapq.heappush(self.heap, (-count, pair[0], pair[1]))
                else:
                    self.pair_counts.pop(pair, None)
            contained = pair in new_pairs
            members = self.pair_words.get(pair)
            if contained:
                if members is None:
                    self.pair_words[pair] = {index}
                else:
                    members.add(index)
            elif members is not None:
                members.discard(index)
                if not members:
                    del self.pair_words[pair]

    def artifact(self, vocab_size: int) -> dict:
        if len(self.merges) != target_merge_count(vocab_size):
            raise BpeError(
                f"learned {len(self.merges)} merges but vocab_size {vocab_size} "
                f"requires {target_merge_count(vocab_size)}"
            )
        return {
            "version": ARTIFACT_VERSION,
            "vocab_size": vocab_size,
            "merges": [{"left": left, "right": right} for left, right in self.merges],
        }

    def token_instance_stats(self) -> dict:
        token_instances = sum(
            len(tokens) * freq for tokens, freq in zip(self.word_tokens, self.word_freqs)
        )
        return {"token_instances": token_instances}


class BpeEncoder:
    """Applies trained merges to text; used for validation and tests."""

    def __init__(self, artifact: dict):
        merges = [(merge["left"], merge["right"]) for merge in artifact["merges"]]
        self.vocab_size = artifact["vocab_size"]
        self.ranks: dict[tuple[int, int], int] = {}
        self.children: dict[int, tuple[int, int]] = {}
        for index, (left, right) in enumerate(merges):
            output = FIRST_MERGE_ID + index
            self.ranks[(left, right)] = output
            self.children[output] = (left, right)

    def encode_word(self, word: str) -> list[int]:
        tokens = [BYTE_TOKEN_BASE + byte for byte in word.encode("utf-8")]
        while len(tokens) >= 2:
            best_pair = None
            best_rank = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            left, right = best_pair
            output = self.ranks[best_pair]
            merged = []
            position = 0
            while position < len(tokens):
                if (
                    position + 1 < len(tokens)
                    and tokens[position] == left
                    and tokens[position + 1] == right
                ):
                    merged.append(output)
                    position += 2
                else:
                    merged.append(tokens[position])
                    position += 1
            tokens = merged
        return tokens

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pretokenize(text):
            ids.extend(self.encode_word(piece))
        return ids

    def token_bytes(self, token: int, out: list[int]) -> None:
        children = self.children.get(token)
        if children is None:
            byte = token - BYTE_TOKEN_BASE
            if not 0 <= byte < BYTE_TOKEN_COUNT:
                raise BpeError(f"token {token} is not a byte or merge token")
            out.append(byte)
            return
        left, right = children
        self.token_bytes(left, out)
        self.token_bytes(right, out)

    def decode(self, tokens: list[int]) -> str:
        byte_values: list[int] = []
        for token in tokens:
            self.token_bytes(token, byte_values)
        return bytes(byte_values).decode("utf-8")


def load_shard_texts(shard_dir: Path) -> tuple[list[str], dict]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BpeError(f"missing shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {shard["name"]: shard["sha256"] for shard in manifest.get("shards", [])}
    texts: list[str] = []
    for name in sorted(expected):
        path = shard_dir / name
        if file_sha256(path) != expected[name]:
            raise BpeError(f"shard checksum mismatch: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise BpeError(f"{path}: record must be an object with a text field")
            texts.append(record["text"])
    return texts, manifest


def build_manifest(
    artifact_path: Path,
    artifact: dict,
    vocab_size: int,
    corpus: dict,
    prepare_stats: dict,
    final_stats: dict,
    elapsed_seconds: int,
) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact": artifact_path.name,
        "artifact_sha256": file_sha256(artifact_path),
        "config": {
            "vocab_size": vocab_size,
            "target_merges": target_merge_count(vocab_size),
            "pretokenizer": PRETOKENIZER_DESCRIPTION,
            "tie_break": TIE_BREAK_DESCRIPTION,
            "min_pair_count": MIN_PAIR_COUNT,
        },
        "corpus": corpus,
        "stats": {**prepare_stats, **final_stats},
        "training_seconds": elapsed_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--shards", type=Path, help="shard directory with manifest.json")
    source_group.add_argument(
        "--input", action="append", type=Path, help="raw JSONL input (repeatable)"
    )
    parser.add_argument("--vocab-size", type=int, default=MIN_VOCAB_SIZE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    vocab_size = validate_vocab_size(args.vocab_size)
    trainer = BpeTrainer(target_merge_count(vocab_size))

    if args.shards is not None:
        texts, shard_manifest = load_shard_texts(args.shards)
        corpus = {
            "kind": "phase2-shards",
            "manifest_sha256": file_sha256(args.shards / "manifest.json"),
            "sources": shard_manifest.get("sources", []),
            "docs": shard_manifest.get("stats", {}).get("kept"),
        }
    else:
        texts = []
        source_entries = []
        for source in args.input:
            if not source.is_file():
                parser.error(f"input file not found: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise BpeError(f"{source}: record must be an object with a text field")
                texts.append(record["text"])
            source_entries.append({"name": source.name, "sha256": file_sha256(source)})
        corpus = {
            "kind": "raw-jsonl",
            "sources": sorted(source_entries, key=lambda entry: entry["name"]),
        }

    for text in texts:
        trainer.add_text(text)

    started = time.monotonic()
    prepare_stats = trainer.prepare()
    print(f"word types: {prepare_stats['word_types']}", flush=True)
    trainer.train()
    elapsed = int(time.monotonic() - started)

    artifact = trainer.artifact(vocab_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(artifact) + "\n", encoding="utf-8")

    final_stats = trainer.token_instance_stats()
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = build_manifest(
        args.output, artifact, vocab_size, corpus, prepare_stats, final_stats, elapsed
    )
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    print(f"merges learned: {len(trainer.merges)}")
    print(f"artifact: {args.output}")
    print(f"manifest: {manifest_path}")
    print(f"training seconds: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

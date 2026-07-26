#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Prepare leakage-safe text-distillation prompts from BONES-SEED v004.

The input is read with Python's :mod:`csv` module; pandas is intentionally not
required.  The seven human-description columns are Unicode/whitespace
normalised, empty values are discarded, and duplicate prompt text is emitted
only once.  A stable, seeded hash of the ``content_name`` family assigns every
member of a motion group to exactly one of train/validation/test.

Example:

    uv run python scripts/minilm/prepare_prompts.py \
        --input datasets/bones-seed/metadata/seed_metadata_v004.csv \
        --output artifacts/minilm/prompts.jsonl \
        --seed 2026 --sample-size 50000

Each output line is a JSON object with ``text``, ``split``, ``group``, and
``source`` keys.  ``--sample-size 0`` (the default) keeps all unique prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DESCRIPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)
GROUP_COLUMNS = ("content_name", "take_org_name", "take_name", "move_name")
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_INPUT = Path("datasets/bones-seed/metadata/seed_metadata_v004.csv")
DEFAULT_OUTPUT = Path("artifacts/minilm/prompts.jsonl")

_WHITESPACE_RE = re.compile(r"\s+")
_MIRROR_SUFFIX_RE = re.compile(r"_M$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PromptRecord:
    """One JSONL row."""

    text: str
    split: str
    group: str
    source: str


def normalize_text(value: str | None) -> str:
    """Return NFKC-normalised text with surrounding/repeated whitespace removed."""

    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def content_group(row: Mapping[str, str | None], row_number: int) -> str:
    """Return a stable content-family key, coalescing mirrored ``*_M`` clips."""

    for column in GROUP_COLUMNS:
        value = normalize_text(row.get(column))
        if not value:
            continue
        if column == "content_name":
            value = _MIRROR_SUFFIX_RE.sub("", value)
        # Identifiers are case-insensitive for grouping.  Preserve separators:
        # direction/angle tokens carry motion semantics and should not be merged.
        return value.casefold()
    raise ValueError(f"CSV row {row_number} has none of the grouping columns populated: {', '.join(GROUP_COLUMNS)}")


def split_for_group(group: str, seed: int, ratios: Sequence[float]) -> str:
    """Map ``group`` to a split using a platform-independent seeded SHA-256 hash."""

    digest = hashlib.sha256(f"ardy-minilm-split-v1\0{seed}\0{group}".encode()).digest()
    unit_value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    train_threshold = ratios[0]
    val_threshold = ratios[0] + ratios[1]
    if unit_value < train_threshold:
        return "train"
    if unit_value < val_threshold:
        return "val"
    return "test"


def _validate_ratios(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("--split-ratios requires exactly three values: TRAIN VAL TEST")
    ratios = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError("split ratios must be finite and non-negative")
    total = sum(ratios)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("split-ratio sum must be finite and positive")
    # Accept weights as well as already-normalised probabilities.
    return tuple(value / total for value in ratios)  # type: ignore[return-value]


def extract_prompts(
    csv_path: Path,
    *,
    seed: int,
    split_ratios: Sequence[float],
) -> list[PromptRecord]:
    """Extract and globally deduplicate normalised prompts from ``csv_path``."""

    ratios = _validate_ratios(split_ratios)
    # text -> deterministic representative.  Global text deduplication also
    # prevents a generic description shared by two content groups from
    # appearing in different splits.
    selected: dict[str, PromptRecord] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header")
        missing = [column for column in DESCRIPTION_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        if not any(column in reader.fieldnames for column in GROUP_COLUMNS):
            raise ValueError(f"{csv_path} needs at least one grouping column: {', '.join(GROUP_COLUMNS)}")

        for row_number, row in enumerate(reader, start=2):
            group = content_group(row, row_number)
            split = split_for_group(group, seed, ratios)
            for source in DESCRIPTION_COLUMNS:
                text = normalize_text(row.get(source))
                if not text:
                    continue
                candidate = PromptRecord(text=text, split=split, group=group, source=source)
                previous = selected.get(text)
                if previous is None or (candidate.group, candidate.source) < (
                    previous.group,
                    previous.source,
                ):
                    selected[text] = candidate

    return sorted(
        selected.values(),
        key=lambda record: (
            SPLIT_NAMES.index(record.split),
            record.group,
            record.text,
            record.source,
        ),
    )


def deterministic_sample(
    records: Sequence[PromptRecord],
    *,
    sample_size: int,
    seed: int,
) -> list[PromptRecord]:
    """Return a seeded hash sample, retaining every available split when possible."""

    if sample_size < 0:
        raise ValueError("--sample-size must be zero or positive")
    if sample_size == 0 or sample_size >= len(records):
        return list(records)

    def rank(index: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(
                (
                    f"ardy-minilm-sample-v1\0{seed}\0{records[index].group}\0"
                    f"{records[index].source}\0{records[index].text}"
                ).encode()
            ).digest(),
            index,
        )

    # Reserve one example per non-empty split whenever the requested size
    # permits it.  A purely global sample is overwhelmingly train-only at
    # small sizes and cannot be consumed by the downstream trainer.
    by_split = {
        split: [index for index, record in enumerate(records) if record.split == split] for split in SPLIT_NAMES
    }
    nonempty_splits = [split for split in SPLIT_NAMES if by_split[split]]
    selected_indices: set[int] = set()
    if sample_size >= len(nonempty_splits):
        for split in nonempty_splits:
            selected_indices.add(min(by_split[split], key=rank))

    remaining = sorted(
        (index for index in range(len(records)) if index not in selected_indices),
        key=rank,
    )
    selected_indices.update(remaining[: sample_size - len(selected_indices)])
    return [record for index, record in enumerate(records) if index in selected_indices]


def write_jsonl(records: Sequence[PromptRecord], output_path: Path) -> None:
    """Atomically write ``records`` as UTF-8 JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in records:
                output_file.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
                output_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the seven BONES-SEED v004 description columns into a "
            "deterministic, leakage-safe JSONL prompt dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="BONES-SEED v004 metadata CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination JSONL")
    parser.add_argument("--seed", type=int, default=2026, help="split and sampling seed")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.90, 0.05, 0.05),
        help="split probabilities (or proportional non-negative weights)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="number of unique prompts to retain; zero keeps all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ratios = _validate_ratios(args.split_ratios)
        records = extract_prompts(args.input, seed=args.seed, split_ratios=ratios)
        unique_count = len(records)
        records = deterministic_sample(records, sample_size=args.sample_size, seed=args.seed)
        missing_splits = [split for split in SPLIT_NAMES if not any(record.split == split for record in records)]
        if missing_splits:
            raise ValueError(
                "prepared training data has no "
                f"{'/'.join(missing_splits)} examples; use a sample size of at "
                f"least {len(SPLIT_NAMES)} and positive split ratios"
            )
        write_jsonl(records, args.output)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    split_counts = {split: sum(record.split == split for record in records) for split in SPLIT_NAMES}
    group_splits: dict[str, str] = {}
    for record in records:
        old_split = group_splits.setdefault(record.group, record.split)
        if old_split != record.split:  # defensive invariant check
            raise RuntimeError(f"group {record.group!r} appears in both {old_split} and {record.split}")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "unique_before_sampling": unique_count,
                "written": len(records),
                "groups": len(group_splits),
                "splits": split_counts,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

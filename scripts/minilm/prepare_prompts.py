#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Prepare a leakage-resistant MiniLM corpus from NVIDIA timeline annotations.

The source is the pinned ``timelines.jsonl`` file from
``nvidia/SEED-Timeline-Annotations``.  Overview descriptions and atomic event
descriptions are Unicode/whitespace normalized and globally deduplicated.
Recordings that share an actor/take-independent motion family, including
``propagated_from_filename`` links, are assigned to one deterministic split.

The source revision and SHA-256 are intentionally fixed in code.  Supplying
``--input`` avoids a Hub download but does not bypass the source hash check.
Alongside the prompt JSONL, the script writes a provenance sidecar containing
the dataset identity, extraction policy, counts, and prompt-manifest hash.

Example:

    uv run python scripts/minilm/prepare_prompts.py \
        --output artifacts/data/prompts-core40-timeline.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from ardy.minilm_teacher_cache import (
    TIMELINE_PROMPT_DEDUPLICATION,
    TIMELINE_PROMPT_GROUPING,
    TIMELINE_PROMPT_MAX_CHARACTERS,
    TIMELINE_PROMPT_NORMALIZATION,
    TIMELINE_SPLIT_HASH_NAMESPACE,
    normalize_timeline_prompt,
    timeline_prompt_deduplication_key,
)

DATASET_REPO = "nvidia/SEED-Timeline-Annotations"
DATASET_REVISION = "b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f"
DATASET_FILENAME = "timelines.jsonl"
DATASET_SHA256 = "379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8"
DATASET_SIZE_BYTES = 80_373_523
DATASET_LICENSE = "CC BY 4.0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}"
DEFAULT_OUTPUT = Path("artifacts/data/prompts-core40-timeline.jsonl")
SPLIT_NAMES = ("train", "val", "test")
SOURCE_NAMES = ("overview_description", "events.description")
MAX_PROMPT_CHARACTERS = TIMELINE_PROMPT_MAX_CHARACTERS

_CLIP_NAME_RE = re.compile(r"^(?P<take>.+)__a\d+(?:_m)?$", flags=re.IGNORECASE)
_TAKE_SUFFIX_RE = re.compile(r"_(?P<suffix>\d{3})$")
_DIRECTION_ANGLE_TOKENS = frozenset(f"{angle:03d}" for angle in range(0, 361, 45))


@dataclass(frozen=True)
class PromptRecord:
    """One prompt-manifest row."""

    text: str
    split: str
    group: str
    source: str


@dataclass(frozen=True)
class TimelineRow:
    """The annotation fields consumed from one source JSONL row."""

    filename: str
    propagated_from_filename: str | None
    overview_description: str
    event_descriptions: tuple[str, ...]


class DisjointSet:
    """Small deterministic union-find used for propagated recording families."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._minimum: dict[str, str] = {}

    def add(self, value: str) -> None:
        if value not in self._parent:
            self._parent[value] = value
            self._minimum[value] = value

    def find(self, value: str) -> str:
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent, child = left_root, right_root
        else:
            parent, child = right_root, left_root
        self._parent[child] = parent
        self._minimum[parent] = min(self._minimum[left_root], self._minimum[right_root])
        self._minimum.pop(child)

    def canonical(self, value: str) -> str:
        return self._minimum[self.find(value)]

    @property
    def item_count(self) -> int:
        return len(self._parent)

    @property
    def component_count(self) -> int:
        return len({self.find(value) for value in self._parent})


def normalize_text(value: str | None) -> str:
    """Return NFKC-normalized text with repeated whitespace collapsed."""

    return normalize_timeline_prompt(value)


def prompt_deduplication_key(text: str) -> str:
    """Return a case- and punctuation-insensitive prompt identity key.

    Timeline annotations contain surface variants that differ only in commas,
    terminal punctuation, or hyphenation. Treating punctuation as a word
    boundary keeps those near-duplicates from crossing dataset splits while
    preserving distinct alphanumeric wording.
    """

    return timeline_prompt_deduplication_key(text)


def motion_family(filename: str) -> str:
    """Remove mirror, actor, and terminal take identifiers from a filename.

    Timeline rows include multiple actors (``__A017``), mirrored clips
    (``_M``), and terminal numeric take identifiers.  These variants describe
    the same underlying motion concept and must not straddle data splits.
    Direction and angle tokens elsewhere in the name remain significant.
    """

    normalized = normalize_text(filename).casefold()
    if not normalized:
        raise ValueError("timeline filename must be a non-empty string")
    match = _CLIP_NAME_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(f"timeline filename does not match <take>__A<actor>[_M]: {filename!r}")
    family = match.group("take")
    take_suffix = _TAKE_SUFFIX_RE.search(family)
    if take_suffix is not None and take_suffix.group("suffix") not in _DIRECTION_ANGLE_TOKENS:
        family = family[: take_suffix.start()]
    if not family:
        raise ValueError(f"timeline filename has no stable motion family: {filename!r}")
    return family


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(input_path: Path | None) -> tuple[Path, str]:
    """Resolve the pinned source file and verify its published SHA-256."""

    if input_path is None:
        path = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=DATASET_FILENAME,
                revision=DATASET_REVISION,
            )
        )
        resolved_from = "hugging_face_hub"
    else:
        path = input_path
        resolved_from = "local_input"
    actual_size = path.stat().st_size
    if actual_size != DATASET_SIZE_BYTES:
        raise ValueError(
            f"{path} does not match pinned {DATASET_REPO}@{DATASET_REVISION}/{DATASET_FILENAME}: "
            f"expected {DATASET_SIZE_BYTES} bytes, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != DATASET_SHA256:
        raise ValueError(
            f"{path} does not match pinned {DATASET_REPO}@{DATASET_REVISION}/{DATASET_FILENAME}: "
            f"expected SHA-256 {DATASET_SHA256}, got {actual_sha256}"
        )
    return path, resolved_from


def _require_string(value: Any, *, path: Path, line_number: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
    return value


def iter_timeline_rows(path: Path) -> Iterator[TimelineRow]:
    """Strictly parse the annotation fields used by the corpus builder."""

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            expected_row_keys = {
                "overview_description",
                "num_events",
                "events",
                "filename",
                "propagated_from_filename",
            }
            if set(value) != expected_row_keys:
                raise ValueError(
                    f"{path}:{line_number}: expected fields {sorted(expected_row_keys)}, got {sorted(value)}"
                )

            filename = _require_string(
                value.get("filename"),
                path=path,
                line_number=line_number,
                field="filename",
            )
            propagated = value.get("propagated_from_filename")
            if propagated is not None:
                propagated = _require_string(
                    propagated,
                    path=path,
                    line_number=line_number,
                    field="propagated_from_filename",
                )
            overview = _require_string(
                value.get("overview_description"),
                path=path,
                line_number=line_number,
                field="overview_description",
            )
            events = value.get("events")
            if not isinstance(events, list):
                raise TypeError(f"{path}:{line_number}: events must be a list")
            num_events = value.get("num_events")
            if not isinstance(num_events, int) or isinstance(num_events, bool) or num_events != len(events):
                raise ValueError(
                    f"{path}:{line_number}: num_events must equal the events length ({num_events!r} != {len(events)})"
                )

            event_descriptions: list[str] = []
            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    raise TypeError(f"{path}:{line_number}: events[{event_index}] must be an object")
                expected_event_keys = {"start_time", "end_time", "description"}
                if set(event) != expected_event_keys:
                    raise ValueError(
                        f"{path}:{line_number}: events[{event_index}] expected fields "
                        f"{sorted(expected_event_keys)}, got {sorted(event)}"
                    )
                start_time = event["start_time"]
                end_time = event["end_time"]
                if (
                    not isinstance(start_time, (int, float))
                    or isinstance(start_time, bool)
                    or not isinstance(end_time, (int, float))
                    or isinstance(end_time, bool)
                    or not math.isfinite(start_time)
                    or not math.isfinite(end_time)
                    or start_time < 0.0
                    or end_time <= start_time
                ):
                    raise ValueError(
                        f"{path}:{line_number}: events[{event_index}] must have finite, "
                        "ordered, non-negative start_time/end_time"
                    )
                event_descriptions.append(
                    _require_string(
                        event.get("description"),
                        path=path,
                        line_number=line_number,
                        field=f"events[{event_index}].description",
                    )
                )
            yield TimelineRow(
                filename=filename,
                propagated_from_filename=propagated,
                overview_description=overview,
                event_descriptions=tuple(event_descriptions),
            )


def build_recording_components(path: Path) -> tuple[DisjointSet, int, int]:
    """Build connected components from normalized recording-family links."""

    components = DisjointSet()
    row_count = 0
    filenames: set[str] = set()
    propagation_references: set[str] = set()
    for row in iter_timeline_rows(path):
        row_count += 1
        normalized_filename = normalize_text(row.filename).casefold()
        if normalized_filename in filenames:
            raise ValueError(f"{path} contains duplicate filename {row.filename!r}")
        filenames.add(normalized_filename)
        family = motion_family(row.filename)
        components.add(family)
        if row.propagated_from_filename is not None:
            propagation_references.add(normalize_text(row.propagated_from_filename).casefold())
            propagated_family = motion_family(row.propagated_from_filename)
            components.union(family, propagated_family)
    if row_count == 0:
        raise ValueError(f"{path} contains no timeline records")
    missing_references = len(propagation_references - filenames)
    return components, row_count, missing_references


def split_for_group(group: str, seed: int, ratios: Sequence[float]) -> str:
    """Map a connected motion group to a platform-independent seeded split."""

    digest = hashlib.sha256(f"{TIMELINE_SPLIT_HASH_NAMESPACE}\0{seed}\0{group}".encode()).digest()
    unit_value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit_value < ratios[0]:
        return "train"
    if unit_value < ratios[0] + ratios[1]:
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
    return tuple(value / total for value in ratios)  # type: ignore[return-value]


def extract_prompts(
    jsonl_path: Path,
    *,
    seed: int,
    split_ratios: Sequence[float],
) -> tuple[list[PromptRecord], dict[str, int]]:
    """Extract and globally deduplicate overview and event descriptions."""

    ratios = _validate_ratios(split_ratios)
    components, row_count, missing_references = build_recording_components(jsonl_path)
    selected: dict[str, PromptRecord] = {}
    raw_source_counts: Counter[str] = Counter()
    dropped_prompt_too_long = 0

    for row in iter_timeline_rows(jsonl_path):
        group = components.canonical(motion_family(row.filename))
        split = split_for_group(group, seed, ratios)
        candidates = (
            ("overview_description", row.overview_description),
            *(("events.description", description) for description in row.event_descriptions),
        )
        for source, raw_text in candidates:
            raw_source_counts[source] += 1
            text = normalize_text(raw_text)
            if len(text) > MAX_PROMPT_CHARACTERS:
                dropped_prompt_too_long += 1
                continue
            candidate = PromptRecord(text=text, split=split, group=group, source=source)
            deduplication_key = prompt_deduplication_key(text)
            previous = selected.get(deduplication_key)
            candidate_priority = (
                SOURCE_NAMES.index(candidate.source),
                candidate.group,
                candidate.text,
            )
            previous_priority = (
                (
                    SOURCE_NAMES.index(previous.source),
                    previous.group,
                    previous.text,
                )
                if previous is not None
                else None
            )
            if previous is None or candidate_priority < previous_priority:
                selected[deduplication_key] = candidate

    records = sorted(
        selected.values(),
        key=lambda record: (
            SPLIT_NAMES.index(record.split),
            record.group,
            record.text,
            record.source,
        ),
    )
    extraction_counts = {
        "timeline_rows": row_count,
        "recording_families": components.item_count,
        "recording_components": components.component_count,
        "missing_propagation_references": missing_references,
        "raw_overview_descriptions": raw_source_counts["overview_description"],
        "raw_event_descriptions": raw_source_counts["events.description"],
        "dropped_prompt_too_long": dropped_prompt_too_long,
        "unique_descriptions": len(records),
    }
    return records, extraction_counts


def deterministic_sample(
    records: Sequence[PromptRecord],
    *,
    sample_size: int,
    seed: int,
) -> list[PromptRecord]:
    """Return a deterministic prompt sample, reserving each non-empty split."""

    if sample_size < 0:
        raise ValueError("--sample-size must be zero or positive")
    if sample_size == 0 or sample_size >= len(records):
        return list(records)

    def rank(index: int) -> tuple[bytes, int]:
        record = records[index]
        return (
            hashlib.sha256(
                f"ardy-minilm-nvidia-timeline-sample-v1\0{seed}\0"
                f"{record.group}\0{record.source}\0{record.text}".encode()
            ).digest(),
            index,
        )

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


def read_prompt_manifest(path: Path) -> list[PromptRecord]:
    """Read and strictly validate a prompt JSONL produced by this script."""

    records: list[PromptRecord] = []
    seen_texts: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            expected_keys = {"text", "split", "group", "source"}
            unknown_keys = set(value) - expected_keys
            missing_keys = expected_keys - set(value)
            if unknown_keys or missing_keys:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly text/split/group/source; "
                    f"missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}"
                )
            if not all(isinstance(value[key], str) and value[key] for key in ("text", "group", "source")):
                raise ValueError(f"{path}:{line_number}: text/group/source must be non-empty strings")
            if value["text"] != normalize_text(value["text"]):
                raise ValueError(f"{path}:{line_number}: text is not in canonical NFKC/whitespace form")
            if len(value["text"]) > MAX_PROMPT_CHARACTERS:
                raise ValueError(f"{path}:{line_number}: text exceeds {MAX_PROMPT_CHARACTERS} characters")
            if value["split"] not in SPLIT_NAMES:
                raise ValueError(f"{path}:{line_number}: invalid split {value['split']!r}")
            if value["source"] not in SOURCE_NAMES:
                raise ValueError(f"{path}:{line_number}: invalid source {value['source']!r}")
            record = PromptRecord(**value)
            deduplication_key = prompt_deduplication_key(record.text)
            if deduplication_key in seen_texts:
                raise ValueError(f"{path}:{line_number}: duplicate prompt text {record.text!r}")
            seen_texts.add(deduplication_key)
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no prompt records")
    return records


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_jsonl(records: Sequence[PromptRecord], output_path: Path) -> None:
    """Atomically write ``records`` as compact UTF-8 JSONL."""

    content = "".join(
        json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n" for record in records
    )
    _atomic_write_text(output_path, content)


def write_metadata(metadata: Mapping[str, Any], output_path: Path) -> None:
    _atomic_write_text(
        output_path,
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def default_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".metadata.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic, leakage-resistant MiniLM prompts from the "
            "pinned NVIDIA SEED Timeline Annotations JSONL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="optional local copy of the pinned timelines.jsonl; its SHA-256 is still verified",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination prompt JSONL")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="provenance sidecar; defaults to OUTPUT with .metadata.json suffix",
    )
    parser.add_argument("--seed", type=int, default=20260726, help="split and sampling seed")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.80, 0.10, 0.10),
        help="split probabilities (or proportional non-negative weights)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="number of unique prompts to retain; zero keeps the full corpus",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_output = args.metadata_output or default_metadata_path(args.output)
    try:
        if args.output.resolve() == metadata_output.resolve():
            raise ValueError("--output and --metadata-output must be different files")
        ratios = _validate_ratios(args.split_ratios)
        source_path, resolved_from = resolve_dataset_path(args.input)
        source_resolved = source_path.resolve()
        if args.output.resolve() == source_resolved:
            raise ValueError("--output must not overwrite the pinned source dataset")
        if metadata_output.resolve() == source_resolved:
            raise ValueError("--metadata-output must not overwrite the pinned source dataset")
        records, extraction_counts = extract_prompts(
            source_path,
            seed=args.seed,
            split_ratios=ratios,
        )
        unique_before_sampling = len(records)
        records = deterministic_sample(records, sample_size=args.sample_size, seed=args.seed)
        missing_splits = [split for split in SPLIT_NAMES if not any(record.split == split for record in records)]
        if missing_splits:
            raise ValueError(
                "prepared training data has no "
                f"{'/'.join(missing_splits)} examples; use a sample size of at "
                f"least {len(SPLIT_NAMES)} and positive split ratios"
            )

        group_splits: dict[str, str] = {}
        for record in records:
            previous_split = group_splits.setdefault(record.group, record.split)
            if previous_split != record.split:
                raise RuntimeError(f"group {record.group!r} appears in both {previous_split} and {record.split}")
        split_counts = Counter(record.split for record in records)
        split_group_counts = Counter(group_splits.values())
        source_counts = Counter(record.source for record in records)

        write_jsonl(records, args.output)
        manifest_sha256 = sha256_file(args.output)
        metadata = {
            "format": "ardy-minilm-prompt-provenance",
            "format_version": 1,
            "dataset": {
                "repo_id": DATASET_REPO,
                "revision": DATASET_REVISION,
                "filename": DATASET_FILENAME,
                "sha256": DATASET_SHA256,
                "size_bytes": DATASET_SIZE_BYTES,
                "resolved_from": resolved_from,
                "owner": "NVIDIA",
                "license": DATASET_LICENSE,
                "url": DATASET_URL,
            },
            "preparation": {
                "sources": list(SOURCE_NAMES),
                "normalization": TIMELINE_PROMPT_NORMALIZATION,
                "deduplication": TIMELINE_PROMPT_DEDUPLICATION,
                "max_prompt_characters": MAX_PROMPT_CHARACTERS,
                "grouping": TIMELINE_PROMPT_GROUPING,
                "split_hash_namespace": TIMELINE_SPLIT_HASH_NAMESPACE,
                "seed": args.seed,
                "split_ratios": {split: ratio for split, ratio in zip(SPLIT_NAMES, ratios, strict=True)},
                "sample_size": args.sample_size,
            },
            "counts": {
                **extraction_counts,
                "unique_before_sampling": unique_before_sampling,
                "written": len(records),
                "groups_written": len(group_splits),
                "splits": {split: split_counts.get(split, 0) for split in SPLIT_NAMES},
                "split_groups": {split: split_group_counts.get(split, 0) for split in SPLIT_NAMES},
                "sources": {source: source_counts.get(source, 0) for source in SOURCE_NAMES},
            },
            "manifest": {
                "filename": args.output.name,
                "sha256": manifest_sha256,
            },
        }
        write_metadata(metadata, metadata_output)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "dataset": f"{DATASET_REPO}@{DATASET_REVISION}",
                "input_sha256": DATASET_SHA256,
                "output": str(args.output),
                "output_sha256": manifest_sha256,
                "metadata": str(metadata_output),
                "unique_before_sampling": unique_before_sampling,
                "written": len(records),
                "groups": len(group_splits),
                "splits": {split: split_counts.get(split, 0) for split in SPLIT_NAMES},
                "sources": {source: source_counts.get(source, 0) for source in SOURCE_NAMES},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

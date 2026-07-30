# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for NVIDIA timeline prompt preparation."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts.minilm.prepare_prompts import (
    MAX_PROMPT_CHARACTERS,
    SOURCE_NAMES,
    PromptRecord,
    extract_prompts,
    main,
    motion_family,
    prompt_deduplication_key,
    read_prompt_manifest,
    resolve_dataset_path,
)


def _timeline_row(
    filename: str,
    overview: str,
    events: list[str],
    *,
    propagated_from_filename: str | None = None,
) -> dict:
    return {
        "overview_description": overview,
        "num_events": len(events),
        "events": [
            {
                "start_time": float(index),
                "end_time": float(index + 1),
                "description": description,
            }
            for index, description in enumerate(events)
        ],
        "filename": filename,
        "propagated_from_filename": propagated_from_filename,
    }


class TimelinePreparationTests(unittest.TestCase):
    def test_motion_family_coalesces_actor_take_and_mirror_variants(self) -> None:
        self.assertEqual(motion_family("Jump_Left_002__A019_M"), "jump_left")
        self.assertEqual(motion_family("Jump_Left_001__A017"), "jump_left")
        self.assertEqual(motion_family("turn_ff_180_R_003__A244"), "turn_ff_180_r")
        self.assertEqual(motion_family("exercise_1__A033"), "exercise_1")
        self.assertEqual(motion_family("body_check_01__A033"), "body_check_01")
        self.assertEqual(motion_family("jump_ff_270__A045"), "jump_ff_270")
        self.assertEqual(motion_family("jump_ff_360__A045"), "jump_ff_360")
        with self.assertRaisesRegex(ValueError, "does not match"):
            motion_family("missing-actor-suffix")

    def test_propagated_components_share_split_and_duplicate_text_is_global(self) -> None:
        rows = [
            _timeline_row(
                "Walk_Left_001__A017",
                "  A person walks left. ",
                ["A person starts walking.", "Shared event."],
                propagated_from_filename="stride_left_004__A019",
            ),
            _timeline_row(
                "stride_left_009__A020_M",
                "A person strides to the right.",
                ["Shared event."],
            ),
            _timeline_row(
                "Jump_001__A017",
                "Ａ person jumps.\n",
                ["a PERSON JUMPS.", "A person lands.", "x" * (MAX_PROMPT_CHARACTERS + 1)],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timelines.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            records, counts = extract_prompts(
                path,
                seed=20260730,
                split_ratios=(0.8, 0.1, 0.1),
            )

        by_text = {record.text: record for record in records}
        self.assertEqual(len(by_text), len(records))
        self.assertEqual(by_text["A person walks left."].group, by_text["A person strides to the right."].group)
        self.assertEqual(by_text["A person walks left."].split, by_text["A person strides to the right."].split)
        self.assertEqual(sum(record.text == "Shared event." for record in records), 1)
        self.assertEqual(by_text["A person jumps."].source, "overview_description")
        self.assertEqual({record.source for record in records}, set(SOURCE_NAMES))
        self.assertEqual(counts["timeline_rows"], 3)
        self.assertEqual(counts["raw_overview_descriptions"], 3)
        self.assertEqual(counts["raw_event_descriptions"], 6)
        self.assertEqual(counts["dropped_prompt_too_long"], 1)
        self.assertEqual(counts["unique_descriptions"], 6)

    def test_prompt_deduplication_key_ignores_case_and_punctuation(self) -> None:
        variants = (
            "A person is standing idle facing forward.",
            "a PERSON is standing idle, facing forward!",
            "A person is standing idle—facing forward",
        )
        self.assertEqual(
            {prompt_deduplication_key(text) for text in variants},
            {"a person is standing idle facing forward"},
        )
        self.assertEqual(
            prompt_deduplication_key("A person holds a T-pose."),
            prompt_deduplication_key("A person holds a T pose"),
        )
        self.assertNotEqual(
            prompt_deduplication_key("A person walks left."),
            prompt_deduplication_key("A person walks right."),
        )

    def test_num_events_mismatch_fails_closed(self) -> None:
        row = _timeline_row("Walk_001__A017", "A person walks.", ["One event."])
        row["num_events"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timelines.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "num_events must equal"):
                extract_prompts(path, seed=1, split_ratios=(0.8, 0.1, 0.1))

    def test_local_source_size_and_hash_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timelines.jsonl"
            payload = b"fixed timeline fixture\n"
            path.write_bytes(payload)
            expected_sha256 = hashlib.sha256(payload).hexdigest()
            with (
                patch(
                    "scripts.minilm.prepare_prompts.DATASET_SIZE_BYTES",
                    len(payload),
                ),
                patch(
                    "scripts.minilm.prepare_prompts.DATASET_SHA256",
                    expected_sha256,
                ),
            ):
                self.assertEqual(resolve_dataset_path(path), (path, "local_input"))

            with (
                patch(
                    "scripts.minilm.prepare_prompts.DATASET_SIZE_BYTES",
                    len(payload),
                ),
                patch(
                    "scripts.minilm.prepare_prompts.DATASET_SHA256",
                    "0" * 64,
                ),
                self.assertRaisesRegex(ValueError, "expected SHA-256"),
            ):
                resolve_dataset_path(path)

            with (
                patch(
                    "scripts.minilm.prepare_prompts.DATASET_SIZE_BYTES",
                    len(payload) + 1,
                ),
                self.assertRaisesRegex(ValueError, "expected .* bytes"),
            ):
                resolve_dataset_path(path)

    def test_manifest_reader_rejects_duplicate_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            rows = [
                {
                    "text": "same",
                    "split": "train",
                    "group": "group-1",
                    "source": "overview_description",
                },
                {
                    "text": "same",
                    "split": "val",
                    "group": "group-2",
                    "source": "events.description",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                read_prompt_manifest(path)

    def test_manifest_reader_rejects_punctuation_only_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            rows = [
                {
                    "text": "A person is standing idle facing forward.",
                    "split": "train",
                    "group": "group-1",
                    "source": "overview_description",
                },
                {
                    "text": "A person is standing idle, facing forward!",
                    "split": "test",
                    "group": "group-2",
                    "source": "events.description",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                read_prompt_manifest(path)

    def test_manifest_reader_enforces_producer_text_invariants(self) -> None:
        invalid_texts = (
            ("x  y", "canonical"),
            ("x" * (MAX_PROMPT_CHARACTERS + 1), "exceeds"),
        )
        for text, message in invalid_texts:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "prompts.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "text": text,
                            "split": "train",
                            "group": "group-1",
                            "source": "overview_description",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_prompt_manifest(path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "Same",
                        "split": "train",
                        "group": "group-1",
                        "source": "overview_description",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "text": "same",
                        "split": "test",
                        "group": "group-2",
                        "source": "events.description",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                read_prompt_manifest(path)

    def test_cli_rejects_overwriting_the_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "timelines.jsonl"
            source.write_bytes(b"fixture")
            with (
                patch(
                    "scripts.minilm.prepare_prompts.resolve_dataset_path",
                    return_value=(source, "local_input"),
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(["--input", str(source), "--output", str(source)]),
                    2,
                )
                self.assertEqual(
                    main(
                        [
                            "--input",
                            str(source),
                            "--output",
                            str(Path(directory) / "prompts.jsonl"),
                            "--metadata-output",
                            str(source),
                        ]
                    ),
                    2,
                )

    def test_manifest_reader_accepts_both_timeline_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            rows = [
                PromptRecord("overview", "train", "group-1", "overview_description"),
                PromptRecord("event", "test", "group-2", "events.description"),
            ]
            path.write_text(
                "".join(json.dumps(record.__dict__) + "\n" for record in rows),
                encoding="utf-8",
            )
            self.assertEqual(read_prompt_manifest(path), rows)


if __name__ == "__main__":
    unittest.main()

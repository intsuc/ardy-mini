# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for leakage-safe MiniLM prompt-corpus expansion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.minilm.prepare_prompts import (
    PromptRecord,
    freeze_evaluation_records,
    read_prompt_manifest,
)


class FrozenEvaluationTests(unittest.TestCase):
    def test_all_training_rows_are_added_while_evaluation_stays_fixed(self) -> None:
        current = [
            PromptRecord("train one", "train", "group-1", "natural"),
            PromptRecord("train two", "train", "group-2", "natural"),
            PromptRecord("validation", "val", "group-3", "natural"),
            PromptRecord("unused validation", "val", "group-4", "natural"),
            PromptRecord("test", "test", "group-5", "natural"),
            PromptRecord("unused test", "test", "group-6", "natural"),
        ]
        reference = [
            current[0],
            current[2],
            current[4],
        ]

        expanded = freeze_evaluation_records(current, reference)

        self.assertEqual(
            expanded,
            [current[0], current[1], current[2], current[4]],
        )
        self.assertEqual(
            [record.text for record in expanded if record.split == "val"],
            ["validation"],
        )
        self.assertEqual(
            [record.text for record in expanded if record.split == "test"],
            ["test"],
        )

    def test_changed_reference_record_fails_closed(self) -> None:
        current = [
            PromptRecord("train", "train", "group-1", "natural"),
            PromptRecord("validation", "val", "group-2", "natural"),
            PromptRecord("test", "test", "group-3", "natural"),
        ]
        reference = [
            current[0],
            PromptRecord("validation", "val", "changed-group", "natural"),
            current[2],
        ]

        with self.assertRaisesRegex(ValueError, "no longer matches"):
            freeze_evaluation_records(current, reference)

    def test_manifest_reader_rejects_duplicate_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            rows = [
                {
                    "text": "same",
                    "split": "train",
                    "group": "group-1",
                    "source": "natural",
                },
                {
                    "text": "same",
                    "split": "val",
                    "group": "group-2",
                    "source": "natural",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate prompt"):
                read_prompt_manifest(path)


if __name__ == "__main__":
    unittest.main()

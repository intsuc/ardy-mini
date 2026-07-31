# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for the allowlisted Model Hub release staging pipeline."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ardy.model_hub_release import (
    MODEL_DISPLAY_NAME,
    MODEL_FORMAT,
    MODEL_ID,
    MODEL_REPOSITORY,
    RAW_MODEL_FILES,
    ReleaseConfig,
    ReleaseValidationError,
    stage_model_hub_release,
    validate_variant_directory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def write_variant(root: Path, variant: str, *, model_id: str = MODEL_ID) -> None:
    files = {}
    for index, raw_path in enumerate(RAW_MODEL_FILES):
        raw = f"{variant}:{index}:{raw_path}".encode()
        transport = gzip_bytes(raw)
        transport_path = f"{raw_path}.gz"
        destination = root / variant / transport_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(transport)
        files[raw_path] = {
            "sha256": sha256(raw),
            "size_bytes": len(raw),
            "transport": {
                "compression": "gzip",
                "path": transport_path,
                "sha256": sha256(transport),
                "size_bytes": len(transport),
            },
        }
    manifest = {
        "files": files,
        "format": MODEL_FORMAT,
        "model": {
            "ardy_checkpoint": {"files": {"denoiser.safetensors": {"sha256": "c" * 64, "size_bytes": 1}}},
            "ardy_model": "ARDY-Core-RP-20FPS-Horizon40",
            "display_name": MODEL_DISPLAY_NAME,
            "id": model_id,
            "minilm_artifact_fingerprint": "a" * 64,
            "revision": "b" * 64,
        },
        "precision": {
            "required_webgpu_features": ["shader-f16"] if variant == "fp16" else [],
        },
        "provenance": {"dataset": "nvidia/SEED-Timeline-Annotations"},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (root / variant / "model.json.gz").write_bytes(gzip_bytes(manifest_bytes))


class ModelHubReleaseTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> ReleaseConfig:
        models = root / "models"
        write_variant(models, "fp16")
        write_variant(models, "fp32")

        templates = root / "templates"
        templates.mkdir()
        (templates / "README.md.template").write_text(
            "# Model\n\nFP16 ${fp16_size_mib} MiB; FP32 ${fp32_size_mib} MiB.\n",
            encoding="utf-8",
        )
        (templates / "MODEL_TERMS.md").write_text("terms\n", encoding="utf-8")
        (templates / "NOTICE").write_text("notice\n", encoding="utf-8")

        license_data = b"fixture license\n"
        license_sources = {
            "format": "ardy-model-license-sources",
            "format_version": 1,
            "licenses": [
                {
                    "filename": "FIXTURE.txt",
                    "name": "Fixture",
                    "sha256": sha256(license_data),
                    "size_bytes": len(license_data),
                    "url": "https://example.invalid/LICENSE",
                }
            ],
        }
        license_sources_path = templates / "LICENSE_SOURCES.json"
        license_sources_path.write_text(json.dumps(license_sources), encoding="utf-8")
        license_cache = root / "license-cache"
        license_cache.mkdir()
        (license_cache / "FIXTURE.txt").write_bytes(license_data)

        reports = root / "reports"
        reports.mkdir()
        manifests = {}
        for variant in ("fp16", "fp32"):
            with gzip.open(models / variant / "model.json.gz", "rb") as handle:
                manifests[variant] = json.load(handle)
        fp16_report = {
            "contract_validation": {
                "candidate_graph_sha256": {
                    graph: manifests["fp16"]["files"][f"{graph}.onnx"]["sha256"]
                    for graph in ("decoder", "denoiser", "text_encoder")
                },
                "reference_graph_sha256": {
                    graph: manifests["fp32"]["files"][f"{graph}.onnx"]["sha256"]
                    for graph in ("decoder", "denoiser", "text_encoder")
                },
                "tokenizer_payload_sha256": {
                    raw_path: manifests["fp32"]["files"][raw_path]["sha256"]
                    for raw_path in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
                },
            },
            "models": {
                "candidate": {"id": "internal", "revision": "b" * 64},
                "reference": {"id": "internal", "revision": "b" * 64},
            },
        }
        summary = {
            "distribution": {
                "third_party_notices": "../THIRD_PARTY_MODELS_AND_DATA.md",
            },
            "test_evaluation": {"motion_fidelity": {"scope": {"checkpoint_sha256": "c" * 64}}},
            "winner": {"artifact_fingerprint": "a" * 64},
        }
        (reports / "browser_fp16_ablation.json").write_text(json.dumps(fp16_report), encoding="utf-8")
        (reports / "minilm_core40_summary.json").write_text(json.dumps(summary), encoding="utf-8")

        return ReleaseConfig(
            model_directory=models,
            output_directory=root / "release",
            repository_root=REPOSITORY_ROOT,
            template_directory=templates,
            reports_directory=reports,
            license_sources_path=license_sources_path,
            license_cache_directory=license_cache,
            use_hardlinks=False,
            allow_dirty_source=True,
        )

    def test_stages_only_allowlisted_files_with_provenance_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_fixture(root)
            output = stage_model_hub_release(config)

            self.assertEqual(json.loads((output / "config.json").read_text())["hub_repository"], MODEL_REPOSITORY)
            self.assertTrue((output / "MODEL_PROVENANCE.json").is_file())
            self.assertEqual((output / "LICENSES/FIXTURE.txt").read_bytes(), b"fixture license\n")
            self.assertFalse((output / "README.md").read_text().find("${") >= 0)
            public_summary = json.loads(
                (output / "reports/minilm_core40_summary.json").read_text()
            )
            self.assertEqual(
                public_summary["distribution"]["third_party_notices"],
                "../MODEL_TERMS.md",
            )
            self.assertEqual(
                json.loads(
                    (config.reports_directory / "minilm_core40_summary.json").read_text()
                )["distribution"]["third_party_notices"],
                "../THIRD_PARTY_MODELS_AND_DATA.md",
            )

            checksum_lines = (output / "SHA256SUMS").read_text().splitlines()
            checksum_paths = {line.split("  ", 1)[1] for line in checksum_lines}
            actual_paths = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            }
            self.assertEqual(checksum_paths, actual_paths)
            for line in checksum_lines:
                expected, relative = line.split("  ", 1)
                self.assertEqual(sha256((output / relative).read_bytes()), expected)

    def test_rejects_unallowlisted_variant_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_fixture(root)
            (config.model_directory / "fp16/teacher-cache.pt").write_bytes(b"private")
            with self.assertRaisesRegex(ReleaseValidationError, "allowlist mismatch"):
                validate_variant_directory(config.model_directory / "fp16", "fp16")

    def test_rejects_non_llama_prefixed_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            write_variant(models, "fp16", model_id="ardy-minilm-core40-browser-v1")
            with self.assertRaisesRegex(ReleaseValidationError, "model.id"):
                validate_variant_directory(models / "fp16", "fp16")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Build a provenance-checked, allowlisted Hugging Face model release tree."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any

MODEL_ID = "Llama-3-ARDY-Mini-Core40-Browser"
MODEL_DISPLAY_NAME = "Llama 3 ARDY Mini Core40 Browser"
MODEL_REPOSITORY = "intsuc/Llama-3-ARDY-Mini-Core40-Browser"
MODEL_PUBLISHER = "intsuc"
MODEL_CONTACT = "i@intsuc.dev"
SOURCE_REPOSITORY = "https://github.com/intsuc/ardy-mini"
MODEL_FORMAT = "ardy-browser-model-files"
VARIANTS = ("fp16", "fp32")
RAW_MODEL_FILES = (
    "decoder.onnx",
    "denoiser.onnx",
    "text_encoder.onnx",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
TRANSPORT_FILES = tuple(f"{path}.gz" for path in RAW_MODEL_FILES)
VARIANT_FILES = ("model.json.gz", *TRANSPORT_FILES)
PUBLIC_REPORTS = (
    "browser_fp16_ablation.json",
    "minilm_core40_summary.json",
)
SOURCE_REPORT_TERMS_PATH = "../THIRD_PARTY_MODELS_AND_DATA.md"
PUBLIC_REPORT_TERMS_PATH = "../MODEL_TERMS.md"
STATIC_RELEASE_FILES = (
    "MODEL_TERMS.md",
    "NOTICE",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_ID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_LICENSE_BYTES = 1024 * 1024
MAX_RAW_MODEL_FILE_BYTES = 2 * 1024 * 1024 * 1024


class ReleaseValidationError(RuntimeError):
    """Raised when release input is incomplete, stale, or unsafe."""


@dataclass(frozen=True)
class ReleaseConfig:
    """Inputs for one deterministic Model Hub release staging run."""

    model_directory: Path
    output_directory: Path
    repository_root: Path
    template_directory: Path
    reports_directory: Path
    license_sources_path: Path
    license_cache_directory: Path | None = None
    use_hardlinks: bool = True
    allow_dirty_source: bool = False
    replace: bool = False


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"Expected a JSON object in {path}")
    return value


def _public_report_bytes(
    filename: str,
    source_bytes: bytes,
) -> tuple[bytes, dict[str, Any]]:
    """Prepare an aggregate report for the Model Hub directory layout."""

    try:
        document = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"Could not parse public report {filename}: {exc}") from exc
    if not isinstance(document, dict):
        raise ReleaseValidationError(f"Expected a JSON object in public report {filename}")

    if filename == "minilm_core40_summary.json":
        distribution = document.get("distribution")
        if not isinstance(distribution, dict):
            raise ReleaseValidationError("MiniLM summary has no distribution metadata")
        if distribution.get("third_party_notices") != SOURCE_REPORT_TERMS_PATH:
            raise ReleaseValidationError(
                "MiniLM summary third-party notice path changed; review the public release transformation"
            )
        source_fragment = json.dumps(SOURCE_REPORT_TERMS_PATH).encode("utf-8")
        public_fragment = json.dumps(PUBLIC_REPORT_TERMS_PATH).encode("utf-8")
        if source_bytes.count(source_fragment) != 1:
            raise ReleaseValidationError(
                "MiniLM summary must contain exactly one source third-party notice path"
            )
        source_bytes = source_bytes.replace(source_fragment, public_fragment)
        distribution["third_party_notices"] = PUBLIC_REPORT_TERMS_PATH

    return source_bytes, document


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gzip_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        with gzip.open(path, "rb") as handle:
            data = handle.read(MAX_MANIFEST_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ReleaseValidationError(f"Could not decompress {path}: {exc}") from exc
    if len(data) > MAX_MANIFEST_BYTES:
        raise ReleaseValidationError(f"Manifest exceeds {MAX_MANIFEST_BYTES} bytes: {path}")
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ReleaseValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"Expected a JSON object in {path}")
    return value, data


def _safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or path.as_posix() != value:
        raise ReleaseValidationError(f"{field} is not a canonical relative POSIX path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseValidationError(f"{field} contains an unsafe path component: {value!r}")
    return path


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReleaseValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_size(value: Any, *, field: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ReleaseValidationError(f"{field} must be an integer from 0 through {maximum}")
    return value


def _verify_gzip_transport(path: Path, entry: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    transport = entry.get("transport")
    if not isinstance(transport, Mapping) or transport.get("compression") != "gzip":
        raise ReleaseValidationError(f"{field}.transport must declare gzip compression")
    compressed_size = _require_size(
        transport.get("size_bytes"), field=f"{field}.transport.size_bytes", maximum=MAX_RAW_MODEL_FILE_BYTES
    )
    compressed_sha256 = _require_sha256(transport.get("sha256"), field=f"{field}.transport.sha256")
    if path.stat().st_size != compressed_size:
        raise ReleaseValidationError(f"Compressed size mismatch for {path}")
    if _sha256_file(path) != compressed_sha256:
        raise ReleaseValidationError(f"Compressed SHA-256 mismatch for {path}")

    raw_size = _require_size(entry.get("size_bytes"), field=f"{field}.size_bytes", maximum=MAX_RAW_MODEL_FILE_BYTES)
    raw_sha256 = _require_sha256(entry.get("sha256"), field=f"{field}.sha256")
    digest = hashlib.sha256()
    decompressed_size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                decompressed_size += len(chunk)
                if decompressed_size > raw_size or decompressed_size > MAX_RAW_MODEL_FILE_BYTES:
                    raise ReleaseValidationError(f"Decompressed file is larger than declared: {path}")
                digest.update(chunk)
    except (OSError, EOFError) as exc:
        raise ReleaseValidationError(f"Could not verify gzip transport {path}: {exc}") from exc
    if decompressed_size != raw_size:
        raise ReleaseValidationError(f"Raw size mismatch for {path}")
    if digest.hexdigest() != raw_sha256:
        raise ReleaseValidationError(f"Raw SHA-256 mismatch for {path}")
    return {
        "raw_sha256": raw_sha256,
        "raw_size_bytes": raw_size,
        "transport_sha256": compressed_sha256,
        "transport_size_bytes": compressed_size,
    }


def validate_variant_directory(variant_directory: Path, variant: str) -> dict[str, Any]:
    """Verify an exporter-owned model directory and return public identities."""

    if variant not in VARIANTS:
        raise ReleaseValidationError(f"Unsupported variant: {variant}")
    if not variant_directory.is_dir() or variant_directory.is_symlink():
        raise ReleaseValidationError(f"Missing real variant directory: {variant_directory}")
    actual_files = {
        path.relative_to(variant_directory).as_posix() for path in variant_directory.rglob("*") if path.is_file()
    }
    symlinks = [path for path in variant_directory.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ReleaseValidationError(f"Variant contains symlinks: {', '.join(map(str, symlinks))}")
    if actual_files != set(VARIANT_FILES):
        missing = sorted(set(VARIANT_FILES) - actual_files)
        extra = sorted(actual_files - set(VARIANT_FILES))
        raise ReleaseValidationError(f"Variant {variant} allowlist mismatch; missing={missing}, extra={extra}")

    manifest_path = variant_directory / "model.json.gz"
    manifest, manifest_bytes = _read_gzip_json(manifest_path)
    if manifest.get("format") != MODEL_FORMAT:
        raise ReleaseValidationError(f"Unexpected manifest format in {manifest_path}")
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ReleaseValidationError(f"Missing model identity in {manifest_path}")
    if model.get("id") != MODEL_ID:
        raise ReleaseValidationError(f"model.id must be {MODEL_ID!r} in {manifest_path}")
    if model.get("display_name") != MODEL_DISPLAY_NAME:
        raise ReleaseValidationError(f"model.display_name must be {MODEL_DISPLAY_NAME!r} in {manifest_path}")
    provenance = manifest.get("provenance", model.get("provenance"))
    if not isinstance(provenance, Mapping) or not provenance:
        raise ReleaseValidationError(f"Missing release provenance in {manifest_path}")

    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(RAW_MODEL_FILES):
        raise ReleaseValidationError(f"Manifest file allowlist mismatch in {manifest_path}")
    file_identities: dict[str, Any] = {}
    for raw_path in RAW_MODEL_FILES:
        entry = files[raw_path]
        if not isinstance(entry, Mapping):
            raise ReleaseValidationError(f"files.{raw_path} must be an object")
        transport = entry.get("transport")
        if not isinstance(transport, Mapping):
            raise ReleaseValidationError(f"files.{raw_path}.transport must be an object")
        declared_transport_path = _safe_relative_path(
            str(transport.get("path", "")), field=f"files.{raw_path}.transport.path"
        ).as_posix()
        expected_transport_path = f"{raw_path}.gz"
        if declared_transport_path != expected_transport_path:
            raise ReleaseValidationError(
                f"Transport for {raw_path} must be {expected_transport_path!r}, got {declared_transport_path!r}"
            )
        file_identities[raw_path] = _verify_gzip_transport(
            variant_directory / declared_transport_path,
            entry,
            field=f"files.{raw_path}",
        )

    required_features = manifest.get("precision", {}).get("required_webgpu_features")
    expected_features = ["shader-f16"] if variant == "fp16" else []
    if required_features != expected_features:
        raise ReleaseValidationError(
            f"Variant {variant} must require WebGPU features {expected_features}, got {required_features!r}"
        )
    total_transport_bytes = manifest_path.stat().st_size + sum(
        identity["transport_size_bytes"] for identity in file_identities.values()
    )
    return {
        "files": file_identities,
        "manifest_gzip_sha256": _sha256_file(manifest_path),
        "manifest_gzip_size_bytes": manifest_path.stat().st_size,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "manifest_size_bytes": len(manifest_bytes),
        "model": dict(model),
        "precision": manifest.get("precision"),
        "provenance": dict(provenance),
        "total_transport_bytes": total_transport_bytes,
    }


def _git_value(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ReleaseValidationError(f"Git command failed: git {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _source_identity(repository_root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    commit = _git_value(repository_root, "rev-parse", "HEAD")
    if GIT_OBJECT_ID_RE.fullmatch(commit) is None:
        raise ReleaseValidationError("Source commit is not a SHA-1 or SHA-256 Git object ID")
    dirty_output = _git_value(repository_root, "status", "--porcelain", "--untracked-files=normal")
    dirty = bool(dirty_output)
    if dirty and not allow_dirty:
        raise ReleaseValidationError(
            "Tracked source changes are present. Commit them before staging, or use --allow-dirty-source for a non-release check."
        )
    return {
        "commit": commit,
        "dirty": dirty,
        "repository": SOURCE_REPOSITORY,
    }


def _validate_report_bindings(
    variants: Mapping[str, Mapping[str, Any]], reports: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind existing evaluations to release graph bytes, not mutable manifest metadata."""

    summary = reports["minilm_core40_summary.json"]
    winner = summary.get("winner")
    if not isinstance(winner, Mapping):
        raise ReleaseValidationError("MiniLM summary has no winner identity")
    expected_student = variants["fp32"]["model"].get("minilm_artifact_fingerprint")
    if winner.get("artifact_fingerprint") != expected_student:
        raise ReleaseValidationError("MiniLM summary is not for the released student artifact")

    model_identity = variants["fp32"]["model"]
    checkpoint = model_identity.get("ardy_checkpoint")
    checkpoint_files = checkpoint.get("files") if isinstance(checkpoint, Mapping) else None
    denoiser = checkpoint_files.get("denoiser.safetensors") if isinstance(checkpoint_files, Mapping) else None
    expected_checkpoint = denoiser.get("sha256") if isinstance(denoiser, Mapping) else None
    summary_checkpoint = (
        summary.get("test_evaluation", {}).get("motion_fidelity", {}).get("scope", {}).get("checkpoint_sha256")
    )
    if not expected_checkpoint or summary_checkpoint != expected_checkpoint:
        raise ReleaseValidationError("MiniLM summary is not for the released ARDY checkpoint")

    ablation = reports["browser_fp16_ablation.json"]
    report_models = ablation.get("models")
    contract = ablation.get("contract_validation")
    if not isinstance(report_models, Mapping) or not isinstance(contract, Mapping):
        raise ReleaseValidationError("FP16 report is missing model or contract identities")
    expected_roles = {"candidate": "fp16", "reference": "fp32"}
    for report_role, variant in expected_roles.items():
        report_model = report_models.get(report_role)
        if not isinstance(report_model, Mapping):
            raise ReleaseValidationError(f"FP16 report is missing {report_role} identity")
        graph_hashes = contract.get(f"{report_role}_graph_sha256")
        if not isinstance(graph_hashes, Mapping):
            raise ReleaseValidationError(f"FP16 report is missing {report_role} graph hashes")
        for graph in ("decoder", "denoiser", "text_encoder"):
            expected_hash = variants[variant]["files"][f"{graph}.onnx"]["raw_sha256"]
            if graph_hashes.get(graph) != expected_hash:
                raise ReleaseValidationError(f"FP16 report {report_role} {graph} hash does not match {variant}")

    tokenizer_hashes = contract.get("tokenizer_payload_sha256")
    if not isinstance(tokenizer_hashes, Mapping):
        raise ReleaseValidationError("FP16 report is missing tokenizer hashes")
    for raw_path in ("tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
        expected_hash = variants["fp32"]["files"][raw_path]["raw_sha256"]
        if tokenizer_hashes.get(raw_path) != expected_hash:
            raise ReleaseValidationError(f"FP16 report tokenizer hash does not match {raw_path}")

    return {
        "method": "Exact raw ONNX and tokenizer SHA-256 plus checkpoint and student identities",
        "browser_fp16_report_model_ids": {role: report_models[role].get("id") for role in expected_roles},
        "browser_fp16_report_model_revisions": {role: report_models[role].get("revision") for role in expected_roles},
        "note": (
            "The FP16 evaluation report retains the internal pre-release model ID and manifest hashes. "
            "The release changes provenance-bound manifest metadata, not evaluated graph or tokenizer bytes; "
            "the exact payload hashes above bind it."
        ),
    }


def _validate_license_sources(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = _load_json(path)
    if document.get("format") != "ardy-model-license-sources" or document.get("format_version") != 1:
        raise ReleaseValidationError(f"Unsupported license source manifest: {path}")
    entries = document.get("licenses")
    if not isinstance(entries, list) or not entries:
        raise ReleaseValidationError(f"License source manifest contains no licenses: {path}")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, value in enumerate(entries):
        if not isinstance(value, dict):
            raise ReleaseValidationError(f"licenses[{index}] must be an object")
        filename = _safe_relative_path(str(value.get("filename", "")), field=f"licenses[{index}].filename")
        if len(filename.parts) != 1 or filename.suffix not in {".txt", ".md"}:
            raise ReleaseValidationError(f"licenses[{index}].filename must be a flat text filename")
        if filename.as_posix() in seen:
            raise ReleaseValidationError(f"Duplicate license filename: {filename}")
        seen.add(filename.as_posix())
        url = value.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ReleaseValidationError(f"licenses[{index}].url must be HTTPS")
        validated.append(
            {
                "filename": filename.as_posix(),
                "name": str(value.get("name", "")),
                "sha256": _require_sha256(value.get("sha256"), field=f"licenses[{index}].sha256"),
                "size_bytes": _require_size(
                    value.get("size_bytes"), field=f"licenses[{index}].size_bytes", maximum=MAX_LICENSE_BYTES
                ),
                "url": url,
            }
        )
    return document, validated


def _download_bounded(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ardy-mini-model-release/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_LICENSE_BYTES + 1)
    except OSError as exc:
        raise ReleaseValidationError(f"Could not download license from {url}: {exc}") from exc
    if len(data) > MAX_LICENSE_BYTES:
        raise ReleaseValidationError(f"License download exceeds {MAX_LICENSE_BYTES} bytes: {url}")
    return data


def _license_bytes(entry: Mapping[str, Any], cache_directory: Path | None) -> bytes:
    cache_path = cache_directory / str(entry["filename"]) if cache_directory is not None else None
    if cache_path is not None and cache_path.is_file() and not cache_path.is_symlink():
        data = cache_path.read_bytes()
    else:
        data = _download_bounded(str(entry["url"]))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
    if len(data) != entry["size_bytes"] or _sha256_bytes(data) != entry["sha256"]:
        if cache_path is not None and cache_path.exists():
            cache_path.unlink()
        raise ReleaseValidationError(f"Pinned license identity mismatch: {entry['filename']}")
    return data


def _copy_file(source: Path, destination: Path, *, use_hardlinks: bool) -> None:
    if not source.is_file() or source.is_symlink():
        raise ReleaseValidationError(f"Release source must be a real file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_hardlinks:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _write_sha256sums(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleaseValidationError(f"Release staging tree contains a symlink: {path}")
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _model_family_consistent(variants: Mapping[str, Mapping[str, Any]]) -> None:
    fp16_model = variants["fp16"]["model"]
    fp32_model = variants["fp32"]["model"]
    fields = ("id", "display_name", "revision", "minilm_artifact_fingerprint", "ardy_model")
    mismatches = [field for field in fields if fp16_model.get(field) != fp32_model.get(field)]
    if mismatches:
        raise ReleaseValidationError(f"FP16/FP32 model identities disagree: {mismatches}")


def _replace_directory(staging: Path, destination: Path, *, replace: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not replace:
        raise ReleaseValidationError(
            f"Output already exists: {destination}; pass --replace to replace a prior release tree"
        )
    marker = destination / "config.json"
    if not marker.is_file():
        raise ReleaseValidationError(f"Refusing to replace unrecognized directory without config.json: {destination}")
    existing = _load_json(marker)
    if existing.get("model_id") != MODEL_ID or existing.get("hub_repository") != MODEL_REPOSITORY:
        raise ReleaseValidationError(f"Refusing to replace a directory for another model: {destination}")
    backup = destination.with_name(f".{destination.name}.replaced")
    if backup.exists():
        raise ReleaseValidationError(f"Stale replacement backup exists: {backup}")
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def stage_model_hub_release(config: ReleaseConfig) -> Path:
    """Validate inputs and atomically stage the complete public model repository."""

    repository_root = config.repository_root.resolve()
    model_directory = config.model_directory.resolve()
    output_directory = config.output_directory.resolve()
    template_directory = config.template_directory.resolve()
    reports_directory = config.reports_directory.resolve()
    if output_directory == repository_root:
        raise ReleaseValidationError("Output directory must not be the repository root")
    source = _source_identity(repository_root, allow_dirty=config.allow_dirty_source)

    variants = {variant: validate_variant_directory(model_directory / variant, variant) for variant in VARIANTS}
    _model_family_consistent(variants)

    reports: dict[str, dict[str, Any]] = {}
    report_documents: dict[str, dict[str, Any]] = {}
    report_payloads: dict[str, bytes] = {}
    for filename in PUBLIC_REPORTS:
        path = reports_directory / filename
        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            raise ReleaseValidationError(f"Could not read public report {path}: {exc}") from exc
        payload, document = _public_report_bytes(filename, source_bytes)
        report_payloads[filename] = payload
        report_documents[filename] = document
        reports[filename] = {
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    evaluation_binding = _validate_report_bindings(variants, report_documents)

    license_document, license_entries = _validate_license_sources(config.license_sources_path.resolve())
    template_path = template_directory / "README.md.template"
    try:
        model_card_template = Template(template_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReleaseValidationError(f"Could not read model-card template {template_path}: {exc}") from exc
    for filename in STATIC_RELEASE_FILES:
        if not (template_directory / filename).is_file():
            raise ReleaseValidationError(f"Missing static release file: {template_directory / filename}")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.staging-", dir=output_directory.parent))
    try:
        for variant in VARIANTS:
            for relative in VARIANT_FILES:
                _copy_file(
                    model_directory / variant / relative,
                    staging / variant / relative,
                    use_hardlinks=config.use_hardlinks,
                )
        for filename in PUBLIC_REPORTS:
            report_path = staging / "reports" / filename
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_bytes(report_payloads[filename])
        for filename in STATIC_RELEASE_FILES:
            _copy_file(template_directory / filename, staging / filename, use_hardlinks=False)

        licenses_directory = staging / "LICENSES"
        licenses_directory.mkdir(parents=True, exist_ok=True)
        for entry in license_entries:
            (licenses_directory / entry["filename"]).write_bytes(_license_bytes(entry, config.license_cache_directory))
        (licenses_directory / "SOURCES.json").write_bytes(_canonical_json_bytes(license_document))

        model_card = model_card_template.substitute(
            fp16_size_mib=f"{variants['fp16']['total_transport_bytes'] / (1024**2):.2f}",
            fp32_size_mib=f"{variants['fp32']['total_transport_bytes'] / (1024**2):.2f}",
        )
        (staging / "README.md").write_text(model_card, encoding="utf-8")

        family_config = {
            "display_name": MODEL_DISPLAY_NAME,
            "format": "ardy-browser-model-family",
            "format_version": 1,
            "hub_repository": MODEL_REPOSITORY,
            "model_id": MODEL_ID,
            "publisher": {"contact": MODEL_CONTACT, "name": MODEL_PUBLISHER},
            "variants": {
                "fp16": {
                    "manifest": "fp16/model.json.gz",
                    "required_webgpu_features": ["shader-f16"],
                },
                "fp32": {
                    "manifest": "fp32/model.json.gz",
                    "required_webgpu_features": [],
                },
            },
        }
        (staging / "config.json").write_bytes(_canonical_json_bytes(family_config))

        provenance = {
            "format": "ardy-model-hub-provenance",
            "format_version": 1,
            "hub": {"repository": MODEL_REPOSITORY, "visibility": "public", "gated": False},
            "model": {
                "display_name": MODEL_DISPLAY_NAME,
                "id": MODEL_ID,
                "publisher": MODEL_PUBLISHER,
                "contact": MODEL_CONTACT,
            },
            "reports": reports,
            "evaluation_binding": evaluation_binding,
            "source": source,
            "variants": variants,
            "license_sources_sha256": _sha256_bytes(_canonical_json_bytes(license_document)),
        }
        (staging / "MODEL_PROVENANCE.json").write_bytes(_canonical_json_bytes(provenance))
        _write_sha256sums(staging)
        _replace_directory(staging, output_directory, replace=config.replace)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_directory

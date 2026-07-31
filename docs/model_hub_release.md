# Model Hub release

The public browser weights are released separately from the Apache-2.0 source
repository as the composite model
[`intsuc/Llama-3-ARDY-Mini-Core40-Browser`](https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser).
The source tree does not track generated ONNX files or a generated Hub staging
directory.

## Prepare a release

Start from a clean, committed source revision. Export and verify both browser
variants, then build the allowlisted release tree:

```bash
uv run --extra browser python scripts/export_browser_models.py \
  --checkpoints-dir checkpoints \
  --minilm-artifact artifacts/minilm-ardy-core40 \
  --output-directory artifacts/browser/ardy-minilm-core40-browser-v1

uv run python scripts/prepare_model_hub_release.py
```

The release builder performs the following checks before producing output:

- both manifests use the formal `Llama-3-ARDY-Mini-Core40-Browser` identity;
- FP16 and FP32 identities agree and declare the expected WebGPU features;
- each variant contains exactly the five declared gzip transports and its
  compressed manifest—unexpected files, symlinks, and unsafe paths fail;
- compressed and decompressed sizes and SHA-256 hashes match every manifest;
- the two aggregate public reports are valid JSON;
- the MiniLM summary's source-distribution notice link is rewritten to the
  release-local `MODEL_TERMS.md`, and the published bytes are hashed in
  provenance;
- official license texts match hash-pinned upstream copies; and
- the tracked source worktree is clean.

By default, large files are hard-linked into the ignored
`artifacts/model-hub/` staging directory when possible. Pass `--copy` for an
independent byte copy. The resulting `SHA256SUMS` covers the complete upload
except for `SHA256SUMS` itself.

`--allow-dirty-source` exists only for local pipeline checks. Do not publish a
release whose `MODEL_PROVENANCE.json` records `source.dirty: true`.

## Stage and upload

Create the Hub repository as private for staging, upload only the generated
directory, review it, and make it public and ungated after validation:

```bash
uvx --from huggingface-hub hf repo create \
  intsuc/Llama-3-ARDY-Mini-Core40-Browser \
  --repo-type model --private --exist-ok

HF_XET_HIGH_PERFORMANCE=1 uvx --from huggingface-hub hf upload \
  intsuc/Llama-3-ARDY-Mini-Core40-Browser \
  artifacts/model-hub/Llama-3-ARDY-Mini-Core40-Browser \
  . --repo-type model
```

After upload, record the resulting full Hub commit SHA. The production app
must use an immutable base URL:

```text
https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser/resolve/<full-commit-sha>/
```

Do not configure `main`, a moving tag, a private repository, or a gated
repository as the browser download origin. Verify the actual Hub URL through
the application's cross-origin-isolated production build, including CORS,
byte ranges, cancel/resume, SHA-256 validation, cache restore, mixed-FP16
selection, and FP32 fallback.

## Update policy

Publish changed model bytes as a new immutable Hub commit and release tag.
Regenerate the complete staging tree so its model card, provenance, and
checksums cannot drift from the artifacts. Production deploys should pin the
new full commit only after the same browser smoke tests pass.

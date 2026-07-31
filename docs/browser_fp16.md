# Browser mixed-FP16 model files

The browser exporter applies a reviewed, graph-specific mixed-FP16 policy
before writing `model.json.gz` and one gzip transport per declared model file.
Public graph inputs and outputs remain FP32, so diffusion state and the
JavaScript runtime contract do not change.
The text encoder and autoregressive denoiser remain byte-identical to their
FP32 exports after continuation-rollout ablation. Only the structured decoder
contains FP16 compute. This mixed-FP16 variant requires WebGPU's native
`shader-f16` feature. It is shipped beside a complete FP32 variant; the
inference worker reports the feature set of the adapter it will use before any
model manifest is requested, and the browser selects mixed FP16 or FP32 from
that result without exposing a precision setting. Both variants use
WebGPU—FP32 is a model fallback, not a CPU or WebAssembly
execution-provider fallback.

## Selected precision policy

| Graph | FP16 regions | FP32 regions |
|---|---|---|
| Text encoder | None | The complete graph and all 111 floating-point initializers |
| Denoiser | None | The complete graph and all 222 floating-point initializers |
| Decoder | Neural input/output projections, value projections, and feed-forward layers | QK score `MatMul_1` → mask `Add_3` → `Softmax`; every `LayerNormalization`; FSQ/root preprocessing, statistics conversion, 6D rotation normalization, rotation conversion, FK, contacts, and world-space processing outside `/decoder/` |

This policy is implemented by
[`ardy/browser/precision.py`](../ardy/browser/precision.py). Conversion uses
ONNX Runtime's transformer FP16 converter with the complete IEEE FP16 range,
keeps FP32 I/O, removes redundant converter casts, restores stable topological
order, and rejects external tensor data. The text encoder and denoiser bypass
conversion and are copied byte-for-byte. This avoids numeric changes and a
rewrite expansion observed when an all-blocked graph was still passed through
the converter. The exporter runs the full ONNX checker and creates CPU
sessions with graph optimizations disabled to exercise the decoder's portable
FP16 primitives.

The implementation follows ONNX Runtime's
[float16 conversion guidance](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).
The mixed variant's deployment requirement comes from WebGPU/WGSL's optional
[`shader-f16` feature](https://gpuweb.github.io/gpuweb/wgsl/); the FP32 variant
declares no optional WebGPU feature.

## Ablation

All component comparisons use the same FP32 ONNX export as their reference.
Candidate motion rollouts use identical prompts, initial noise, ten-step
eta-zero DDIM schedules, browser recentering, and FSQ requantization. Each
model consumes its own generated history, matching independent browser
playback rather than forcing both candidates onto FP32 history.

### Rejected blanket conversion

Converting every internal operation and initializer to FP16 reduced the three
graphs to roughly half their original size, but the denoiser produced NaNs.
The first NaN occurred in the final root Transformer layer: the FP16 QK score
matrix overflowed before `Softmax`. Keeping `Softmax` alone in FP32 could not
recover values that had already overflowed. This established the minimum
contiguous FP32 attention island: score `MatMul` → scale (when present) →
mask addition → `Softmax`.

### Text encoder

Isolated condition-vector error was not a safe proxy for autoregressive motion
quality. The least aggressive text candidate with a material size reduction
therefore received the full 64-prompt × 3-seed × 5-window audit. It converted
only the word-embedding table to FP16; positional and token-type embeddings,
all six Transformer layers, pooling, adapter, and root/body heads remained
FP32. Its conditions were extremely close to the reference (RMSE
`1.544e-4`, minimum cosine `0.99999982`), but one continuation diverged:

| Scope | MPJPE mean | MPJPE p95 | MPJPE maximum |
|---|---:|---:|---:|
| Accumulated 200 frames | 8.496 mm | 24.604 mm | 735.802 mm |
| Fifth window | 32.558 mm | 82.756 mm | 3,279.665 mm |

One continuation at seed `12031` also reached `125.28°`
global-rotation error and only `30.625%` contact agreement. More aggressive
text-FP16 candidates could not provide a stronger tail guarantee, so the
selected 112,430,592-byte text graph is copied exactly from the FP32 export.
Its condition RMSE, MAE, and maximum absolute error are all zero (the reported
cosine mean is one).

### Autoregressive denoiser

Single-window tests initially favored a denoiser with FP32 attention scores,
normalization, output heads, and CFG blending while the remaining compute used
FP16. A subsequent 64-prompt × 3-seed × 5-window audit rejected it:
first-window MPJPE was 5.21 mm mean / 13.22 mm p95, but fifth-window MPJPE
reached 516 mm p95 and accumulated MPJPE reached 53.99 mm mean / 188.62 mm
p95. Generated output is fed back as the next window's history, so a small
one-step error was not a safe selection metric.

Keeping only the first layer, last layer, both boundary layers, timestep path,
input path, root branch, body branch, or the last compute stage in FP32 was
also tested. Those partial denoiser policies still produced 43.5–135.2 mm mean
MPJPE on the screening set. Together with the NaN-producing blanket conversion,
these continuation results selected an exact FP32 identity graph for the
denoiser.

### Rejected FP16-storage / FP32-compute denoiser

Two additional candidates stored denoiser weights in FP16 and inserted casts
back to FP32 before every original computation:

| Storage candidate | Denoiser ONNX bytes | Accumulated MPJPE p95 | Fifth-window MPJPE p95 | Fifth-window maximum |
|---|---:|---:|---:|---:|
| All 222 initializers FP16 | 296,233,763 | 116.13 mm | 323.29 mm | 501.42 mm |
| 71 initializers ≥ 1 MiB FP16; small weights FP32 | 297,506,584 | 41.34 mm | 91.19 mm | 946.70 mm |

The selective candidate improved p95 but had a 94.7 cm fifth-window outlier.
More importantly, ONNX Runtime's production graph optimization
constant-folded the FP16-to-FP32 casts and expanded the denoiser back to
approximately 590.7 MB. It therefore reduced download size without providing
a dependable runtime weight-memory or speed benefit, while still quantizing
the weights. Both storage-only variants were rejected.

### Decoder

The decoder pilot used hybrid tokens from real FP32 DDIM rollouts:

| Candidate | ONNX bytes | MPJPE | Global rotation error | Root/contact |
|---|---:|---:|---:|---:|
| QK + LayerNorm FP32 | 36,105,244 | 0.873 mm | 0.614° | Small root drift; contacts matched |
| Above + wrapper geometry FP32 | 36,181,508 | 0.326 mm | 0.081° | Root exact; contacts matched |
| Above + neural output projection FP32 | 37,532,000 | 0.323 mm | 0.080° | Root exact; contacts matched |

The output projection cost 1.35 MB for no material motion improvement, so it
remains FP16.

## Final 64 × 3 × 5 evaluation

The adopted decoder-only mixed-FP16 model was evaluated on 64 frozen test
prompts from the pinned NVIDIA SEED Timeline Annotations corpus, three fixed
seeds, and five consecutive 40-frame windows (192 paired 200-frame motions;
960 windows total). These are fidelity errors relative to the FP32 browser
graphs, not paper-comparable semantic motion metrics:

| Metric and scope | Mean | p95 | Maximum |
|---|---:|---:|---:|
| MPJPE, accumulated 200 frames | 0.265241 mm | 0.349735 mm | 0.409339 mm |
| MPJPE, fifth window | 0.259722 mm | 0.350215 mm | 0.412430 mm |
| Global rotation, accumulated | 0.076471° | 0.090211° | 0.101946° |
| Root ADE, accumulated | 0 | 0 | 0 |
| Root FDE, accumulated | 0 | 0 | 0 |

Accumulated contact agreement was `99.998046875%`, with F1
`0.99998785` and IoU `0.99997569`. Text conditions and denoiser outputs were
exactly equal to their FP32 references; all measured difference came from the
decoder.

The Git-tracked machine-readable aggregate, including prompt-corpus
provenance, content hashes, evaluation parameters, per-window distributions,
model identities, and the Python/ONNX Runtime/CPU evaluation environment, is
[`reports/browser_fp16_ablation.json`](../reports/browser_fp16_ablation.json).
It deliberately excludes prompt text, local absolute paths, and per-case
worst-case attribution. The detailed local report retains those diagnostics
under the ignored `artifacts/` directory.

Reproduce the paired evaluation with:

```bash
uv run --extra browser python scripts/export_browser_models.py \
  --checkpoints-dir checkpoints \
  --minilm-artifact artifacts/minilm-ardy-core40 \
  --output-directory artifacts/browser/ardy-minilm-core40-browser-v1

uv run --extra browser python scripts/evaluate_browser_fp16.py \
  --reference-dir \
    artifacts/browser/ardy-minilm-core40-browser-v1/fp32 \
  --candidate-dir artifacts/browser/ardy-minilm-core40-browser-v1/fp16 \
  --prompts artifacts/data/prompts-core40-timeline.jsonl \
  --prompt-metadata artifacts/data/prompts-core40-timeline.metadata.json \
  --split test \
  --count 64 \
  --seeds 12031,987654,20260729 \
  --cfg-weight 3.5 \
  --windows 5 \
  --output artifacts/evaluation/browser-fp16-timeline-detailed.json \
  --public-output reports/browser_fp16_ablation.json
```

`--public-output` requires the canonical NVIDIA Timeline provenance sidecar.
The evaluator validates that sidecar against the complete prompt-manifest
filename, SHA-256, row count, and split counts. The public aggregate records
that identity and the digest of the deterministically selected prompts, so the
selection can be reproduced without checking prompt bodies into Git.
Before creating any ONNX Runtime session, it decompresses the bounded
`model.json.gz`, validates the fixed format and schema, rejects unsafe or
duplicate paths and undeclared directory entries, and checks the compressed
and raw size and SHA-256 of every declared transport. It materializes verified
raw files in an evaluator-owned temporary directory, then validates the
complete FP32 reference graphs, the production mixed-precision policy, public
graph I/O, and the absence of external ONNX data. Every compressed input is
prehashed and rechecked after use; any non-finite inference output or metric
aborts report creation.

## Size result

| Distribution total | FP32 bytes | Mixed-FP16 bytes | Saved |
|---|---:|---:|---:|
| Raw manifest and model files | 775,579,900 | 740,128,936 | 35,450,964 |
| `model.json.gz` and per-file gzip transports | 717,533,539 | 684,221,164 | 33,312,375 |

The mixed-FP16 distribution is 4.6427% smaller after per-file gzip, saving
31.77 MiB (0.0310 GiB) on transfer. The production policy keeps the text
encoder and denoiser byte-identical to FP32 and applies mixed FP16 only to the
decoder. The browser presents neither precision label; the automatically
selected variant changes the download from 717,533,539 bytes (684.29 MiB) to
684,221,164 bytes (652.52 MiB) on adapters with `shader-f16`. In the recorded
ablation, the decompressed candidate manifest SHA-256 is
`07a0cb3984cc324e9318549d03494d3fa7193727367ae45b2087ca8a978b6d0b`;
the matching FP32 reference manifest SHA-256 is
`982095622886845ed985f353ab5e3806db5da74f3aa0b85acb0415e81e48e5a1`.
The report records these portable manifest identities together with the model
ID, immutable revision, and measured raw/transport totals; it records no local
directory path. These are verified file/transfer reductions, not a claim that
peak GPU memory drops by the same amount. Runtime allocations also include
FP32 regions, activations, staging buffers, and browser/driver overhead.

CPU FP16 inference is intentionally not used by the application and was slower
than FP32 in the evaluation environment: the candidate took `1.30087`
seconds per five-window case versus `1.23139` seconds for the FP32 reference.
This is diagnostic only. Actual deployment timing must use the opt-in WebGPU
Playwright test documented in [`browser_demo.md`](browser_demo.md).

## WebGPU validation in this environment

The automatic selection and real-model path were exercised with Chromium
`151.0.7922.34` (Dawn
`583f3600453cc982a3dac39308cac8939875d7af`) on an NVIDIA GB10, Vulkan
`1.4.312`, driver `580.159.3.0`. The inference adapter did not expose optional
`shader-f16`, so the app selected only the FP32 model-family directory. It
downloaded and verified the real transports, initialized the decoder, text
encoder, and denoiser with ONNX Runtime WebGPU, and generated 40 frames without
a page error or console error. The complete Playwright test took about 15
seconds across repeated runs against the local development model server.

The same non-persistent Chromium configuration also retained all 717,512,045
declared FP32 file-transport bytes as 93 Cache Storage entries with no entry
larger than 8 MiB, reconstructed the 547,498,825-byte compressed denoiser, and
reloaded its 590,701,706 raw bytes while offline. This verifies the bounded
cache representation used for large graphs. Mixed-FP16 WebGPU timing and GPU
memory still require a browser/driver combination that exposes `shader-f16`;
the CPU ablation above remains the fidelity basis for that variant.

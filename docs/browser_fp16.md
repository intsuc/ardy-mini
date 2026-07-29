# Browser mixed-FP16 model pack

The browser exporter applies a reviewed, graph-specific mixed-FP16 policy
before writing the gzip model pack. Public graph inputs and outputs remain
FP32, so diffusion state and the JavaScript runtime contract do not change.
The text encoder and autoregressive denoiser remain byte-identical to their
FP32 exports after continuation-rollout ablation. Only the structured decoder
contains FP16 compute. The browser requires WebGPU's native `shader-f16`
feature and does not fall back to another backend.

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
The deployment requirement comes from WebGPU/WGSL's optional
[`shader-f16` feature](https://gpuweb.github.io/gpuweb/wgsl/).

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

The worst case was prompt `walk ff stop 180 slow` with seed `12031`. Its fifth
window also reached `125.28°` global-rotation error and only `30.625%` contact
agreement. More aggressive text-FP16 candidates could not provide a stronger
tail guarantee, so the selected 112,430,592-byte text graph is copied exactly
from the FP32 export. Its condition RMSE, MAE, and maximum absolute error are
all zero (the reported cosine mean is one).

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

The adopted decoder-only mixed-FP16 pack was evaluated on 64 frozen test
prompts, three fixed seeds, and five consecutive 40-frame windows (192 paired
200-frame motions; 960 windows total). These are fidelity errors relative to
the FP32 browser graphs, not paper-comparable semantic motion metrics:

| Metric and scope | Mean | p95 | Maximum |
|---|---:|---:|---:|
| MPJPE, accumulated 200 frames | 0.278312 mm | 0.351433 mm | 0.464571 mm |
| MPJPE, fifth window | 0.278785 mm | 0.361916 mm | 0.598959 mm |
| Global rotation, accumulated | 0.079321° | 0.095760° | 0.106333° |
| Root ADE, accumulated | 0 | 0 | 0 |
| Root FDE, accumulated | 0 | 0 | 0 |

Accumulated contact agreement was `99.994140625%`, with F1
`0.99996181` and IoU `0.99992362`. Text conditions and denoiser outputs were
exactly equal to their FP32 references; all measured difference came from the
decoder.

The complete machine-readable result, including per-window distributions and
worst cases, is
[`reports/browser_fp16_ablation.json`](../reports/browser_fp16_ablation.json).

Reproduce the paired evaluation with:

```bash
uv run --extra browser python scripts/evaluate_browser_fp16.py \
  --reference-pack /path/to/fp32-reference.tar.gz \
  --candidate-pack artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz \
  --prompts artifacts/data/prompts-core40-expanded-frozen-eval.jsonl \
  --split test \
  --count 64 \
  --seeds 12031,987654,20260729 \
  --cfg-weight 3.5 \
  --windows 5 \
  --output reports/browser_fp16_ablation.json
```

## Size result

| Asset | FP32 bytes | Mixed-FP16 bytes | Saved |
|---|---:|---:|---:|
| Text encoder | 112,430,592 | 112,430,592 | 0 |
| Denoiser | 590,701,706 | 590,701,706 | 0 |
| Decoder | 71,642,198 | 36,181,508 | 35,460,690 |
| ONNX total | 774,774,496 | 739,313,806 | 35,460,690 |
| Gzip model pack | 718,137,762 | 684,835,577 | 33,302,185 |

The decoder is 49.497% smaller. This makes the total ONNX payload 4.5769%
smaller and the final gzip pack 4.6373% smaller, saving 33.82 MiB of ONNX data
and 31.76 MiB (0.0310 GiB) on transfer. The candidate archive SHA-256 is
`145995ff6216076d2ee06d7a62a741c6d9a02434278d645a0446cd357aa95868`;
the FP32 reference archive SHA-256 is
`4962bc2c3b7135e8181de1229fc816924ebcd60e3d5ff1d9f5cc02b8505e8663`.
These are verified file/transfer reductions, not a claim that peak GPU memory
drops by the same amount. Runtime allocations also include FP32 regions,
activations, staging buffers, and browser/driver overhead.

CPU FP16 inference is intentionally not used by the application and was slower
than FP32 in the evaluation environment: the candidate took `1.31319`
seconds per five-window case versus `1.25773` seconds for the FP32 reference.
This is diagnostic only. Actual deployment timing must use the opt-in WebGPU
Playwright test documented in [`browser_demo.md`](browser_demo.md).

## WebGPU validation in this environment

The runtime feature gate and model-load order were exercised with Chromium
`151.0.7922.34` (Dawn
`583f3600453cc982a3dac39308cac8939875d7af`) on an NVIDIA GB10, Vulkan
`1.4.312`, driver `580.159.3.0`. The adapter did not expose optional
`shader-f16`, so the application correctly rejected the pack before reading or
decompressing it. NVIDIA and SwiftShader adapters, the default and Vulkan
ANGLE paths, three power preferences, core/compatibility requests, and unsafe
WebGPU developer flags were checked; all reported `shader-f16=false`. The
preflight produced no console error or page error, and its seven focused
runtime tests passed.

Consequently, this environment can validate the converted graphs through CPU
ONNX Runtime and validate the browser feature gate, but it cannot report
mixed-FP16 WebGPU inference time or GPU memory. Those measurements require a
browser/driver combination that actually exposes `shader-f16`.

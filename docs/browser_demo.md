# Fully in-browser MiniLM Core40 demo

The browser demo runs the complete prompt-to-motion path in the client:
WordPiece tokenization, the specialized MiniLM encoder, ten DDIM denoising
steps per Core40 window, autoregressive recentering and FSQ requantization,
motion decoding, Core27 forward kinematics, and 3D playback. No inference API
or Python process is used after the static page and a local model pack have
loaded.

The browser v1 scope is intentionally narrower than the server-backed
interactive demo. It supports the trained student's compatible checkpoint,
`ARDY-Core-RP-20FPS-Horizon40`, typo-free English prompts, text-only
generation, and 2–10 second clips at 20 FPS. Kinematic constraints, motion
correction, live prompt switching, and session import/export remain in the
Python interactive demo.

## Architecture

Inference runs in a dedicated Web Worker so model loading and generation do
not block the UI. ONNX Runtime Web selects WebGPU first and retries with its
WebAssembly execution provider when WebGPU is unavailable or model session
creation fails.

| Graph | Browser input | Browser output |
|---|---|---|
| `text_encoder.onnx` | WordPiece IDs and masks | one direct 2,048-D root/body condition |
| `denoiser.onnx` | fixed 40-frame history + 40-frame generation window | clean 148-D hybrid tokens |
| `decoder.onnx` | hybrid tokens and accumulated root translation | normalized motion and Core27 joint positions |

The JavaScript runtime supplies explicit seeded Gaussian noise and implements
ARDY's deterministic DDIM update. A 40-frame tail is recentered and
requantized between windows, matching the Python Core40 generation path. The
main thread receives only the final motion and joint arrays for three.js
playback.

## Export a local model pack

Model weights are deliberately not committed to this repository. First place
the separately obtained Core40 checkpoint under `checkpoints/` and produce the
trained MiniLM artifact described in
[the encoder guide](minilm_encoder.md). Then install the exporter through
`uv` and create the pack:

```bash
uv sync --extra browser

uv run --extra browser python scripts/export_browser.py \
  --checkpoints-dir checkpoints \
  --minilm-artifact artifacts/minilm-ardy-core40 \
  --output-dir artifacts/browser/core40
```

The exporter checks all three ONNX graphs and compares their outputs with
PyTorch through ONNX Runtime CPU. `manifest.json` records graph I/O names,
tensor dimensions, diffusion and quantization constants, skeleton metadata,
file sizes, SHA-256 digests, compatibility, and model notices.

The measured FP32 pack produced in this environment is 836,704,265 bytes
(798.0 MiB):

| Asset | Bytes |
|---|---:|
| MiniLM condition encoder | 112,430,592 |
| ARDY text-only CFG denoiser | 651,936,916 |
| Motion decoder and Core27 FK | 71,624,514 |
| Tokenizer files | 712,243 |

The final export verification errors were `1.91e-5` for text conditions,
`9.98e-6` for the denoiser, `1.51e-3` for normalized motion, and
`1.23e-4 m` maximum for posed joints. These compare the ONNX graphs with the
same deterministic PyTorch attention path used during export.

## Run the demo

Install the pinned browser dependencies and start Vite:

```bash
cd web
npm ci
npm run dev
```

Open the printed localhost URL, choose **Import model pack**, and select
`artifacts/browser/core40`. The app verifies every declared file before
creating an inference session. If browser storage has enough capacity, the
validated pack is copied to the origin-private file system for later visits.

Chrome or Edge with WebGPU is strongly recommended. WebGPU requires HTTPS or
localhost. The WebAssembly fallback works without a supported GPU but is a
compatibility path for this roughly 0.84 GB model, not a real-time guarantee.
Allow ample device memory and at least about 0.9 GB of persistent browser
storage.

The Vite development and preview servers send:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

This enables multithreaded WebAssembly when the browser supports it. Without
cross-origin isolation, the runtime safely selects one WASM thread. A
production static host should send the same headers and serve all JavaScript,
ONNX Runtime `.mjs`/`.wasm` files, and any remotely hosted model assets from
the same origin. `npm run build` copies the version-matched ONNX Runtime 1.27.0
assets and emits the static app under `web/dist/`; it does not copy model
weights. The build also copies source and browser-runtime license notices into
`web/dist/notices/`. The measured static shell is about 28 MB, dominated by
the pinned 24.3 MB Asyncify WASM binary.

## Validation

```bash
uv run --extra browser --with pytest python -m pytest -q \
  tests/test_browser_export.py

cd web
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

The unit tests cover the export wrappers, manifest contract, tokenizer and
model-pack validation, portable PRNG, DDIM math, autoregressive masks,
recentering, requantization, and viewer controls. The Playwright test exercises
the static browser shell without requiring licensed weights. A real-pack
test is opt-in:

```bash
ARDY_BROWSER_MODEL_PACK=../artifacts/browser/core40 \
ARDY_BROWSER_BACKEND=wasm \
npm run test:e2e -- e2e/real-model.spec.ts

ARDY_BROWSER_MODEL_PACK=../artifacts/browser/core40 \
ARDY_BROWSER_BACKEND=webgpu \
npm run test:e2e -- e2e/real-model.spec.ts
```

### Measurements from this environment

The real-pack test passed in Chromium 151 with cross-origin isolation enabled.
It loaded and hash-checked all 836,704,265 model-pack bytes, created all three
sessions, generated a finite `[1,40,330]` motion tensor plus
`[1,40,27,3]` joints, and rendered the result:

| Forced backend | Model verification + session load | 40-frame runtime | Notes |
|---|---:|---:|---|
| WebAssembly | 3.46 s | 0.72 s | browser reported 20 logical cores |
| Native NVIDIA WebGPU | 3.56 s | 0.86 s cold, 0.21 s warm | hardware adapter, not a software fallback |

The WebGPU execution-provider request is made without a second requested
provider, so failure to acquire a real adapter is observable. `auto` then
creates separate WASM sessions and reports WebAssembly truthfully. This
fallback was also exercised in headless Chromium without the required GPU
launch configuration.

For the same prompt, seed, and explicit input noise, the browser WebGPU and
WASM results differed by at most `1.48e-4` in normalized motion and
`3.82e-6 m` in joint coordinates. An additional two-window (80-frame)
WASM-vs-PyTorch parity run measured mean MPJPE `5.04e-5 m`, p95 MPJPE
`3.01e-4 m`, and maximum joint-coordinate error `1.02e-3 m`.

The Chromium WASM run increased summed browser-process RSS by approximately
2.54 GiB after model loading. Treat that as an environment-specific
whole-browser measurement rather than an ONNX tensor-only requirement; browser
version, execution provider, thread count, and allocator behavior all affect
it.

## Distribution and trust

The model pack contains the separately licensed ARDY checkpoint and locally
trained MiniLM weights. Keeping it under the ignored `artifacts/` directory
does not grant redistribution rights. Review
[the third-party model and data notices](../THIRD_PARTY_MODELS_AND_DATA.md)
before sharing a pack.

A pack's SHA-256 manifest detects corruption after export; it is not a digital
signature. Import model packs only from a source you trust.

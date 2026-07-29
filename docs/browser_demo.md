# Fully in-browser MiniLM Core40 app

The browser app runs ARDY Mini locally from prompt to playback: WordPiece
tokenization, the specialized MiniLM condition encoder, deterministic DDIM
sampling, optional kinematic constraints, autoregressive
recentering/requantization, structured motion decoding, optional JavaScript
postprocessing, and three.js visualization. After the static page and a local
model pack have loaded, no Python process or inference API is involved.

The interface is a deliberately simple technical demo rather than a marketing
page or a feature-for-feature copy of the Python/Viser application. Its three
working areas keep prompt/session controls, the 3D preview and playback
timeline, and the less frequently used planning/output controls visibly
separate.

The supported model artifact is intentionally narrow:

- `ARDY-Core-RP-20FPS-Horizon40`;
- the MiniLM student trained specifically for that checkpoint;
- well-formed, typo-free English motion prompts;
- 20 FPS and a 40-frame (2-second) generation horizon.

One request may generate 40–200 frames. A browser generation session can grow
beyond that by appending additional 40-frame chunks.

## What is available in the browser

### Streaming and session editing

The worker decodes and emits each generated chunk instead of waiting for the
entire requested clip. Core40 produces at most 40 new frames per window.

| Operation | Effect |
|---|---|
| Replace / **Restart** | Reset the random stream, initial transform, history, and generated motion, then generate a new session. |
| Append / continuous generation | Continue from the current session end while retaining the selected recent history. |
| Branch / **Restart from now** | Discard motion after the playhead and continue from that point. |
| **Apply live** prompt | Preserve motion through the configured replan buffer, discard the later future, and continue statefully with the updated prompt. |

ARDY hybrid tokens represent four motion frames. A branch therefore rounds
down to the nearest complete four-frame token. For example, branching at frame
19 continues from frame 16. The browser has no motion encoder with which to
re-encode an incomplete token.

The initial transform supplies root X/Z translation and heading before the
first window. The history control selects up to the preceding 40 frames.
Future crop controls how far beyond the current 40-frame generation horizon
constraints are exposed to the constraint graph, subject to the fixed
200-frame conditioned graph capacity.

### Constraints and guidance

The constraint-aware denoiser (one of the pack's four ONNX graphs) provides
the separated conditioning categories needed for interactive planning:

- root position and optional heading;
- sparse or densely interpolated root waypoints/trajectory;
- full-body pose keyframes captured from generated motion;
- left/right hand and left/right foot end-effector (EE)
  position/orientation constraints;
- start/end constraint intervals and constraints beyond the current generation
  horizon;
- root waypoint sequences derived from a target speed and heading, with the
  same two-second current-to-target velocity transition used by Viser.

With three or more root waypoints, **Dense trajectory** applies a bounded
browser-native smoothing pass after interpolation. It keeps every sample
within the native path smoother's six-centimetre deviation envelope while
removing sharp corners.

Text CFG and constraint CFG have independent controls. A window with no active
kinematic constraint uses the unconstrained denoiser graph. A window
with active constraints uses the constraint-aware graph with separate
history, generation, and future masks.

Constraints guide generation; they do not turn the diffusion model into a
general inverse-kinematics solver. In particular, the browser does not
implement the native Viser demo's rotation-space IK correction. End-effector
positions and rotations are diffusion-conditioning targets. The optional
lightweight postprocess can tighten root/full-body position targets and reduce
visible foot sliding after decoding.

### Structured output and viewer

The decoder can return all of the following for every emitted chunk:

- normalized `[T, 330]` ARDY motion features;
- world-space joint positions `[T, J, 3]`;
- local and global joint rotation matrices `[T, J, 3, 3]`;
- root positions and global root headings;
- four predicted foot-contact channels.

The viewer consumes dynamic skeleton names, parents, root index, and contact
metadata instead of assuming one fixed topology. The current compatible pack
describes Core27, while imported motion/session data may carry another
validated skeleton. Display controls expose the skeleton, root trajectory,
predicted contacts, joint orientation axes, constraints, initial transform,
and waypoints. The optional **Body proxy** is generated directly from the
active skeleton as joint spheres and bone capsules, so it adapts to validated
dynamic skeleton metadata without an external character asset.

The body proxy is not an SMPL body, a skinned character mesh, or a replacement
for a production character renderer. The browser app does not import scene
meshes. Its reference overlay instead accepts a compatible structured motion
JSON or browser session and draws a time-aligned reference skeleton; no
separate mesh asset is required.

### Browser postprocessing versus native correction

The optional browser postprocess is TypeScript/JavaScript code applied to
decoded joint/root positions. It can:

- blend horizontal root corrections around constrained frames;
- preserve full-body root targets;
- reduce contact-run foot sliding with bounded whole-body translations;
- report root-error and foot-sliding metrics without mutating its input.

It is not the native C++ motion-correction extension used by the Python
pipeline, and that extension is not embedded in the ONNX pack. The manifest
therefore reports `motion_correction_included: false`. Do not treat the
browser postprocess and native correction as numerically equivalent.

## Architecture

Inference runs in a dedicated Web Worker so ONNX execution does not block the
main UI. ONNX Runtime Web selects WebGPU first in `auto` mode and creates
separate WebAssembly sessions if WebGPU session creation fails.

| Graph | Main inputs | Outputs |
|---|---|---|
| `text_encoder.onnx` | WordPiece IDs, attention mask, token types | direct 2,048-D root/body condition (two 1,024-D branches) |
| `denoiser.onnx` | text CFG, up to 40 history frames, 40 generation frames, text condition, timestep | clean 148-D hybrid tokens for unconstrained windows |
| `denoiser_constraints.onnx` | independent text/constraint CFG, history/generation/future masks, text condition, sparse observed motion | clean 148-D hybrid tokens for constrained windows |
| `decoder.onnx` | hybrid tokens, valid-token mask, accumulated root translation | normalized motion, joints, local/global rotations, roots/headings, contacts |

The JavaScript runtime supplies a reproducible portable Gaussian random stream
and implements ARDY's ten-step, eta-zero DDIM update. Between windows, it
retains global hybrid tokens, recenters the latest history, and requantizes
the latent body features with the manifest's FSQ constants.

The worker protocol supports replace, append, branch, chunk progress,
continuation restore, future constraints, rich motion arrays, and capability
reporting. Typed-array snapshots are transferred to the main thread so
streaming cannot detach state that the worker still needs.

## Export a local model pack

Model weights are deliberately absent from the Git repository and static web
build. First obtain the compatible Core40 checkpoint under `checkpoints/` and
train or otherwise produce the compatible MiniLM artifact described in
[the encoder guide](minilm_encoder.md).

From the repository root:

```bash
uv sync --extra browser

uv run --extra browser python scripts/export_browser.py \
  --checkpoints-dir checkpoints \
  --minilm-artifact artifacts/minilm-ardy-core40 \
  --output-dir artifacts/browser/core40
```

The exporter checks all four ONNX files and, unless `--skip-verify` is passed,
compares each graph with its PyTorch source through ONNX Runtime CPU.
`manifest.json` records graph contracts, tensor dimensions, diffusion and
quantization constants, normalization statistics, motion layout, skeleton
metadata, capabilities, file sizes, SHA-256 digests, model compatibility, and
license notices.

The measured FP32 payload in this environment is 1,488,773,547 bytes
(approximately 1.39 GiB, or 1.49 GB decimal):

| Asset | Bytes |
|---|---:|
| MiniLM condition encoder | 112,430,592 |
| Unconstrained ARDY denoiser | 651,936,916 |
| Constraint-aware ARDY denoiser | 652,051,598 |
| Structured motion decoder | 71,642,198 |
| Tokenizer files | 712,243 |

Small exporter/version differences can change the exact byte count. Plan for
at least 1.6 GB of origin storage if the app is allowed to persist a validated
copy, and substantially more working memory while four inference sessions are
loaded.

## Run the app

Install the pinned browser dependencies and start Vite:

```bash
cd web
npm ci
npm run dev
```

Open the printed localhost URL, choose **Choose model pack**, and select the
`artifacts/browser/core40` directory. The app validates every file declared in
the manifest before creating inference sessions. When origin storage is
available, it can retain the validated pack in the origin-private file system
for later visits.

The header reports the backend that actually loaded:

- **WebGPU** is preferred and requires a supporting browser plus HTTPS or
  localhost.
- **WebAssembly** is the compatibility fallback. It can be substantially
  slower and is not a real-time guarantee for this model size.

The preview supports drag/swipe orbit, wheel/pinch zoom, and keyboard
operation. With the preview focused, Space toggles playback, Left/Right seeks,
Shift+Arrow orbits, Plus/Minus zooms, and Home resets the camera. Generate uses
Command+Enter on Apple platforms and Control+Enter elsewhere.

`prefers-reduced-motion: reduce` disables automatic motion playback and
looping; a new result opens paused at frame zero and remains manually
playable. Viewer rendering also stops when neither playback nor camera
damping requires another frame, and pauses its clock while the page is hidden.

For a production build:

```bash
cd web
npm test
npm run build
npm run preview
```

`web/dist/` contains the app, the pinned ONNX Runtime browser assets, source
and runtime notices, and generated third-party license information. It does
not contain the model pack.

## Session and motion I/O

Browser persistence uses explicit, versioned, validated data structures—not
pickle or executable Python objects.

- Session import accepts the versioned browser-session JSON schema and the
  binary `.ardysession` container. The UI exports `.ardysession`, which stores
  motion, skeleton metadata, exact normalized constraint values/masks, editor
  state, waypoints, initial transform, output visibility, provenance, and
  continuation data with little-endian typed-array payloads. This avoids JSON
  number expansion for large arrays.
- When a compatible continuation payload is present, append/branch generation
  can resume. Motion-only or incompatible imports remain playback-capable but
  require a generation restart.
- **Export motion** downloads both a structured JSON file—which preserves
  normalized motion, rotations, roots, joints, contacts, and skeleton
  metadata—and a flat CSV with frame/time, per-joint XYZ positions, root
  components, and contact channels.
- **Import reference** accepts a structured motion JSON or a browser session.
  The reference must use a skeleton compatible with the active clip.

Import reconstructs known fields and validates versions, shapes, finite
values, skeleton topology, array sizes, constraint ranges, and continuation
dimensions. Unknown objects are not evaluated. Downloads are assembled with
browser `Blob` URLs and remain local.

## Privacy and hosting

Prompts, seeds, constraints, generation state, imported sessions, and generated
motion remain in the browser. Model selection reads local files; inference
does not upload them. The app has no inference service dependency.

Vite development and preview send:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These headers enable cross-origin isolation and multithreaded WASM where the
browser supports it. Without isolation, the runtime selects one WASM thread.
A production static host should send the same headers and serve the app, ONNX
Runtime `.mjs`/`.wasm` assets, and any hosted model assets from compatible
origins. WebGPU itself requires a secure context.

## Validation

Run exporter and browser tests with:

```bash
uv run --extra browser --with pytest python -m pytest -q \
  tests/test_browser_export.py

cd web
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

The current exported pack records these maximum PyTorch-versus-ONNX Runtime
CPU errors:

| Output | Maximum absolute error |
|---|---:|
| MiniLM text conditions | `1.91e-5` |
| Unconstrained denoiser tokens | `9.98e-6` |
| Constraint-aware denoiser tokens | `1.18e-4` |
| Normalized motion | `1.51e-3` |
| Posed joints | `1.23e-4 m` |
| Local rotations | `9.36e-4` |
| Global rotations | `1.09e-3` |
| Root positions / headings / contacts | `0` |

Browser runtime and memory measurements are device-, browser-, driver-, and
execution-provider-specific. Measurements made before constraint-graph support
do not describe the present pack and should not be used as requirements.

An opt-in real-pack Playwright run can force either provider:

```bash
cd web

ARDY_BROWSER_MODEL_PACK=../artifacts/browser/core40 \
ARDY_BROWSER_BACKEND=wasm \
npm run test:e2e -- e2e/real-model.spec.ts

ARDY_BROWSER_MODEL_PACK=../artifacts/browser/core40 \
ARDY_BROWSER_BACKEND=webgpu \
npm run test:e2e -- e2e/real-model.spec.ts
```

Set `ARDY_BROWSER_REDUCED_MOTION=1` to exercise paused initial playback.

## Current limitations

- The trained MiniLM condition heads are checkpoint-specific. This pack is not
  interchangeable with Core8, G1, SOMA, or another 2,048-D model.
- Prompt support is limited to well-formed, typo-free English motion
  descriptions.
- Branching crops to the preceding complete four-frame token; a partial token
  cannot be continued exactly.
- Each generation call is limited to 40–200 frames. Longer sessions are built
  through append/streaming operations.
- The optional JavaScript position postprocess is not the native C++ motion
  correction used by Python ARDY and does not implement the native Viser
  rotation-space IK correction.
- The pack supplies Core27 skeleton metadata and no character mesh. The
  built-in body proxy is a joint-driven sphere/capsule visualization—not SMPL
  or a skinned mesh—and the browser has no scene-mesh importer.
- Reference visualization is a compatible motion/session skeleton overlay, not
  a reference character mesh.
- WASM is a fallback, not a performance promise. Four large graph sessions can
  require considerably more RAM than the on-disk pack.

## Distribution and trust

Repository-authored source and static-shell code are Apache-2.0 subject to the
retained notices and attributions; bundled dependencies retain their own
licenses. The source license does not grant rights to redistribute the ARDY
checkpoint, the locally trained MiniLM weights, training data, teacher
artifacts, or a model pack containing them.

Keep local exports under the ignored `artifacts/` directory unless you have
separately reviewed and satisfied every applicable model and data term. See
[THIRD_PARTY_MODELS_AND_DATA.md](../THIRD_PARTY_MODELS_AND_DATA.md) before
sharing a pack.

The manifest's SHA-256 hashes detect corruption and file substitution relative
to that manifest; they are not a digital signature. Import packs only from a
source you trust.

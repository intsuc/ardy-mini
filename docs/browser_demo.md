# Fully in-browser MiniLM Core40 app

The browser app runs ARDY Mini locally from prompt to playback: WordPiece
tokenization, the specialized MiniLM condition encoder, deterministic DDIM
sampling, autoregressive recentering/requantization, structured motion
decoding, and three.js visualization. After the static page and a local model
pack have loaded, no Python process or inference API is involved.

The interface is a deliberately simple technical demo rather than a marketing
page or a feature-for-feature copy of the Python/Viser application. It has two
working areas: the **Input** pane contains model, prompt, clip, and generation
controls; the **Output** pane contains view settings, the 3D preview, and the
playback timeline. The panes stack vertically at narrow viewport widths.

The shell uses React and shadcn/ui preset `buFzUhO`: Lyra components, the
neutral theme, Noto Sans, and Tabler icons on Tailwind CSS v4. Stock shadcn
component styles own regular controls and surfaces. Custom CSS is limited to
the two-pane workspace, canvas, native range/switch controls, and responsive
behavior specific to this technical demo.

The supported model artifact is intentionally narrow:

- `ARDY-Core-RP-20FPS-Horizon40`;
- the MiniLM student trained specifically for that checkpoint;
- well-formed, typo-free English motion prompts;
- 20 FPS and a 40-frame (2-second) generation horizon;
- the current structured-output browser contract (`runtime.contract_revision`
  `3`, model-pack schema `2`) with decoder-local and decoder-global rotation
  tracks.

The form accepts any non-empty prompt of at most 280 characters, but that
validation does not expand the trained model's supported language. Prompts
outside well-formed English are accepted as input with no quality guarantee.

One request may generate 40–200 frames. A browser generation session can grow
beyond that by appending additional 40-frame chunks.

## What is available in the browser

### Streaming and replanning

The worker decodes and emits each generated chunk instead of waiting for the
entire requested clip. Core40 produces at most 40 new frames per window.

| Operation | Effect |
|---|---|
| Replace / **Restart** | Reset the random stream, fixed zero root transform, history, and generated motion, then generate a new session. |
| Append / continuous generation | Continue from the current session end while retaining up to 40 recent history frames. |
| Branch / **Restart from now** | Discard motion after the playhead and continue from that point. |
| **Apply live** prompt | Preserve motion through a fixed 20-frame replan buffer, discard the later future, and continue statefully with the updated prompt. |

ARDY hybrid tokens represent four motion frames. A branch therefore rounds
down to the nearest complete four-frame token. For example, branching at frame
19 continues from frame 16. The browser has no motion encoder with which to
re-encode an incomplete token.

### Fixed generation policy

The browser UI intentionally has no advanced **Motion parameters** panel.
Generation commands use these fixed internal values:

| Setting | Internal value |
|---|---:|
| Text CFG | `3.5` |
| History | up to `40` frames, clamped to the manifest capacity |
| Live-prompt replan buffer | `20` frames |
| Automatic-extension threshold | `10` frames |
| Initial root translation / heading | `[0, 0, 0]` / `0` radians |

Duration (2–10 seconds), seed, backend, continuous generation, and its target
buffer remain normal user controls. The browser model and worker contracts do
not contain a constraint graph or constraint-generation inputs.

The browser application does not expose or apply root/full-body/end-effector
constraints, waypoints, dense trajectories, target velocity/heading, an
initial-transform editor, or browser postprocess parameters. These remain
Python/Viser features.

### Structured output and viewer

The current decoder returns all of the following for every emitted chunk:

- normalized `[T, 330]` ARDY motion features;
- world-space joint positions `[T, J, 3]`;
- local and global joint rotation matrices `[T, J, 3, 3]`;
- root positions and global root headings;
- four predicted foot-contact channels.

The viewer consumes dynamic skeleton names, parents, root index, and contact
metadata instead of assuming one fixed topology. The current compatible pack
describes Core27. Display controls expose the skeleton, root trajectory,
predicted contacts, and joint orientation axes.

Under **View settings**, **Load VRM** accepts a local VRM 0.x or 1.x humanoid
file. [`@pixiv/three-vrm`](https://github.com/pixiv/three-vrm) loads the avatar
only after the user selects it, and the viewer supports show/hide, replacement,
and removal without reloading the motion. Core27 hips translation is scaled to
the avatar, and Core27 joint rotations are retargeted onto the normalized VRM
humanoid. Missing optional VRM bones are skipped.

A VRM is the supported skinned-avatar format; the app has no general
scene-mesh or reference-motion importer.

### VRM rotation-track requirement

VRM bone animation requires either `globalRotations` or `localRotations` in the
motion. The current exported pack supplies both. The manifest should contain:

```json
{
  "schema_version": 2,
  "runtime": { "contract_revision": 3 },
  "graphs": {
    "decoder": {
      "outputs": {
        "localRotations": "local_rotations",
        "globalRotations": "global_rotations"
      }
    }
  }
}
```

The loader accepts only the current gzip archive/schema contract. Older
directory packs and positions-only packs are rejected; regenerate them with
the current exporter.

## Architecture

Inference runs in a dedicated Web Worker so ONNX execution does not block the
main UI. ONNX Runtime Web selects WebGPU first in `auto` mode and creates
separate WebAssembly sessions if WebGPU session creation fails.

| Graph | Main inputs | Outputs |
|---|---|---|
| `text_encoder.onnx` | WordPiece IDs, attention mask, token types | direct 2,048-D root/body condition (two 1,024-D branches) |
| `denoiser.onnx` | text CFG, up to 40 history frames, 40 generation frames, text condition, timestep | clean 148-D hybrid tokens for unconstrained windows |
| `decoder.onnx` | hybrid tokens, valid-token mask, accumulated root translation | normalized motion, joints, local/global rotations, roots/headings, contacts |

The JavaScript runtime supplies a reproducible portable Gaussian random stream
and implements ARDY's ten-step, eta-zero DDIM update. Between windows, it
retains global hybrid tokens, recenters the latest history, and requantizes
the latent body features with the manifest's FSQ constants.

The worker protocol supports replace, append, branch, chunk progress,
continuation restore, rich motion arrays, and capability reporting. It has no
constraint-graph inputs. Typed-array snapshots are transferred to the main
thread so streaming cannot detach state that the worker still needs.

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
  --output artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz
```

The exporter checks all three ONNX files and, unless `--skip-verify` is passed,
compares each graph with its PyTorch source through ONNX Runtime CPU.
`manifest.json` records graph contracts, tensor dimensions, diffusion and
quantization constants, normalization statistics, motion layout, skeleton
metadata, capabilities, file sizes, SHA-256 digests, model compatibility, and
license notices. Before ONNX export, it specializes the denoiser's
non-persistent sinusoidal lookup tables to the ten reachable diffusion
timesteps and the fixed 20-token browser AR window. It then writes a
deterministic POSIX ustar archive through gzip; no unpacked model-pack output is
kept.

Confirm that the result is the current structured-output contract:

```bash
tar -xOzf artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz manifest.json |
jq '{
  schema_version,
  contract_revision: .runtime.contract_revision,
  local_rotations: .graphs.decoder.outputs.localRotations,
  global_rotations: .graphs.decoder.outputs.globalRotations
}'
```

The expected schema/contract revisions are `2` and `3`, and both rotation
output names must be present. The archive contains exactly one manifest, three
ONNX graphs, and two tokenizer files.

The verified export produced in this environment is 718,180,222 bytes
(684.91 MiB, 0.6689 GiB) as `.tar.gz`. Its member payload is 775,577,052 bytes:

| Asset | Bytes |
|---|---:|
| MiniLM condition encoder | 112,430,592 |
| Specialized unconstrained ARDY denoiser | 590,701,706 |
| Structured motion decoder | 71,642,198 |
| Tokenizer files | 712,243 |
| Manifest | 90,313 |

The former four-graph directory occupied 1,488,867,166 bytes in the same
environment. Removing its constraint graph and specializing the remaining
denoiser saves 713,290,114 bytes (680.25 MiB, 0.6643 GiB) before compression;
the final gzip file is 770,686,944 bytes (0.7177 GiB) smaller than that
directory. Small exporter/version differences can change the exact byte count.

## Run the app

Install the pinned browser dependencies and start Vite:

```bash
cd web
npm ci
npm run dev
```

Open the printed localhost URL, choose **Choose model pack**, and select the
`artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz` file. The app
stream-decompresses gzip, validates the POSIX ustar structure and every
manifest-declared file, then creates the three inference sessions. When origin
storage is available, it can retain the original gzip archive in
origin-private storage for later visits.

The desktop UI has an **Input** pane and an **Output** pane:

1. Load the model pack in **Input**, enter a prompt or select one of the
   examples, choose clip duration/seed/backend, and generate.
2. Use the Output pane's **View settings** disclosure for the skeleton,
   contacts, orientation axes, trajectory, and a local VRM avatar.
3. Select **Load VRM** and choose a `.vrm` file. The avatar stays local to the
   current page and can be hidden, replaced, or removed.
4. Use the timeline and playback controls to inspect the generated motion.

There is no right-side Control/Motion-parameters inspector. Kinematic
constraints and detailed planning controls belong to the separate
Python/Viser demo.

The header reports the backend that actually loaded:

- **WebGPU** is preferred and requires a supporting browser plus HTTPS or
  localhost.
- **WebAssembly** is the compatibility fallback. It can be substantially
  slower and is not a real-time guarantee for this model size.

The preview supports drag/swipe orbit, wheel/pinch zoom, and keyboard
operation. The camera follows the generated root position while preserving the
chosen orbit and distance. With the preview focused, W/A/S/D translates the
camera, Space toggles playback, Left/Right seeks, Shift+Arrow orbits,
Plus/Minus zooms, and Home resets the camera around the current pose. Generate
uses Command+Enter on Apple platforms and Control+Enter elsewhere.

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

## Session codec compatibility

The repository retains versioned JSON and binary session codecs for validation
and library-level compatibility tests. The current browser interface does not
expose session import/export, motion export, or reference-motion import entry
points.

## Privacy and hosting

Prompts, seeds, model-pack files, selected VRM avatars, generation state, and
generated motion remain in the browser. Model and avatar selection read local
files; inference does not upload them. VRM object URLs are revoked after
loading. The app has no inference service dependency.

The selected `.tar.gz` model pack may be retained as one file in
origin-private storage when the browser permits it. A selected VRM avatar is
page-local; select it again after reloading the page.
If the app finds the former unpacked directory-pack cache, it removes that
unsupported cache before asking for the current archive.

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
| Normalized motion | `1.51e-3` |
| Posed joints | `1.23e-4 m` |
| Local rotations | `9.36e-4` |
| Global rotations | `1.09e-3` |
| Root positions / headings / contacts | `0` |

Browser runtime and memory measurements are device-, browser-, driver-, and
execution-provider-specific. The current pack has three graphs; measurements
from the former four-graph directory pack or positions-only packs do not
describe this contract.

An opt-in real-pack Playwright run can force either provider:

```bash
cd web

ARDY_BROWSER_MODEL_PACK=../artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz \
ARDY_BROWSER_BACKEND=wasm \
npm run test:e2e -- e2e/real-model.spec.ts

ARDY_BROWSER_MODEL_PACK=../artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz \
ARDY_BROWSER_BACKEND=webgpu \
npm run test:e2e -- e2e/real-model.spec.ts
```

Set `ARDY_BROWSER_REDUCED_MOTION=1` to exercise paused initial playback.

## Current limitations

- The trained MiniLM condition heads are checkpoint-specific. This pack is not
  interchangeable with Core8, G1, SOMA, or another 2,048-D model.
- Prompt support is limited to well-formed, typo-free English motion
  descriptions. The form accepts other text but does not promise useful output.
- Branching crops to the preceding complete four-frame token; a partial token
  cannot be continued exactly.
- Each generation call is limited to 40–200 frames. Longer sessions are built
  through append/streaming operations.
- The app has no detailed motion-parameter editor, kinematic-constraint
  authoring, waypoint/target-velocity planning, JavaScript motion-correction
  pass, or native Viser rotation-space IK.
- VRM retargeting is designed for the current Core27 skeleton names and a VRM
  humanoid rig. Missing optional VRM bones are skipped; expressions and
  non-humanoid animation are not driven.
- VRM animation requires local or global rotation tracks. Positions-only
  motion moves the hips but leaves the avatar in its rest/T pose; there is no
  position-derived rotation fallback.
- The pack supplies Core27 skeleton metadata and no bundled character mesh.
  VRM is the only supported local character format; the browser has no general
  scene-mesh importer.
- A loaded VRM is page-local and must be selected again after a reload.
- WASM is a fallback, not a performance promise. Three large graph sessions can
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

No VRM asset is bundled with the repository or static build. Loading one
locally does not grant redistribution or usage rights for that avatar; follow
the permissions and attribution metadata embedded by its author.

The manifest's SHA-256 hashes detect corruption and file substitution relative
to that manifest; they are not a digital signature. Import packs only from a
source you trust.

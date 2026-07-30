# Third-party models and data

This file complements [LICENSE](LICENSE) and [ATTRIBUTIONS.MD](ATTRIBUTIONS.MD).
Apache-2.0 applies to this repository's source code; it does not grant rights
to separately obtained models, checkpoints, datasets, or locally generated
artifacts.

## Distribution boundary

This source repository does not distribute ARDY checkpoints, LLM2Vec adapters,
Meta Llama 3 weights, pretrained or fine-tuned MiniLM weights, teacher
embeddings or caches, prompt manifests, NVIDIA SEED Timeline Annotations,
BONES-SEED runtime data or metadata, prompt-level evaluation reports, or
generated motion samples. Users must obtain each external resource from its
official provider and accept its separate terms.

The local `artifacts/`, `datasets/`, `checkpoints/`, and `outputs/` directories
are ignored by Git to keep external inputs and generated outputs outside the
source distribution.

## Upstream ARDY

- Source: [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)
- Source license: [Apache-2.0](LICENSE)
- Imported revision: `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- Modifications: summarized in [NOTICE](NOTICE) and marked in each changed
  upstream file.

Released ARDY checkpoints are obtained separately. The Core40 checkpoint
required by the documented MiniLM workflow is governed by its
[checkpoint license](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40/blob/main/LICENSE)
and the applicable
[NVIDIA Open Model terms](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
No NVIDIA checkpoint is included here.

## LLM2Vec and Meta Llama 3

The teacher path is Built with Meta Llama 3. Meta Llama 3 is governed by the
[Meta Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE)
and its Acceptable Use Policy. The foundation-model weights are not included.

The repository retains patched LLM2Vec source from McGill NLP. Its MIT notice
is reproduced in [ATTRIBUTIONS.MD](ATTRIBUTIONS.MD). The separately obtained
LLM2Vec adapters identify their own terms; those terms do not replace the
Meta Llama 3 terms for the foundation model.

## all-MiniLM-L6-v2

The optional student-training workflow downloads
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
separately. Its
[Apache-2.0 license and sentence-transformers copyright notice](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/LICENSE)
apply to the base model. This repository contains only code for producing and
loading a local fine-tuned artifact; it does not include student weights.

The source-code license in this repository does not, by itself, authorize
redistribution of a locally trained student. Before distributing one, review
the terms for every training-data and teacher component and prepare a separate
model card and license package.

## BONES-SEED

The optional Python/Viser kinematic-constraint demo can read separately
obtained BONES-SEED motion files and metadata at runtime. BONES-SEED is gated
and restricted. Users must independently qualify for the license or obtain an
appropriate separate license under the
[BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).

## NVIDIA SEED Timeline Annotations

The MiniLM training and evaluation corpus consists exclusively of English
motion descriptions from NVIDIA Corporation's
[Timeline Annotations for BONES-SEED Humanoid Motion Dataset](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations)
(`nvidia/SEED-Timeline-Annotations`). The dataset is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The training and evaluation recipe pins dataset revision
`b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f`. At that revision,
`timelines.jsonl` has SHA-256
`379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8`.

Changes from the source annotations: corpus preparation selects English
overview and event descriptions, normalizes text, drops descriptions longer
than 512 characters, deduplicates prompts, and assigns motion groups to
disjoint train, validation, and test splits. Corpus preparation rejects
malformed source records. This repository includes only aggregate measurements
derived from that corpus; the source annotations and detailed derived prompt
manifests remain outside the source distribution.

## Browser runtime dependencies

The optional browser demo uses
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime) under the MIT
license,
[Hugging Face Tokenizers.js](https://github.com/huggingface/tokenizers.js)
under Apache-2.0, and [three.js](https://github.com/mrdoob/three.js) under the
MIT license. Development and testing use Vite and Vitest under the MIT license
and TypeScript and Playwright under Apache-2.0. The exact versions are pinned in
`web/package-lock.json`. The production build copies source, ONNX Runtime Web,
Tokenizers.js, and three.js notices into `web/dist/notices/`; Vite also emits
the bundled dependency notices in `web/dist/third-party-licenses.md`.

The browser exporter creates a model pack from separately obtained ARDY and
MiniLM artifacts. That generated pack remains outside Git and does not acquire
the source repository's Apache-2.0 license. Anyone distributing a pack must
review the ARDY checkpoint, MiniLM artifact, training-data, and teacher-model
terms and supply the required model notices independently.

## Unitree assets

The upstream repository includes Unitree-derived G1 assets. The retained BSD
3-Clause notice and non-endorsement condition are reproduced in
[ATTRIBUTIONS.MD](ATTRIBUTIONS.MD).

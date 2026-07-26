# Distilled MiniLM text encoder

This experimental path replaces ARDY's production LLM2Vec prompt encoder with
a specialized encoder based on
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Its supported input domain is well-formed, typo-free English motion prompts.
No robustness claim is made for misspellings, other languages, or general
natural-language tasks.

This repository distributes source code and aggregate documentation only.
Model weights, teacher caches, prompt manifests, dataset records, generated
motions, and prompt-level reports are written under Git-ignored local
directories and are not part of the source distribution. Review
[the third-party model and data notices](../THIRD_PARTY_MODELS_AND_DATA.md)
before obtaining inputs or redistributing a locally produced artifact.

Training data includes [Motion Data by Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the
[BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).

## Design

The released ARDY text path first produces one 4096-dimensional LLM2Vec token.
The two denoiser stages then apply independent, checkpoint-trained projections:

```text
LLM2Vec(prompt) -> e [4096]
root condition  = W_root e + b_root [1024]
body condition  = W_body e + b_body [1024]
```

[`MotionConditionStudent`](../ardy/model/minilm_encoder.py) fine-tunes MiniLM,
pools its contextual token outputs, passes the pooled vector through a shared
adapter, and predicts the two **bias-free** values directly:

```text
MiniLM(prompt) -> shared adapter
                -> [W_root e, W_body e] [2048]
```

The artifact supports standard masked mean pooling and the
`mean_cls_max_std` rich pooling used by the training command below.
[`MiniLMArdyEncoder`](../ardy/model/minilm_encoder.py) exposes ARDY's existing
`(tensor, lengths)` contract and returns `[B, 1, 2048]` for a list of prompts.

[`DualConditionTextProjection`](../ardy/model/backbone.py) keeps the original
checkpoint parameter names and handles both paths:

- a 4096-dimensional input executes the original `Linear(4096, 1024)`;
- a 2048-dimensional input selects the root or body half and adds that
  branch's original checkpoint bias.

This preserves existing checkpoint loading and lets the original and distilled
encoders use the same denoiser. The embedding cache key includes the MiniLM
artifact fingerprint, so embeddings from incompatible artifacts or the legacy
encoder are not reused accidentally.

### Why the student target excludes bias

ARDY classifier-free guidance constructs the unconditional text condition by
zeroing the encoder output. In the original path, the denoiser still computes
`W * 0 + b = b`. The student therefore learns `[W_root e, W_body e]`, not
`[W_root e + b_root, W_body e + b_body]`. The denoiser adds the original bias
for both conditional and zeroed unconditional inputs. Training on biased
targets would add the bias twice for conditional inference.

### Checkpoint scope

The root/body projection matrices are part of each ARDY checkpoint. The
artifact produced by this guide is consequently specific to
`ARDY-Core-RP-20FPS-Horizon40` (Core40); it is not interchangeable with
Core8, G1, or another checkpoint merely because their branch width is also
1024. The artifact records `compatible_ardy_models`, and normal
[`load_model`](../ardy/model/load_model.py) use checks this value before
attaching the encoder.

Training provenance in the artifact and teacher-cache metadata includes the
source checkpoint path/hash and teacher identity. A local artifact is not
covered merely by this repository's Apache-2.0 source license. Do not
redistribute it without separately reviewing the NVIDIA Open Model, Meta
Llama, LLM2Vec, source-corpus, and MiniLM terms and preparing an appropriate
model card and license package.

## Reproducible `uv` pipeline

The commands below use the repository root as the working directory, CUDA
with bfloat16 where supported, and these local inputs:

```text
datasets/bones-seed/metadata/seed_metadata_v004.csv
checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors
```

Create the environment and authenticate with Hugging Face before caching the
gated LLM2Vec/Meta Llama teacher:

```bash
uv sync --python 3.11 --no-install-project
uv run hf auth login
```

### 1. Prepare prompts

[`prepare_prompts.py`](../scripts/minilm/prepare_prompts.py) normalizes and
globally deduplicates the seven BONES-SEED description columns. It assigns
whole content families to deterministic train/validation/test splits, so
mirrored or related descriptions do not leak across splits.

```bash
uv run python scripts/minilm/prepare_prompts.py \
  --input datasets/bones-seed/metadata/seed_metadata_v004.csv \
  --output artifacts/data/prompts-core40-16384.jsonl \
  --sample-size 16384 \
  --split-ratios 0.8 0.1 0.1 \
  --seed 20260726
```

### 2. Cache teacher targets

[`cache_teacher.py`](../scripts/minilm/cache_teacher.py) runs the production
LLM2Vec wrapper with its production batch size of one. Each atomic shard stores
the raw float32 `[N, 4096]` teacher embeddings and float32 `[N, 2048]`
checkpoint-projected, bias-free targets. Re-running the same command resumes a
compatible partial cache after validating its identity, shape, finiteness, and
shard hashes.

```bash
uv run python scripts/minilm/cache_teacher.py \
  --input artifacts/data/prompts-core40-16384.jsonl \
  --output-dir artifacts/teacher-core40-16384 \
  --checkpoint checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors \
  --shard-size 256 \
  --device cuda
```

Do not combine shards created with another checkpoint, teacher revision, or
prompt manifest. Use a new output directory for a different target.

### 3. Train the student

[`train.py`](../scripts/minilm/train.py) jointly optimizes normalized
regression, root/body cosine fidelity, and within-batch relational fidelity.
It selects the best epoch by validation root/body cosine and writes a
self-contained tokenizer, MiniLM backbone, condition heads, compatibility
metadata, and training report.

```bash
uv run python scripts/minilm/train.py \
  --cache-dir artifacts/teacher-core40-16384 \
  --output-dir artifacts/minilm-ardy-core40 \
  --ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --pooling-mode mean_cls_max_std \
  --adapter-dim 1536 \
  --epochs 50 \
  --head-warmup-epochs 0 \
  --batch-size 128 \
  --head-lr 3e-3 \
  --backbone-lr 1e-4 \
  --cosine-weight 0.5 \
  --relational-weight 0.05 \
  --seed 20260726 \
  --device cuda
```

The artifact layout is:

```text
artifacts/minilm-ardy-core40/
  ardy_minilm_config.json
  condition_heads.safetensors
  training_report.json
  backbone/
```

New artifacts use format v2. The config keeps training provenance under
`metadata` and contains a deterministic size/SHA-256 manifest for the
condition heads and every saved backbone/tokenizer file; the artifact
fingerprint covers that manifest and the canonical config. Loading verifies
the complete payload before model construction. Existing format-v1 artifacts
remain loadable through their legacy heads/config fingerprint.

### 4. Evaluate condition fidelity

Condition evaluation uses the held-out prompt split and reports root, body,
and overall cosine mean/p5, RMSE, normalized RMSE, and MAE:

```bash
uv run python scripts/minilm/evaluate_conditions.py \
  --teacher-cache artifacts/teacher-core40-16384 \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --split test \
  --batch-size 64 \
  --device cuda \
  --dtype bfloat16 \
  --output artifacts/evaluation/conditions.json
```

### 5. Evaluate paired motion fidelity

[`evaluate_motion.py`](../scripts/minilm/evaluate_motion.py) feeds cached
LLM2Vec embeddings and live MiniLM conditions into the same Core40 denoiser
with identical prompts, seeds, diffusion settings, and CFG weights:

```bash
uv run python scripts/minilm/evaluate_motion.py \
  --cache-dir artifacts/teacher-core40-16384 \
  --student-path artifacts/minilm-ardy-core40 \
  --checkpoints-dir checkpoints \
  --model core \
  --prompt-manifest artifacts/data/prompts-core40-16384.jsonl \
  --sources content_natural_desc_1 content_natural_desc_2 \
            content_natural_desc_3 content_natural_desc_4 \
  --num-prompts 64 \
  --seeds 0 1 2 \
  --duration 4 \
  --device cuda \
  --output artifacts/evaluation/motion_metrics_clean_english.json \
  --sample-dir outputs/minilm-motion-comparison-clean
```

The report includes root ADE/FDE, global and root-aligned MPJPE, joint velocity
error, motion cosine, heading error, foot-contact agreement, and errors
normalized by the original encoder's different-seed diversity. A few paired
`.npz` samples are also written for visual inspection. The manifest and source
filter bind this run to the four natural-description columns; omit
`--prompt-manifest` and `--sources` to evaluate every held-out description
type in cache order.

### 6. Benchmark memory and latency

Run the production wrappers in separate fresh processes. Both benchmarks use
external batch size one, synchronized CUDA timing, and the same prompt:

```bash
uv run python scripts/minilm/benchmark_encoders.py teacher \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 30 \
  --output artifacts/evaluation/encoder-teacher.json

uv run python scripts/minilm/benchmark_encoders.py student \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 30 \
  --output artifacts/evaluation/encoder-student.json
```

For the fairest end-to-end condition comparison, include the same Core40 model
and its real root/body projections on both sides:

```bash
uv run python scripts/minilm/benchmark_encoders.py full-teacher \
  --model core \
  --checkpoints-dir checkpoints \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 100 \
  --output artifacts/evaluation/full-teacher.json

uv run python scripts/minilm/benchmark_encoders.py full-student \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --model core \
  --checkpoints-dir checkpoints \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 100 \
  --output artifacts/evaluation/full-student.json
```

The JSON reports load time, first and warm encoding latency, parameter count,
Linux RSS, and PyTorch CUDA allocated/reserved memory. PyTorch allocator
figures exclude CUDA driver/context memory; use the RSS and CUDA fields
together, especially on unified-memory systems.

## Using the artifact

Force local mode so an already-running text-encoder API cannot silently take
precedence:

```bash
TEXT_ENCODER_MODE=local \
TEXT_ENCODER=minilm \
MINILM_TEXT_ENCODER_PATH=artifacts/minilm-ardy-core40 \
CHECKPOINTS_DIR=checkpoints \
uv run python scripts/generate.py \
  "A person turns left and walks forward." \
  --model core \
  --duration 5 \
  --seed 0 \
  --output minilm-core40
```

The same environment variables can be used with
`uv run python scripts/run_demo.py`. Keep `--model core`/Core40 with this
artifact.

## Result artifacts and evaluation limits

The completed experiment, exact environment, aggregate tables, and
interpretation are recorded in
[`minilm_results.md`](minilm_results.md). A sanitized, prompt-free
machine-readable copy of the aggregate values is available at
[`reports/minilm_core40_summary.json`](../reports/minilm_core40_summary.json).

The commands above generate detailed reports locally under `artifacts/`.
That directory is ignored by Git because those outputs contain local paths,
model provenance, and—in the case of motion evaluation—licensed prompt text.
Do not force-add them to a public repository.

The public ARDY repository does not include the proprietary Rigplay evaluation
split or the paper's TMR evaluator. There is therefore no public,
paper-comparable FID or R-precision calculation in this workflow. The
condition metrics measure agreement with the original encoder's projected
conditions on held-out BONES-SEED text. The motion metrics measure paired
fidelity to the original encoder under matched stochastic generation and
normalize several errors by teacher seed diversity. They are useful regression
and replacement-fidelity measures, but they are not ground-truth semantic
motion accuracy or substitutes for the unpublished Rigplay/TMR evaluation.

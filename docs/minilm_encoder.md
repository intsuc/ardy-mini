# Distilled MiniLM text encoder

This encoder path replaces ARDY's production LLM2Vec prompt encoder with a
specialized encoder based on
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Its supported input domain is well-formed, typo-free English motion prompts.
No robustness claim is made for misspellings, other languages, or general
natural-language tasks.

This repository distributes source code and aggregate documentation only.
Model weights, teacher caches, prompt manifests, source dataset records,
generated motions, and prompt-level reports are written under Git-ignored
local directories and are not part of the source distribution. Review
[the third-party model and data notices](../THIRD_PARTY_MODELS_AND_DATA.md)
before obtaining inputs or redistributing a locally produced artifact.
Measured aggregate outcomes are documented separately in
[`minilm_results.md`](minilm_results.md).

The prompt corpus comes only from NVIDIA's
[`nvidia/SEED-Timeline-Annotations`](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations)
dataset at revision
`b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f`. NVIDIA publishes the
annotations under CC BY 4.0. The pinned `timelines.jsonl` is 80,373,523 bytes
with SHA-256
`379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8`.

## Design

The original ARDY text path first produces one 4096-dimensional LLM2Vec token.
The two denoiser stages then apply independent, checkpoint-trained
projections:

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
`mean_cls_max_std` rich pooling used below.
[`MiniLMArdyEncoder`](../ardy/model/minilm_encoder.py) exposes ARDY's existing
`(tensor, lengths)` contract and returns `[B, 1, 2048]` for a list of prompts.

[`DualConditionTextProjection`](../ardy/model/backbone.py) keeps the original
checkpoint parameter names and handles both encoder paths:

- a 4096-dimensional input executes the original `Linear(4096, 1024)`;
- a 2048-dimensional input selects the root or body half and adds that
  branch's original checkpoint bias.

This lets both encoders use the same denoiser without changing checkpoint
loading. The embedding-cache key includes the MiniLM artifact fingerprint, so
embeddings from a different artifact or from LLM2Vec cannot be reused
accidentally.

### Why the student target excludes bias

ARDY classifier-free guidance constructs the unconditional text condition by
zeroing the encoder output. In the original path, the denoiser still computes
`W * 0 + b = b`. The student therefore learns `[W_root e, W_body e]`, not
`[W_root e + b_root, W_body e + b_body]`. The denoiser adds the original bias
for conditional and zeroed unconditional inputs. Training on biased targets
would add the bias twice during conditional inference.

### Checkpoint scope

The root/body projection matrices belong to each ARDY checkpoint. The
artifact produced by this guide is therefore specific to
`ARDY-Core-RP-20FPS-Horizon40` (Core40); it is not interchangeable with Core8,
G1, or another checkpoint merely because their branch width is also 1024.
The artifact records `compatible_ardy_models`, and normal
[`load_model`](../ardy/model/load_model.py) use verifies that value before
attaching the encoder.

The local teacher cache retains the source checkpoint path/hash and teacher
identity. The portable student artifact records checkpoint and prompt-manifest
filenames/hashes, not machine-local absolute paths. A locally trained artifact
is not covered merely by this repository's Apache-2.0 source license. Review
the NVIDIA Open Model, Meta Llama, LLM2Vec, MiniLM, and CC BY 4.0 dataset terms
before redistribution, and publish the required attribution and model
documentation.

## Reproducible `uv` pipeline

Run the commands from the repository root. Training and performance
benchmarks use CUDA BF16 where supported; saved-artifact selection and final
quality evaluation use FP32. The pipeline requires the Core40 denoiser
checkpoint:

```text
checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors
```

Create the environment and authenticate with Hugging Face before caching the
gated LLM2Vec/Meta Llama teacher:

```bash
uv sync --python 3.11 --no-install-project
uv run hf auth login
```

### 1. Prepare the NVIDIA Timeline prompts

[`prepare_prompts.py`](../scripts/minilm/prepare_prompts.py) downloads the
pinned dataset revision, verifies both its byte size and SHA-256, and reads
`overview_description` plus every `events[].description`. Supplying a local
file with `--input PATH` does not bypass those identity checks.

```bash
uv run python scripts/minilm/prepare_prompts.py \
  --output artifacts/data/prompts-core40-timeline.jsonl \
  --split-ratios 0.8 0.1 0.1 \
  --seed 20260726
```

Preparation applies Unicode NFKC normalization, trims and collapses
whitespace, and deduplicates prompts globally with a case- and
punctuation-insensitive alphanumeric-token key. This prevents punctuation and
hyphenation variants of the same sentence from crossing splits. When the same
prompt occurs in both fields, `overview_description` takes priority.
Descriptions longer than the 512-character limit are filtered.

To prevent related recordings from crossing splits, filenames are normalized
and case-folded; actor, mirror, and terminal three-digit non-angle take
suffixes are removed;
and families connected through `propagated_from_filename` are unioned before
the deterministic group hash assigns a split. The prepared manifest SHA-256
is
`c80b0656dd04c28da1d665b4b9a9422f975be5c72b0fe9011b976a3672bb1eac`.
The command also writes
`artifacts/data/prompts-core40-timeline.metadata.json`, a canonical
`ardy-minilm-prompt-provenance` sidecar bound to that manifest hash. It records
the fixed dataset identity and license, extraction policy, normalization,
deduplication, grouping, seed, ratios, and counts. The validated corpus and
split counts are reported in [`minilm_results.md`](minilm_results.md).

### 2. Cache teacher targets

[`cache_teacher.py`](../scripts/minilm/cache_teacher.py) requires and validates
the provenance sidecar before loading the teacher. It runs the production
LLM2Vec wrapper with its production internal batch size of one. Each atomic
shard stores raw float32 `[N, 4096]` teacher embeddings and float32
`[N, 2048]` checkpoint-projected, bias-free targets:

```bash
uv run python scripts/minilm/cache_teacher.py \
  --input artifacts/data/prompts-core40-timeline.jsonl \
  --output-dir artifacts/teacher-core40-timeline \
  --checkpoint checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors \
  --shard-size 256 \
  --device cuda
```

Teacher-cache format v3 embeds the complete validated corpus provenance and
the sidecar hash in `metadata.json`, together with prompt/checkpoint hashes,
teacher identity and revisions, device/runtime details, dimensions, dtypes,
split counts, elapsed time, and the ordered shard manifest. A repeated
compatible command validates finished shards and resumes atomically. A
different corpus, checkpoint, teacher, device provenance, or target definition
must use a fresh cache directory.

### 3. Train fresh 50- and 100-epoch candidates

[`train.py`](../scripts/minilm/train.py) jointly optimizes normalized
regression, root/body cosine fidelity, and within-batch relational fidelity.
It keeps the epoch with the best mean validation root/body cosine and writes a
self-contained tokenizer, MiniLM backbone, condition heads, compatibility
metadata, and `training_report.json`.

Train both candidates from fresh initialization against the one Timeline
cache. Do not supply a separate evaluation cache:

```bash
uv run python scripts/minilm/train.py \
  --cache-dir artifacts/teacher-core40-timeline \
  --output-dir artifacts/minilm-ardy-core40-timeline-50e \
  --ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --pooling-mode mean_cls_max_std \
  --adapter-dim 1536 \
  --epochs 50 \
  --lr-schedule-epochs 100 \
  --head-warmup-epochs 0 \
  --batch-size 128 \
  --head-lr 3e-3 \
  --backbone-lr 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --cosine-weight 0.5 \
  --relational-weight 0.05 \
  --train-max-length 128 \
  --runtime-max-length 128 \
  --num-workers 0 \
  --seed 20260726 \
  --device cuda

uv run python scripts/minilm/train.py \
  --cache-dir artifacts/teacher-core40-timeline \
  --output-dir artifacts/minilm-ardy-core40-timeline-100e \
  --ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --pooling-mode mean_cls_max_std \
  --adapter-dim 1536 \
  --epochs 100 \
  --lr-schedule-epochs 100 \
  --head-warmup-epochs 0 \
  --batch-size 128 \
  --head-lr 3e-3 \
  --backbone-lr 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --cosine-weight 0.5 \
  --relational-weight 0.05 \
  --train-max-length 128 \
  --runtime-max-length 128 \
  --num-workers 0 \
  --seed 20260726 \
  --device cuda
```

Both commands use the same deterministic 100-epoch LR schedule, so the
50-epoch run is a prefix-matched control rather than a separately compressed
cosine schedule. Within each run, training uses BF16 autocast validation to
retain its best epoch. That choice is recorded as the within-run checkpoint
selection; it is not the final comparison between the two saved artifacts.

Evaluate both saved artifacts over the complete validation split in FP32:

```bash
uv run python scripts/minilm/evaluate_conditions.py \
  --teacher-cache artifacts/teacher-core40-timeline \
  --student-path artifacts/minilm-ardy-core40-timeline-50e \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --split val \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --output artifacts/evaluation/conditions-timeline-val-fp32-50e.json

uv run python scripts/minilm/evaluate_conditions.py \
  --teacher-cache artifacts/teacher-core40-timeline \
  --student-path artifacts/minilm-ardy-core40-timeline-100e \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --split val \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --output artifacts/evaluation/conditions-timeline-val-fp32-100e.json
```

Select only between these two saved artifacts using the higher FP32 mean of
`root.cosine.mean` and `body.cosine.mean`; an exact tie selects 50 epochs.
Do not inspect test metrics during this choice. This no-retraining protocol
cannot re-rank discarded epoch states in FP32: each saved artifact remains the
epoch its training run selected under BF16 validation.

After selecting the saved winner, copy it to the canonical runtime location
on a clean run:

```bash
test ! -e artifacts/minilm-ardy-core40
cp -a artifacts/minilm-ardy-core40-timeline-100e \
  artifacts/minilm-ardy-core40
```

The selected candidate identity and validation scores are recorded in
[`minilm_results.md`](minilm_results.md).

The installed artifact layout is:

```text
artifacts/minilm-ardy-core40/
  ardy_minilm_config.json
  condition_heads.safetensors
  training_report.json
  backbone/
```

The artifact config stores training and teacher/corpus provenance plus a
deterministic size/SHA-256 manifest for the condition heads and every saved
backbone/tokenizer file. Loading validates the complete payload before model
construction.

### 4. Evaluate condition fidelity

After validation-only selection, evaluate the installed winner on the held-out
Timeline test split:

```bash
uv run python scripts/minilm/evaluate_conditions.py \
  --teacher-cache artifacts/teacher-core40-timeline \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --split test \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --output artifacts/evaluation/conditions-timeline-test.json
```

The report includes root, body, and overall cosine mean/p5, RMSE, normalized
RMSE, and MAE.

### 5. Evaluate paired motion fidelity

[`evaluate_motion.py`](../scripts/minilm/evaluate_motion.py) feeds cached
LLM2Vec embeddings and live MiniLM conditions into the same Core40 denoiser
with identical prompts, seeds, diffusion settings, and CFG weights:

```bash
uv run python scripts/minilm/evaluate_motion.py \
  --cache-dir artifacts/teacher-core40-timeline \
  --student-path artifacts/minilm-ardy-core40 \
  --checkpoints-dir checkpoints \
  --model core \
  --prompt-manifest artifacts/data/prompts-core40-timeline.jsonl \
  --sources overview_description events.description \
  --num-prompts 64 \
  --seeds 0 1 2 \
  --duration 4 \
  --diffusion-steps 10 \
  --cfg-weight 2 2 \
  --student-dtype float32 \
  --repeatability-check \
  --device cuda \
  --output artifacts/evaluation/motion-metrics-timeline-test.json \
  --sample-dir outputs/minilm-motion-comparison-timeline
```

The report includes root ADE/FDE, global and root-aligned MPJPE, joint velocity
error, motion cosine, heading error, foot-contact agreement, and errors
normalized by the original encoder's different-seed diversity. A few paired
`.npz` samples are written for visual inspection. The prompt manifest and
source filter bind selection to the two Timeline fields. The final protocol
uses 64 evenly spaced test prompts in manifest/cache order, seeds 0/1/2,
4 seconds at 20 FPS, 10 diffusion steps, CFG weights `[2, 2]`, and the
student text encoder in FP32. The evaluator configures deterministic PyTorch,
cuDNN, cuBLAS, and TF32 settings before model construction, records them, and
requires the first teacher rollout to be bitwise repeatable. It also records
the selected prompt-list digest without putting prompt text in the public
aggregate.

### 6. Benchmark memory and latency

Run the production wrappers in separate fresh processes. Both benchmarks use
external batch size one, synchronized CUDA timing, and the same prompt:

```bash
uv run python scripts/minilm/benchmark_encoders.py teacher \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 100 \
  --output artifacts/evaluation/encoder-teacher.json

uv run python scripts/minilm/benchmark_encoders.py student \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --device cuda \
  --dtype bfloat16 \
  --warm-runs 100 \
  --output artifacts/evaluation/encoder-student.json
```

For the end-to-end condition comparison, include the same Core40 model and
its real root/body projections on both sides:

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

Run the commands sequentially on an otherwise idle machine, letting each
process exit before starting the next. The teacher benchmark requires the
three exact, revision-pinned
Hugging Face snapshots to already be present in the local cache; the teacher
cache step above satisfies that requirement. Network access is disabled for
benchmark snapshot resolution.

`timing_seconds.load` includes path-specific Python imports, local
pinned-snapshot cache resolution for the teacher (or local artifact validation
for the student), model construction, and device transfer. Common interpreter
and benchmark-module startup happens before the load timer. Full-stack load
timing also includes the same local Core40 construction. CUDA context
initialization occurs during the synchronization immediately before the stage
timer and is excluded. The first measured encode is reported separately and
serves as the warmup before all `--warm-runs` calls are individually measured.
Load latency remains sensitive to the operating-system page cache, so compare
it only between runs on the same idle machine and keep the command order in
the published report.

The JSON reports load time, first and warm encoding latency, parameter count,
Linux RSS, and PyTorch CUDA allocated/reserved memory. PyTorch allocator
figures are absolute allocator values and exclude CUDA driver/context and
non-PyTorch allocations. `MemAvailable` is system-wide and can move because
of other processes or page cache. On unified-memory systems, RSS/MemAvailable
and CUDA allocator figures can describe overlapping physical memory, so do
not add them. Use resident RSS for the process footprint, lifetime
peak RSS for the process peak, and CUDA allocated/reserved values for the
PyTorch device allocator view.

### 7. Build the public aggregate

Keep every raw report above under the ignored `artifacts/` directory. After
both candidates have completed and the validation winner alone has been
evaluated, build the Git-tracked aggregate:

```bash
uv run python scripts/minilm/build_public_summary.py \
  --training-50 artifacts/minilm-ardy-core40-timeline-50e/training_report.json \
  --training-100 artifacts/minilm-ardy-core40-timeline-100e/training_report.json \
  --validation-fp32-50 artifacts/evaluation/conditions-timeline-val-fp32-50e.json \
  --validation-fp32-100 artifacts/evaluation/conditions-timeline-val-fp32-100e.json \
  --condition-test artifacts/evaluation/conditions-timeline-test.json \
  --motion-test artifacts/evaluation/motion-metrics-timeline-test.json \
  --benchmark-encoder-teacher artifacts/evaluation/encoder-teacher.json \
  --benchmark-encoder-student artifacts/evaluation/encoder-student.json \
  --benchmark-full-teacher artifacts/evaluation/full-teacher.json \
  --benchmark-full-student artifacts/evaluation/full-student.json \
  --output reports/minilm_core40_summary.json
```

[`build_public_summary.py`](../scripts/minilm/build_public_summary.py) verifies
that the complete 50-epoch history exactly matches the first 50 epochs of the
100-epoch history after excluding per-epoch `seconds`. It labels the training
reports' within-run checkpoint choices as BF16, then selects between the two
saved artifacts solely from their complete-split FP32 validation reports. An
exact tie selects the shorter run. It requires both validation reports and the
winner-only condition, motion, and student benchmark reports to identify the
same teacher cache, candidate artifact, and Core40 checkpoint. Test metrics are
never used for selection.

The output is built from an explicit allowlist. It records raw-report hashes,
Timeline provenance, validation selection, winner-only aggregate test metrics,
and benchmark summaries, but not prompt bodies, per-case dataset record
identifiers, local absolute paths, commands, or benchmark timing sample
arrays. The command fails closed if such private fields reach the public
schema and writes the aggregate only after all required raw reports exist.

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

The same environment variables work with
`uv run python scripts/run_demo.py`. Keep `--model core`/Core40 with this
artifact.

## Evaluation limits

Detailed local reports under `artifacts/` contain local paths, teacher/model
provenance, and prompt text. The directory is ignored by Git; do not
force-add these files to a public repository.

The public ARDY repository does not include the proprietary Rigplay
evaluation split or the paper's TMR evaluator, so this workflow cannot produce
paper-comparable FID or R-precision. Condition metrics measure agreement with
the original encoder's projected conditions on held-out NVIDIA Timeline text.
Motion metrics measure paired replacement fidelity under matched stochastic
generation and normalize several errors by teacher seed diversity. They are
useful regression measures, but they are neither ground-truth semantic motion
accuracy nor substitutes for the unavailable Rigplay/TMR evaluation. The
published FP32 comparison covers the two saved artifacts only; it does not
retroactively compare every discarded training epoch in FP32. BF16 benchmarks
measure performance and are kept separate from FP32 quality selection and
test evaluation.

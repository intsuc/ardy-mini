# MiniLM Core40 results

This page summarizes the aggregate in
[`reports/minilm_core40_summary.json`](../reports/minilm_core40_summary.json).
That JSON is the authoritative numeric source. It contains aggregate metrics
and reproducibility identities, but no model weights, dataset records, prompt
text, record identifiers, or local absolute paths.

Review the
[third-party model and data notices](../THIRD_PARTY_MODELS_AND_DATA.md) before
reproducing the experiment or distributing a trained artifact.

## Experiment identity

- ARDY model: `ARDY-Core-RP-20FPS-Horizon40`
- ARDY checkpoint SHA-256:
  `1019d0bf269cf8d1b3e3e9b4a384a58c112672959b071279ddb65814d77660cd`
- student base: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- selected artifact: 100-epoch candidate
- selected artifact fingerprint:
  `605639d3e0189f10dd3487c2bcf613825699475573eb34f57ac71113902c2f83`
- selected artifact payload: 113,637,018 bytes
- training and benchmark GPU: NVIDIA GB10
- training seed: `20260726`

The teacher consists of the following revision-pinned models:

- `meta-llama/Meta-Llama-3-8B-Instruct` at
  `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` at
  `31474e395ada192e8ed1586db6be79fb3b70c9c0`
- `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` at
  `baa8ebf04a1c2500e61288e7dad65e8ae42601a7`

Teacher embeddings, bias-free root/body projection targets, and projection
weights were cached in float32. The teacher model used BF16.

## Dataset and split

Training and evaluation prompts come from
[`nvidia/SEED-Timeline-Annotations`](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations),
revision `b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f`. The pinned
`timelines.jsonl` is 80,373,523 bytes with SHA-256
`379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8`
and is identified as NVIDIA data under CC BY 4.0.

Prompts are extracted from `overview_description` and
`events.description`. Preparation applies Unicode NFKC normalization,
whitespace normalization, global case- and punctuation-insensitive
deduplication, and recording-family grouping before deterministic splitting.
Two descriptions over the 512-character limit are excluded.

| Item | Count |
| --- | ---: |
| Timeline rows | 142,220 |
| Unique prompts | 64,287 |
| Recording groups | 3,702 |
| `overview_description` prompts | 17,749 |
| `events.description` prompts | 46,538 |
| Train prompts | 51,482 |
| Validation prompts | 6,710 |
| Test prompts | 6,095 |

The prepared prompt manifest SHA-256 is
`c80b0656dd04c28da1d665b4b9a9422f975be5c72b0fe9011b976a3672bb1eac`.

## Training and validation selection

Both candidates start from the same initialization and use the same
deterministic 100-epoch learning-rate schedule. The public-summary builder
verified that the complete 50-epoch history is identical to the first 50
epochs of the 100-epoch history after excluding timing-only fields.

Within each run, BF16 validation selects the saved epoch. The two saved
artifacts are then evaluated over all 6,710 validation prompts in FP32. The
final rule selects the higher mean of root and body cosine; an exact tie would
select 50 epochs. Test results are not used for either selection.

| Candidate | Updates | Training time | BF16 saved epoch | BF16 score | FP32 saved-artifact score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 epochs | 20,150 | 1,018.048 s | 49 | 0.970748425 | 0.970752333 |
| 100 epochs | 40,300 | 2,015.787 s | 95 | 0.971973747 | 0.971975346 |

The 100-epoch saved artifact is selected. FP32 selection compares the two
saved artifacts; it does not re-evaluate discarded per-epoch states.

## Held-out condition fidelity

The selected artifact is evaluated in FP32 over all 6,095 test prompts.
Targets are the cached 2,048-dimensional root/body conditions. NRMSE is RMSE
divided by target RMS over the same elements.

| Output | Cosine mean | Cosine p5 | RMSE | NRMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Root, 1,024 dimensions | 0.978608545 | 0.937584653 | 1.526259338 | 0.206945778 | 1.098811064 |
| Body, 1,024 dimensions | 0.970331443 | 0.915840109 | 1.420681899 | 0.243763376 | 1.021008061 |
| Overall, 2,048 dimensions | 0.975569941 | 0.930388688 | 1.474415923 | 0.221823257 | 1.059909562 |

These test values are reported only after validation selection and do not
affect the selected candidate.

## Held-out motion fidelity

The motion evaluation selects 64 prompts evenly from the 6,095 eligible test
prompts, using both source fields in manifest order. It uses seeds 0, 1, and
2, producing 192 matched teacher/student cases. Every case uses:

- 4 seconds at 20 FPS, or 80 frames;
- 10 diffusion steps;
- CFG weights `[2, 2]`;
- the same Core40 checkpoint and seed on both sides;
- the MiniLM student in float32;
- raw `motion_rep.inverse` output without post-processing.

Deterministic PyTorch, cuDNN, and cuBLAS settings are enabled, TF32 is
disabled, and the first teacher case passes an exact repeatability check. The
ordered selected-prompt digest is
`22111a86318ab5ea930016e9b4ac569a31ec35b73de67eb3068ca8f6dab10ab8`.

| Metric | MiniLM vs teacher, same seed | Teacher, different seeds | Ratio |
| --- | ---: | ---: | ---: |
| Root ADE | 0.047755542 m | 0.164984873 m | 0.289454063 |
| Root FDE | 0.081806103 m | 0.241389780 m | 0.338896297 |
| Global MPJPE | 0.064752256 m | 0.213393457 m | 0.303440681 |
| Root-aligned MPJPE | 0.031403019 m | 0.097685677 m | 0.321470050 |
| Joint velocity error | 0.100059084 m/s | 0.318162510 m/s | 0.314490492 |
| Heading MAE | 3.157179426° | 10.053720313° | — |
| Motion cosine | 0.996353014 | 0.979775245 | — |
| Foot-contact agreement | 0.968977865 | 0.857519531 | — |
| Foot-contact macro F1 | 0.968527125 | 0.848779143 | — |
| Foot-contact macro IoU | 0.949608218 | 0.799196681 | — |

Ratios are same-seed MiniLM/teacher kinematic errors divided by the original
teacher's different-seed diversity. Lower kinematic errors and ratios indicate
closer replacement fidelity.

## Fresh-process latency and memory

Teacher and student measurements run in separate fresh processes on the same
NVIDIA GB10 system with Python 3.11.15, PyTorch 2.13.0+cu130, Transformers
5.8.1, external batch size one, one first encode, one warmup, and 100
synchronized warm encodes. Encoder model loading requests BF16; the Core40
model in the full-stack benchmark remains float32. All four reports use the
same prompt, represented publicly only by SHA-256
`cdc2de1a9c1ec70bcd9433d7a1b9f0911c310c24c7727ab9d21834ead96838e6`.

The full condition stack measures:

```text
prompt -> encoder -> Core40 root/body condition path -> [1, 1, 2048]
```

| Full condition stack | Teacher | MiniLM | Comparison |
| --- | ---: | ---: | ---: |
| Load time | 60.556224 s | 1.901364 s | 31.848825x faster |
| First encode p50 | 826.838522 ms | 439.608479 ms | 1.880852x faster |
| Warm encode p50 | 99.579304 ms | 1.150588 ms | 86.546453x faster |
| Warm encode mean | 100.024164 ms | 1.230384 ms | 81.295056x faster |
| CUDA allocated peak | 16,130,574,336 B | 1,005,858,304 B | 93.764275% lower |
| Process RSS peak | 17,059,147,776 B | 2,593,554,432 B | 84.796694% lower |
| Parameter bytes | 15,942,634,720 B | 821,469,152 B | 94.847344% lower |

Encoder-only measurements isolate the replaced component:

| Encoder only | Teacher | MiniLM | Comparison |
| --- | ---: | ---: | ---: |
| Load time | 59.904526 s | 1.469676 s | 40.760376x faster |
| First encode p50 | 806.519981 ms | 456.965784 ms | 1.764946x faster |
| Warm encode p50 | 99.631198 ms | 1.236596 ms | 80.568921x faster |
| Warm encode mean | 100.118089 ms | 1.351346 ms | 74.087693x faster |
| CUDA allocated peak | 15,216,503,808 B | 90,812,928 B | 99.403195% lower |
| Process RSS peak | 16,866,308,096 B | 1,706,418,176 B | 89.882681% lower |
| Parameter bytes | 15,177,621,504 B | 56,455,936 B | 99.628032% lower |

CUDA figures are PyTorch allocator values and exclude driver/context memory.
RSS includes transient loading. These can describe overlapping physical
memory on a unified-memory system and must not be added. Parameter bytes
describe loaded tensors, not the serialized artifact size. Diffusion, motion
decoding, post-processing, and rendering are outside the timed operation.

Teacher loading includes local resolution of pinned Hugging Face snapshots;
student loading includes local artifact validation. Load results can vary
with the operating-system page cache.

## Reproduce the reports

Run from the repository root. The complete pipeline requires the Core40
checkpoint and access to the revision-pinned teacher models.

Prepare prompts and cache teacher conditions:

```bash
uv sync --python 3.11 --no-install-project
uv run hf auth login

uv run python scripts/minilm/prepare_prompts.py \
  --output artifacts/data/prompts-core40-timeline.jsonl \
  --split-ratios 0.8 0.1 0.1 \
  --seed 20260726

uv run python scripts/minilm/cache_teacher.py \
  --input artifacts/data/prompts-core40-timeline.jsonl \
  --output-dir artifacts/teacher-core40-timeline \
  --checkpoint checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors \
  --shard-size 256 \
  --device cuda
```

Train both prefix-matched candidates:

```bash
for epochs in 50 100; do
  uv run python scripts/minilm/train.py \
    --cache-dir artifacts/teacher-core40-timeline \
    --output-dir "artifacts/minilm-ardy-core40-timeline-${epochs}e" \
    --ardy-model ARDY-Core-RP-20FPS-Horizon40 \
    --pooling-mode mean_cls_max_std \
    --adapter-dim 1536 \
    --epochs "${epochs}" \
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
done
```

Evaluate both complete validation splits in FP32:

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

After validation selects 100 epochs, install a clean canonical copy and run
the test evaluations:

```bash
test ! -e artifacts/minilm-ardy-core40
cp -a artifacts/minilm-ardy-core40-timeline-100e \
  artifacts/minilm-ardy-core40

uv run python scripts/minilm/evaluate_conditions.py \
  --teacher-cache artifacts/teacher-core40-timeline \
  --student-path artifacts/minilm-ardy-core40 \
  --expected-ardy-model ARDY-Core-RP-20FPS-Horizon40 \
  --split test \
  --batch-size 64 \
  --device cuda \
  --dtype float32 \
  --output artifacts/evaluation/conditions-timeline-test.json

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

Run each benchmark only after the previous process exits:

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

Build the prompt-free public aggregate:

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

Raw reports remain outside the public aggregate. The builder validates their
lineage, complete FP32 validation protocol, selected artifact, test scope,
deterministic motion protocol, benchmark pairing, and public-field allowlist,
and records each input report's hash.

## Interpretation limits

- Motion metrics measure same-seed replacement fidelity to the teacher, not
  ground-truth text-to-motion semantics.
- The proprietary Rigplay test split and TMR evaluator are unavailable, so
  these values are not paper-comparable FID or R-precision.
- Timing and memory are specific to the recorded software, hardware, and
  fresh-process protocol.
- One deterministic training seed does not establish statistical
  significance.
- Within-run saved epochs were selected with BF16 autocast. FP32 selection
  compares the saved 50- and 100-epoch artifacts, not every discarded epoch.

# MiniLM distillation results

This report records aggregate results from the completed local Core40
experiment. They are measured results, not projected or paper-reported
numbers. The student weights, teacher cache, prompt manifest, dataset records,
generated motions, and prompt-level JSON reports are intentionally excluded
from the public source repository.

Training data includes [Motion Data by Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the
[BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).
See the [third-party model and data notices](../THIRD_PARTY_MODELS_AND_DATA.md)
before reproducing the experiment or redistributing a locally produced model.

## Scope and environment

- ARDY source: upstream `main` at
  `693f74d13b3d04a0a22ce127ee79c929dd89756b`
- ARDY checkpoint: `ARDY-Core-RP-20FPS-Horizon40`
- checkpoint SHA-256:
  `1019d0bf269cf8d1b3e3e9b4a384a58c112672959b071279ddb65814d77660cd`
- student base: `sentence-transformers/all-MiniLM-L6-v2`
- machine: NVIDIA GB10, aarch64 Linux, unified memory
- runtime: Python 3.11.15, PyTorch 2.13.0+cu130, Transformers 5.8.1
- inference precision: bfloat16 encoders; the released ARDY model remains
  float32

The locally retained artifact is 113,689,186 bytes on disk and has fingerprint
`04d94fa2b8e122ac2f8397c65f22adff00d6e8936813b94180360b6599e014b9`.
It is intentionally restricted to the checkpoint above and is not
distributed by this repository.

## Data and training

The prompt corpus was extracted from the seven description fields in
BONES-SEED metadata, normalized, globally deduplicated, and split by content
family to prevent related descriptions from crossing splits.

| Item | Result |
|---|---:|
| Unique prompts before sampling | 50,246 |
| Sampled prompts | 16,384 |
| Train / validation / test | 13,151 / 1,641 / 1,592 |
| Content-family groups | 3,768 |
| Production-teacher cache time | 1,448.49 s |
| Student training | 50 epochs, 215.05 s |
| Best validation root/body mean cosine | 0.95290 |

Teacher caching used the production LLM2Vec wrapper at its required internal
batch size of one. Student training used batch size 128, rich
`mean_cls_max_std` pooling, a 1,536-dimensional shared adapter, and separate
1,024-dimensional root/body heads.

## Held-out condition fidelity

These values compare the two bias-free 1,024-dimensional student conditions
with the original checkpoint-projected LLM2Vec conditions on all 1,592
held-out prompts.

| Branch | Cosine mean | Cosine p5 | RMSE | NRMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Root | 0.96411 | 0.89095 | 1.95419 | 0.26401 | 1.40455 |
| Body | 0.95181 | 0.84966 | 1.83318 | 0.30140 | 1.30885 |
| Combined | 0.95933 | 0.87277 | 1.89465 | 0.27968 | 1.35670 |

The detailed prompt-level report is retained locally under the Git-ignored
`artifacts/` directory. The prompt-free aggregate values are also recorded in
[`reports/minilm_core40_summary.json`](../reports/minilm_core40_summary.json).

## Memory and latency

The primary comparison measures the same operation in fresh processes:

```text
one prompt -> encoder -> Core40 root/body checkpoint projections -> [1, 1, 2048]
```

Both sides use external batch size one, 100 synchronized CUDA warm runs, the
same prompt, the same Core40 model, and include the checkpoint projection
biases. This full-stack measurement also retains the legacy 4,096-dimensional
projection parameters in the student process for checkpoint compatibility.

| Full condition stack | Original LLM2Vec | MiniLM student | Change |
|---|---:|---:|---:|
| Warm latency p50 | 98.970 ms | 1.185 ms | 83.53x faster |
| Warm latency mean | 99.719 ms | 1.269 ms | 78.60x faster |
| Load time | 55.857 s | 1.832 s | 30.49x faster |
| First encode | 798.682 ms | 380.459 ms | 2.10x faster |
| CUDA allocated peak | 15.023 GiB | 0.937 GiB | 93.76% lower |
| Process RSS peak | 15.264 GiB | 2.415 GiB | 84.18% lower |
| Combined parameter bytes | 14.848 GiB | 0.765 GiB | 94.85% lower |

Encoder-only measurements isolate the component being replaced:

| Encoder only | Original LLM2Vec | MiniLM student | Change |
|---|---:|---:|---:|
| Warm latency p50 | 98.938 ms | 1.240 ms | 79.76x faster |
| CUDA allocated peak | 14.172 GiB | 0.085 GiB | 99.40% lower |
| Process RSS peak | 15.709 GiB | 1.590 GiB | 89.88% lower |
| Parameter bytes | 14.135 GiB | 0.053 GiB | 99.63% lower |

CUDA values are PyTorch allocator measurements and exclude the driver/context.
RSS includes transient model loading. On this unified-memory machine both
should be read together. Parameter bytes describe the loaded dtypes, not the
artifact's serialized disk size. The timed operation ends after root/body
condition projection; diffusion, motion decoding, post-processing, and
rendering are deliberately excluded, so the latency ratio is not an
end-to-end motion-generation speedup. Detailed source reports are retained
locally under the Git-ignored `artifacts/` directory.

## Generated-motion fidelity

The motion comparison uses 64 held-out prompts selected only from the four
natural-description fields, as a proxy for the supported well-formed English
domain. Each prompt is generated with seeds 0, 1, and 2, giving 192 matched
teacher/student generations. Every case uses 80 frames at 20 FPS, 10
diffusion steps, CFG weights `[2, 2]`, identical noise, and the same Core40
denoiser. Metrics use the raw `motion_rep.inverse` output without
post-processing.

| Metric | Matched MiniLM vs LLM2Vec | LLM2Vec different-seed diversity | Ratio |
|---|---:|---:|---:|
| Global MPJPE | 0.1352 m | 0.3366 m | 0.402 |
| Root ADE | 0.1087 m | 0.2884 m | 0.377 |
| Root FDE | 0.1850 m | 0.4380 m | 0.422 |
| Root-aligned MPJPE | 0.0523 m | 0.1125 m | 0.465 |
| Joint velocity error | 0.1838 m/s | 0.4735 m/s | 0.388 |
| Heading MAE | 7.477 degrees | 13.734 degrees | — |
| Motion cosine | 0.98778 | 0.94529 | — |
| Foot-contact agreement | 0.94910 | 0.83376 | — |
| Foot-contact macro F1 | 0.94714 | 0.82967 | — |
| Foot-contact macro IoU | 0.91590 | 0.76633 | — |

The ratio column divides replacement error by the original encoder's
different-seed error; lower is better. Detailed case reports are retained
locally under the Git-ignored `artifacts/` directory and are not distributed
because they contain licensed prompt text.

## Interpretation limits

The public ARDY release does not include the proprietary Rigplay test split or
the paper's TMR evaluator. Consequently, these motion results measure
same-seed replacement fidelity to the original encoder, not ground-truth text
semantics and not paper-comparable FID or R-precision. The natural-description
source filter is a reproducible corpus proxy; it is not an automated spelling
or grammar certification. The supported deployment assumption remains
typo-free English motion prompts.

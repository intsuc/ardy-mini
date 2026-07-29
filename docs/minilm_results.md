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

The locally retained artifact is 113,713,137 bytes on disk and has fingerprint
`b2e4af890d4a733049a377b96355eb1f3a5716378f9304010906823cf6af7fcb`.
It is intentionally restricted to the checkpoint above and is not
distributed by this repository.

## Data and training

The prompt corpus was extracted from the seven description fields in
BONES-SEED metadata, normalized, globally deduplicated, and split by content
family. The original validation and test records were then frozen
byte-for-byte while every eligible BONES-SEED training prompt was retained.

| Item | Result |
|---|---:|
| Unique source prompts | 50,246 |
| Baseline train / validation / test | 13,151 / 1,641 / 1,592 |
| Expanded train / validation / test | 40,433 / 1,641 / 1,592 |
| Added unique training prompts | 27,282 |
| Expanded content-family groups | 3,807 |
| Baseline / expanded teacher-cache time | 1,448.49 / 3,755.58 s |

Teacher caching used the production LLM2Vec wrapper at its required internal
batch size of one. Student training used batch size 128, rich
`mean_cls_max_std` pooling, a 1,536-dimensional shared adapter, and separate
1,024-dimensional root/body heads. All runs used one training seed.

| Run | Train prompts | Epochs | Updates | Example exposures | Time | Best val cosine |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13,151 | 50 | 5,150 | 657,550 | 215.05 s | 0.95290 |
| Expanded, compute-matched | 40,433 | 16 | 5,056 | 646,928 | 202.97 s | 0.96055 |
| Expanded, adopted | 40,433 | 50 | 15,800 | 2,021,650 | 626.07 s | 0.96612 |

The 16-epoch control differs from the baseline by only 1.83% in optimizer
updates and 1.62% in example exposures. Its improvement therefore provides a
controlled estimate of the value of prompt diversity. The adopted run
continues to 50 epochs; its best validation score occurs at epoch 49. Its
additional gain shows that 16 epochs was not the optimum, but combines the
larger corpus with additional optimization and should not be interpreted as a
second compute-matched comparison.

Most of the expansion adds wording diversity within existing motion families:
the training corpus grows by 207.45%, while represented training families
increase from 3,034 to 3,073.

### Corpus expansion approaches

The adopted expansion uses only previously unused BONES-SEED descriptions.
This was the cleanest controlled test and introduced no new data-license
domain. No external motion-caption dataset and no synthetic text was used in
the adopted weights.

The following sources were considered but not imported:

| Source | Potential text | Decision for this run |
|---|---:|---|
| [HumanML3D](https://github.com/EricGuo5513/HumanML3D#statistics-of-humanml3d) | 44,970 descriptions | Obtain explicit caption/trained-artifact permission before combining its upstream dataset chain. |
| [KIT Motion-Language](https://motion-annotation.humanoids.kit.edu/dataset/) | 6,353 annotations | Obtain explicit raw-text redistribution and commercial-use terms first. |
| [BABEL](https://babel.is.tue.mpg.de/license.html) | Dense action labels | Excluded because its published terms are research/non-commercial and restrict distribution. |
| [Motion-X](https://github.com/IDEA-Research/Motion-X#1-request-authorization) | Motion annotations | Excluded because its released terms are non-commercial. |

These are conservative project-distribution decisions, not legal advice or a
claim about dataset quality.

A future license-contained augmentation can use project-authored semantic
programs such as action × direction × speed × style × posture × limb, plus
simultaneous/sequential transitions. It should generate only from training
parents, group all surface forms by their canonical program, reject
contradictions and exact/semantic collisions with frozen evaluation prompts,
and record template version, seed, and parent provenance. Vocabulary may be
seeded from resources with explicit terms such as
[WordNet](https://wordnet.princeton.edu/license-and-commercial-use) or
[Wikidata](https://www.wikidata.org/wiki/Wikidata:Licensing), subject to their
notices. This avoids copying another motion dataset's captions; it does not
remove the BONES, teacher-model, or checkpoint terms already applicable to
trained artifacts.

## Held-out condition fidelity

All three runs below use the same 1,592 prompt texts, original float32 teacher
targets, order, and evaluator. Under nearly matched optimization compute, the
expanded run lowers overall RMSE by 8.46%. The adopted run lowers it by 15.14%
from baseline.

| Run | Overall cosine | Cosine p5 | RMSE | NRMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.95933 | 0.87277 | 1.89465 | 0.27968 | 1.35670 |
| Expanded, 16 epochs | 0.96579 | 0.89964 | 1.73431 | 0.25601 | 1.24738 |
| Expanded, 50 epochs | 0.97066 | 0.90470 | 1.60773 | 0.23732 | 1.13203 |

Final branch-level results are:

| Branch | Cosine mean | Cosine p5 | RMSE | NRMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Root | 0.97431 | 0.91548 | 1.65038 | 0.22296 | 1.16578 |
| Body | 0.96492 | 0.89122 | 1.56391 | 0.25713 | 1.09829 |

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
The expanded students have the same graph, 28,227,968 parameters, and tensor
shapes as the baseline student. A fresh baseline/expanded A/B recorded
identical 56,455,936 parameter bytes and 90,812,928-byte CUDA allocation
peaks; repeated latency ranges overlapped. Training for longer therefore
changes fidelity and weight values, not deployment memory or compute.

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

| Metric | Baseline | Expanded 16e | Adopted 50e | Teacher diversity |
|---|---:|---:|---:|---:|
| Global MPJPE | 0.13517 m | 0.12329 m | 0.11748 m | 0.33660 m |
| Root ADE | 0.10871 m | 0.09997 m | 0.09242 m | 0.28836 m |
| Root FDE | 0.18495 m | 0.17643 m | 0.17470 m | 0.43799 m |
| Root-aligned MPJPE | 0.05227 m | 0.04582 m | 0.04724 m | 0.11248 m |
| Joint velocity error | 0.18385 m/s | 0.17115 m/s | 0.15615 m/s | 0.47346 m/s |
| Heading MAE | 7.47676° | 6.52046° | 6.41535° | 13.73390° |
| Motion cosine | 0.98778 | 0.98881 | 0.98830 | 0.94529 |
| Foot-contact agreement | 0.94910 | 0.94521 | 0.95122 | 0.83376 |
| Foot-contact macro F1 | 0.94714 | 0.94250 | 0.94773 | 0.82967 |
| Foot-contact macro IoU | 0.91590 | 0.91089 | 0.91972 | 0.76633 |

The mean of the five kinematic errors normalized by original-encoder
different-seed diversity falls from `0.41077` to `0.37692` in the
compute-matched run and to `0.36364` in the adopted run. This is an 11.47%
reduction from baseline. Compared with 16 epochs, 50 epochs improves global
MPJPE by 4.71%, root ADE by 7.55%, velocity error by 8.76%, and all three
contact metrics. Root-aligned MPJPE regresses by 0.00142 m and motion cosine
by 0.00051 relative to 16 epochs, but both remain better than baseline.

Detailed case reports are retained locally under the Git-ignored `artifacts/`
directory and are not distributed because they contain licensed prompt text.

## Interpretation limits

The public ARDY release does not include the proprietary Rigplay test split or
the paper's TMR evaluator. Consequently, these motion results measure
same-seed replacement fidelity to the original encoder, not ground-truth text
semantics and not paper-comparable FID or R-precision. The natural-description
source filter is a reproducible corpus proxy; it is not an automated spelling
or grammar certification. The supported deployment assumption remains
typo-free English motion prompts. This single-seed, same-source experiment
does not establish statistical significance or cross-dataset language
generalization.

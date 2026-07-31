# Model terms

Llama 3 ARDY Mini Core40 Browser is a composite model. No single license in
this repository replaces the terms that apply to its respective upstream
component. Recipients must comply with all applicable terms described below.

## NVIDIA ARDY

The motion-generation and motion-decoding components are derived from
`nvidia/ARDY-Core-RP-20FPS-Horizon40`. They are licensed by NVIDIA Corporation
under the NVIDIA Open Model License Agreement in
[`LICENSES/NVIDIA_OPEN_MODEL_LICENSE.txt`](LICENSES/NVIDIA_OPEN_MODEL_LICENSE.txt).
The
[NVIDIA Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)
referenced by that agreement also apply.

Required NVIDIA notice:

> Licensed by NVIDIA Corporation under the NVIDIA Open Model License

The ARDY graph architecture and conversion workflow also derive from the
Apache-2.0-licensed [`nv-tlabs/ardy`](https://github.com/nv-tlabs/ardy)
source. The Apache License is included below, and the source modifications are
available from the linked ARDY Mini source repository.

## Meta Llama 3 teacher lineage

The distributed MiniLM condition encoder was trained against conditions
produced by a teacher Built with Meta Llama 3. The Llama weights and LLM2Vec
adapter weights are not included in this repository. The Meta Llama 3
Community License Agreement and Acceptable Use Policy are reproduced in
[`LICENSES/META_LLAMA_3_COMMUNITY_LICENSE.txt`](LICENSES/META_LLAMA_3_COMMUNITY_LICENSE.txt).

Required Meta notice:

> Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

## all-MiniLM-L6-v2 and release-specific modifications

The text encoder is derived from
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
That model is identified by its publisher as Apache-2.0. The Apache License
2.0 is reproduced in [`LICENSES/APACHE-2.0.txt`](LICENSES/APACHE-2.0.txt).

To the extent that intsuc owns copyright in the release-specific training,
conversion, metadata, and other modifications, those contributions are
offered under Apache-2.0. This grant does not alter, replace, or expand rights
under the NVIDIA or Meta terms.

## LLM2Vec

The teacher pipeline used the revision-pinned LLM2Vec MNTP and supervised
adapters from McGill NLP. Their weights are not distributed here. Their MIT
notice is included for lineage and attribution in
[`LICENSES/LLM2VEC-MIT.txt`](LICENSES/LLM2VEC-MIT.txt).

## Training annotations

Training and evaluation descriptions were obtained only from
[`nvidia/SEED-Timeline-Annotations`](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations),
published by NVIDIA Corporation under Creative Commons Attribution 4.0
International. Dataset records, prompt text, and prompt-level reports are not
included here. The attribution, exact revision, source-file hash, and
transformations are recorded in the model card and `MODEL_PROVENANCE.json`.
The CC BY 4.0 legal code is included in
[`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) for reference.

## No additional rights

Nothing in these terms grants permission to use third-party names, logos, or
trademarks except as required for attribution or otherwise allowed by the
applicable license. This model is provided without warranties or conditions
beyond those expressly stated in the applicable licenses.

Questions about this release may be sent to [i@intsuc.dev](mailto:i@intsuc.dev).

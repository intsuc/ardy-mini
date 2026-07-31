---
title: ARDY Mini
emoji: 🕺
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
fullWidth: true
header: mini
short_description: Browser-native text-to-motion generation with WebGPU
models:
  - intsuc/Llama-3-ARDY-Mini-Core40-Browser
datasets:
  - nvidia/SEED-Timeline-Annotations
tags:
  - text-to-motion
  - motion-generation
  - onnx
  - webgpu
  - vrm
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# ARDY Mini

Generate and continuously play interactive human motion entirely in a
WebGPU-capable browser. The demo can preview the generated motion on its
skeleton or on a local VRM avatar.

Built with Meta Llama 3. The browser model is published separately at
[Llama 3 ARDY Mini Core40 Browser](https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser);
review its Model Card and model terms before use.

The model files are downloaded directly from their public Hugging Face Model
Hub repository and cached in the browser. No Hugging Face access token is
embedded in or required by this client. VRM files stay local to the browser.

## Requirements

- A secure browsing context with WebGPU support
- Approximately 653 MiB for the FP16 model on devices supporting
  `shader-f16`, or approximately 684 MiB for the automatically selected FP32
  model
- Typo-free English prompts for the supported input distribution

## Deployment configuration

Set `ARDY_MODEL_BASE_URL` as a public Space Variable to the model repository's
immutable `resolve/<commit-sha>/` URL. Do not configure a Hugging Face token or
other secret: Static Space variables are available to client-side JavaScript.

The source release script builds the pinned web dependencies locally and
stages only the resulting static files and notices. The Space therefore serves
the committed `index.html` directly and does not require a hosted build job.

`VITE_MODEL_BASE_URL` provides the equivalent build-time configuration for
non-Space deployments. `ARDY_MODEL_TERMS_URL` or `VITE_MODEL_TERMS_URL` may be
used to override the model-terms link; otherwise the app derives it from the
immutable model URL.

Source: [nv-tlabs/ardy fork](https://github.com/intsuc/ardy-mini)

Copyright © 2026 intsuc. Contact: [i@intsuc.dev](mailto:i@intsuc.dev)

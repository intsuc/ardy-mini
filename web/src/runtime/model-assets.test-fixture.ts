// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { sha256Hex } from "./hash";
import {
  FP32_PRECISION_FORMAT,
  GRAPH_PRECISION_CONTRACT,
  MIXED_PRECISION_FORMAT,
  MIXED_PRECISION_POLICY_VERSION,
  MIXED_PRECISION_PUBLIC_IO_DTYPE,
  MODEL_FILES_FORMAT,
  MODEL_FILES_SCHEMA_VERSION,
  REQUIRED_WEBGPU_FEATURE,
  type BrowserGraphPrecisionSummary,
  type BrowserModelManifest,
} from "./manifest";

const encoder = new TextEncoder();

export async function gzipTestBytes(
  bytes: Uint8Array,
): Promise<Uint8Array> {
  const input = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(bytes));
      controller.close();
    },
  });
  const compressed = input.pipeThrough(
    new CompressionStream("gzip") as unknown as TransformStream<
      Uint8Array,
      Uint8Array
    >,
  );
  return new Uint8Array(await new Response(compressed).arrayBuffer());
}

export interface ModelTestFixture {
  manifest: BrowserModelManifest;
  manifestBytes: Uint8Array;
  manifestTransportBytes: Uint8Array;
  rawFiles: Map<string, Uint8Array>;
  transports: Map<string, Uint8Array>;
}

export async function createModelTestFixture(
  variant: "fp16" | "fp32" = "fp16",
): Promise<ModelTestFixture> {
  const rawFiles = new Map<string, Uint8Array>([
    ["tokenizer/tokenizer.json", encoder.encode('{"fixture":"tokenizer"}')],
    [
      "tokenizer/tokenizer_config.json",
      encoder.encode('{"fixture":"config"}'),
    ],
    ["text_encoder.onnx", encoder.encode("fixture text encoder")],
    ["denoiser.onnx", encoder.encode("fixture denoiser")],
    ["decoder.onnx", encoder.encode("fixture decoder")],
  ]);
  const transports = new Map<string, Uint8Array>();
  const files: BrowserModelManifest["files"] = {};
  const fileEntries = await Promise.all(
    [...rawFiles].map(async ([path, raw]) => {
      const transportPath = `${path}.gz`;
      const compressed = await gzipTestBytes(raw);
      const [rawSha256, transportSha256] = await Promise.all([
        sha256Hex(raw),
        sha256Hex(compressed),
      ]);
      return {
        path,
        raw,
        transportPath,
        compressed,
        rawSha256,
        transportSha256,
      };
    }),
  );
  for (const {
    path,
    raw,
    transportPath,
    compressed,
    rawSha256,
    transportSha256,
  } of fileEntries) {
    transports.set(transportPath, compressed);
    files[path] = {
      sha256: rawSha256,
      size_bytes: raw.byteLength,
      transport: {
        path: transportPath,
        compression: "gzip",
        sha256: transportSha256,
        size_bytes: compressed.byteLength,
      },
    };
  }

  const summary = <
    GraphName extends keyof BrowserModelManifest["graphs"],
  >(
    graphName: GraphName,
    modelPath: string,
  ): BrowserGraphPrecisionSummary<GraphName> => {
    const outputSize = files[modelPath].size_bytes;
    const identity =
      GRAPH_PRECISION_CONTRACT[graphName].conversion_mode ===
      "fp32-identity";
    const sourceSize = identity ? outputSize : outputSize * 2;
    const sourceInitializers = {
      count_by_dtype: { float: 1 },
      bytes_by_dtype: { float: 4 },
      total_count: 1,
      total_bytes: 4,
    };
    return {
      schema_version: 1,
      graph_name: graphName,
      policy_id: GRAPH_PRECISION_CONTRACT[graphName].policy_id,
      conversion_mode:
        GRAPH_PRECISION_CONTRACT[graphName].conversion_mode,
      source_sha256: identity
        ? files[modelPath].sha256
        : "f".repeat(64),
      output_sha256: files[modelPath].sha256,
      source_size_bytes: sourceSize,
      output_size_bytes: outputSize,
      size_reduction_bytes: sourceSize - outputSize,
      size_reduction_fraction: (sourceSize - outputSize) / sourceSize,
      source_initializers: sourceInitializers,
      output_initializers: identity
        ? sourceInitializers
        : {
            count_by_dtype: { float16: 1 },
            bytes_by_dtype: { float16: 2 },
            total_count: 1,
            total_bytes: 2,
          },
    };
  };
  const precisionGraphs = {
    text_encoder: summary("text_encoder", "text_encoder.onnx"),
    denoiser: summary("denoiser", "denoiser.onnx"),
    decoder: summary("decoder", "decoder.onnx"),
  };
  const sourceBytes = Object.values(precisionGraphs).reduce(
    (total, graph) => total + graph.source_size_bytes,
    0,
  );
  const mixedBytes = Object.values(precisionGraphs).reduce(
    (total, graph) => total + graph.output_size_bytes,
    0,
  );
  const alphas = [
    0.99, 0.95, 0.88, 0.78, 0.66, 0.53, 0.4, 0.28, 0.17, 0.08,
  ];

  const precision: BrowserModelManifest["precision"] =
    variant === "fp16"
      ? {
          format: MIXED_PRECISION_FORMAT,
          policy_version: MIXED_PRECISION_POLICY_VERSION,
          public_io_dtype: MIXED_PRECISION_PUBLIC_IO_DTYPE,
          required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
          source_onnx_bytes: sourceBytes,
          mixed_onnx_bytes: mixedBytes,
          saved_onnx_bytes: sourceBytes - mixedBytes,
          saved_onnx_fraction: (sourceBytes - mixedBytes) / sourceBytes,
          toolchain: {
            torch: "fixture",
            onnx: "fixture",
            onnxruntime: "fixture",
          },
          graphs: precisionGraphs,
        }
      : {
          format: FP32_PRECISION_FORMAT,
          public_io_dtype: MIXED_PRECISION_PUBLIC_IO_DTYPE,
          required_webgpu_features: [],
          onnx_bytes: [...rawFiles]
            .filter(([path]) => path.endsWith(".onnx"))
            .reduce((total, [, bytes]) => total + bytes.byteLength, 0),
          toolchain: {
            torch: "fixture",
            onnx: "fixture",
            onnxruntime: "fixture",
          },
          graphs: {
            text_encoder: {
              model: "text_encoder.onnx",
              sha256: files["text_encoder.onnx"].sha256,
              size_bytes: files["text_encoder.onnx"].size_bytes,
            },
            denoiser: {
              model: "denoiser.onnx",
              sha256: files["denoiser.onnx"].sha256,
              size_bytes: files["denoiser.onnx"].size_bytes,
            },
            decoder: {
              model: "decoder.onnx",
              sha256: files["decoder.onnx"].sha256,
              size_bytes: files["decoder.onnx"].size_bytes,
            },
          },
        };

  const manifest: BrowserModelManifest = {
    format: MODEL_FILES_FORMAT,
    schema_version: MODEL_FILES_SCHEMA_VERSION,
    model: {
      id: "fixture",
      variant: "test",
      revision: "fixture-revision-1",
    },
    files,
    tokenizer: { directory: "tokenizer", max_length: 128 },
    graphs: {
      text_encoder: {
        model: "text_encoder.onnx",
        inputs: {
          inputIds: "input_ids",
          attentionMask: "attention_mask",
          tokenTypeIds: "token_type_ids",
        },
        outputs: { textConditions: "text_conditions" },
      },
      denoiser: {
        model: "denoiser.onnx",
        inputs: {
          cfgWeight: "cfg_weight",
          x: "x",
          historyLength: "history_len",
          generationLength: "generation_len",
          historyMask: "history_mask",
          generationMask: "generation_mask",
          historyTokenMask: "history_token_mask",
          generationTokenMask: "generation_token_mask",
          textConditions: "text_conditions",
          timestep: "timestep",
          firstHeadingAngle: "first_heading_angle",
        },
        outputs: { predX0: "pred_x0" },
      },
      decoder: {
        model: "decoder.onnx",
        inputs: {
          hybridTokens: "hybrid_tokens",
          motionPadMask: "motion_pad_mask",
          globalTranslation: "global_translation",
        },
        outputs: {
          normalizedMotion: "normalized_motion",
          posedJoints: "posed_joints",
          localRotations: "local_rotations",
          globalRotations: "global_rotations",
          rootPositions: "root_positions",
          footContacts: "foot_contacts",
          globalRootHeading: "global_root_heading",
        },
      },
    },
    precision,
    dimensions: {
      fps: 20,
      num_frames_per_token: 4,
      max_tokens: 20,
      max_frames: 80,
      generation_tokens: 10,
      generation_frames: 40,
      history_tokens: 10,
      history_frames: 40,
      root_features_per_frame: 5,
      nframe_root_dim: 20,
      latent_dim: 128,
      hybrid_dim: 148,
      motion_dim: 330,
      body_dim: 325,
      text_condition_dim: 2048,
      num_joints: 27,
    },
    generation: {
      min_frames: 40,
      max_frames: 200,
      default_cfg_weight: 2,
      denoising_steps: 10,
    },
    diffusion: {
      timesteps: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      alphas_cumprod: alphas,
      alphas_cumprod_prev: [1, ...alphas.slice(0, -1)],
    },
    recenter: {
      root_mean: [0, 0, 0, 0, 0],
      root_std: [1, 1, 1, 1, 1],
      position_indices: [0, 1, 2],
      heading_indices: [3, 4],
    },
    latent_quantization: {
      levels: new Array(128).fill(4),
      mean: new Array(128).fill(0),
      std: new Array(128).fill(1),
    },
    runtime: {
      contract_revision: 3,
      text_only: true,
      required_webgpu_features:
        variant === "fp16" ? [REQUIRED_WEBGPU_FEATURE] : [],
    },
    notices: ["fixture only"],
    license_notices: [
      {
        component: "fixture",
        license: "MIT",
        notice: "fixture only",
      },
    ],
  };
  const manifestBytes = encoder.encode(JSON.stringify(manifest));
  return {
    manifest,
    manifestBytes,
    manifestTransportBytes: await gzipTestBytes(manifestBytes),
    rawFiles,
    transports,
  };
}

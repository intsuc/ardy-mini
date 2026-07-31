// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";

import type { Page, Route } from "@playwright/test";

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
} from "../src/runtime/manifest";
import type { BrowserModelVariant } from "../src/runtime/model-variant";
import { WORKER_PROTOCOL_VERSION } from "../src/runtime/protocol";

const developmentModelFamilyPath =
  "/models/ardy-minilm-core40-browser-v1/";

export function developmentModelPath(
  variant: BrowserModelVariant,
): string {
  return `${developmentModelFamilyPath}${variant}/`;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function compressed(bytes: Uint8Array): Uint8Array {
  return gzipSync(bytes, { level: 9 });
}

export interface MockModelFiles {
  variant: BrowserModelVariant;
  basePath: string;
  manifest: BrowserModelManifest;
  manifestTransport: Uint8Array;
  transports: Readonly<Record<string, Uint8Array>>;
  transportSizeBytes: number;
}

export interface MockModelFilesOptions {
  variant?: BrowserModelVariant;
}

function fixturePayload(
  label: string,
  paddingBytes: number,
): Uint8Array {
  const prefix = new TextEncoder().encode(label);
  const bytes = new Uint8Array(prefix.byteLength + paddingBytes);
  bytes.set(prefix);
  let state = 0x9e37_79b9;
  for (let index = prefix.byteLength; index < bytes.byteLength; index += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    bytes[index] = state & 0xff;
  }
  return bytes;
}

function openEndedRangeStart(
  header: string | undefined,
  size: number,
): number | null | undefined {
  if (header === undefined) return undefined;
  const match = /^bytes=(\d+)-$/i.exec(header.trim());
  if (match === null) return null;
  const start = Number(match[1]);
  return Number.isSafeInteger(start) && start < size ? start : null;
}

async function fulfillModelFile(
  route: Route,
  bytes: Uint8Array,
): Promise<void> {
  const start = openEndedRangeStart(
    route.request().method() === "GET"
      ? route.request().headers().range
      : undefined,
    bytes.byteLength,
  );
  const commonHeaders = {
    "Accept-Ranges": "bytes",
  };
  if (start === null) {
    await route.fulfill({
      status: 416,
      contentType: "application/gzip",
      headers: {
        ...commonHeaders,
        "Content-Length": "0",
        "Content-Range": `bytes */${bytes.byteLength}`,
      },
      body: "",
    });
    return;
  }

  const offset = start ?? 0;
  const responseBytes = bytes.subarray(offset);
  await route.fulfill({
    status: start === undefined ? 200 : 206,
    contentType: "application/gzip",
    headers: {
      ...commonHeaders,
      "Content-Length": String(responseBytes.byteLength),
      ...(start === undefined
        ? {}
        : {
            "Content-Range":
              `bytes ${start}-${bytes.byteLength - 1}/${bytes.byteLength}`,
          }),
    },
    ...(route.request().method() === "HEAD"
      ? {}
      : { body: Buffer.from(responseBytes) }),
  });
}

export function createMockModelFiles(
  options: MockModelFilesOptions = {},
): MockModelFiles {
  const variant = options.variant ?? "fp16";
  const encoder = new TextEncoder();
  const rawFiles = new Map<string, Uint8Array>([
    ["tokenizer/tokenizer.json", encoder.encode('{"fixture":"tokenizer"}')],
    [
      "tokenizer/tokenizer_config.json",
      encoder.encode('{"fixture":"config"}'),
    ],
    ["text_encoder.onnx", encoder.encode("fixture text encoder")],
    ["denoiser.onnx", encoder.encode("fixture denoiser")],
    [
      "decoder.onnx",
      fixturePayload(
        "fixture decoder",
        variant === "fp32" ? 4 * 1024 : 0,
      ),
    ],
  ]);
  const files: BrowserModelManifest["files"] = {};
  const transports: Record<string, Uint8Array> = {};
  for (const [path, raw] of rawFiles) {
    const transport = compressed(raw);
    transports[`${path}.gz`] = transport;
    files[path] = {
      sha256: sha256(raw),
      size_bytes: raw.byteLength,
      transport: {
        path: `${path}.gz`,
        compression: "gzip",
        sha256: sha256(transport),
        size_bytes: transport.byteLength,
      },
    };
  }

  const precisionSummary = <
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
    text_encoder: precisionSummary(
      "text_encoder",
      "text_encoder.onnx",
    ),
    denoiser: precisionSummary("denoiser", "denoiser.onnx"),
    decoder: precisionSummary("decoder", "decoder.onnx"),
  };
  const sourceBytes = Object.values(precisionGraphs).reduce(
    (total, graph) => total + graph.source_size_bytes,
    0,
  );
  const mixedBytes = Object.values(precisionGraphs).reduce(
    (total, graph) => total + graph.output_size_bytes,
    0,
  );
  const fp32Graphs = {
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
  };
  const fp32Bytes = Object.values(fp32Graphs).reduce(
    (total, graph) => total + graph.size_bytes,
    0,
  );
  const alphas = [
    0.99, 0.95, 0.88, 0.78, 0.66, 0.53, 0.4, 0.28, 0.17, 0.08,
  ];

  const manifest: BrowserModelManifest = {
    format: MODEL_FILES_FORMAT,
    schema_version: MODEL_FILES_SCHEMA_VERSION,
    model: {
      id: "ardy-minilm-core40-browser-v1",
      variant: "MiniLM Core40 interactive",
      revision: "a".repeat(64),
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
    precision:
      variant === "fp16"
        ? {
            format: MIXED_PRECISION_FORMAT,
            policy_version: MIXED_PRECISION_POLICY_VERSION,
            public_io_dtype: MIXED_PRECISION_PUBLIC_IO_DTYPE,
            required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
            source_onnx_bytes: sourceBytes,
            mixed_onnx_bytes: mixedBytes,
            saved_onnx_bytes: sourceBytes - mixedBytes,
            saved_onnx_fraction:
              (sourceBytes - mixedBytes) / sourceBytes,
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
            onnx_bytes: fp32Bytes,
            toolchain: {
              torch: "fixture",
              onnx: "fixture",
              onnxruntime: "fixture",
            },
            graphs: fp32Graphs,
          },
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
  const manifestBytes = encoder.encode(
    `${JSON.stringify(manifest)}\n`,
  );
  const basePath = developmentModelPath(variant);
  return {
    variant,
    basePath,
    manifest,
    manifestTransport: compressed(manifestBytes),
    transports,
    transportSizeBytes: Object.values(files).reduce(
      (total, file) => total + file.transport.size_bytes,
      0,
    ),
  };
}

export async function routeMockModelFiles(
  page: Page,
  files: MockModelFiles,
): Promise<void> {
  const modelPath = files.basePath;
  await page.route(
    `**${modelPath}**`,
    async (route) => {
      const requestPath = new URL(route.request().url()).pathname;
      if (
        requestPath ===
        `${modelPath}model.json.gz`
      ) {
        await fulfillModelFile(route, files.manifestTransport);
        return;
      }
      const relativePath = requestPath.slice(
        modelPath.length,
      );
      const transport = files.transports[relativePath];
      if (transport !== undefined) {
        await fulfillModelFile(route, transport);
        return;
      }
      await route.fulfill({ status: 404, body: "Not found" });
    },
  );
}

export async function installMockModelWorker(
  page: Page,
  manifest: BrowserModelManifest,
): Promise<void> {
  await page.addInitScript(
    ({ model, protocolVersion }) => {
      class MockModelWorker extends EventTarget {
        constructor() {
          super();
          queueMicrotask(() => {
            this.emit({
              type: "workerReady",
              protocolVersion,
            });
          });
        }

        emit(data: unknown): void {
          this.dispatchEvent(new MessageEvent("message", { data }));
        }

        postMessage(value: unknown): void {
          if (
            typeof value !== "object" ||
            value === null ||
            !("type" in value) ||
            !("requestId" in value)
          ) {
            return;
          }
          const command = value as {
            type: string;
            requestId: string;
            baseUrl?: unknown;
          };
          if (command.type === "getWebGpuCapabilities") {
            queueMicrotask(() => {
              this.emit({
                type: "webGpuCapabilities",
                requestId: command.requestId,
                shaderF16:
                  model.runtime.required_webgpu_features.includes(
                    "shader-f16",
                  ),
              });
            });
            return;
          }
          if (command.type === "getStatus") {
            queueMicrotask(() => {
              this.emit({
                type: "status",
                requestId: command.requestId,
                status: { state: "empty" },
              });
            });
            return;
          }
          if (command.type === "loadModel") {
            if (typeof command.baseUrl !== "string") {
              throw new TypeError("loadModel.baseUrl must be a string");
            }
            const baseUrl = command.baseUrl;
            setTimeout(() => {
              void (async () => {
                const {
                  loadModelAssets,
                  markModelCacheComplete,
                } = await import("/src/runtime/model-assets.ts");
                const assets = await loadModelAssets(
                  baseUrl,
                  (progress) => {
                    this.emit({
                      type: "progress",
                      requestId: command.requestId,
                      ...progress,
                    });
                  },
                );
                this.emit({
                  type: "progress",
                  requestId: command.requestId,
                  stage: "loading-tokenizer",
                  completed: 0,
                  total: 1,
                });
                await Promise.all(
                  Object.keys(model.files).map((path) =>
                    assets.read(path),
                  ),
                );
                await new Promise((resolve) =>
                  setTimeout(resolve, 200),
                );
                this.emit({
                  type: "progress",
                  requestId: command.requestId,
                  stage: "loading-tokenizer",
                  completed: 1,
                  total: 1,
                });
                for (let completed = 1; completed <= 3; completed += 1) {
                  this.emit({
                    type: "progress",
                    requestId: command.requestId,
                    stage: "loading-sessions",
                    completed,
                    total: 3,
                  });
                }
                await markModelCacheComplete(assets);
                this.emit({
                  type: "modelLoaded",
                  requestId: command.requestId,
                  model: {
                    id: model.model.id,
                    variant: model.model.variant,
                    revision: model.model.revision,
                    fps: model.dimensions.fps,
                    minFrames: model.generation.min_frames,
                    maxFrames: model.generation.max_frames,
                    generationFrames:
                      model.dimensions.generation_frames,
                    capabilities: {
                      streamingChunks: true,
                      sessionContinuation: true,
                      branching: true,
                      richMotionOutputs: true,
                      motionCorrection: false,
                    },
                    manifest: model,
                  },
                });
              })().catch((error: unknown) => {
                const normalized =
                  error instanceof Error
                    ? error
                    : new Error(String(error));
                this.emit({
                  type: "error",
                  requestId: command.requestId,
                  error: {
                    name: normalized.name,
                    message: normalized.message,
                    stack: normalized.stack,
                  },
                });
              });
            }, 75);
            return;
          }
          if (command.type === "resetSession") {
            queueMicrotask(() => {
              this.emit({
                type: "sessionReset",
                requestId: command.requestId,
                seed: 2,
                frameCount: 0,
              });
            });
            return;
          }
          if (command.type === "dispose") {
            queueMicrotask(() => {
              this.emit({
                type: "disposed",
                requestId: command.requestId,
              });
            });
          }
        }

        terminate(): void {}
      }

      Object.defineProperty(globalThis, "Worker", {
        configurable: true,
        value: MockModelWorker,
      });
    },
    {
      model: manifest,
      protocolVersion: WORKER_PROTOCOL_VERSION,
    },
  );
}

export const missingDevelopmentModelRoute =
  `**${developmentModelFamilyPath}**`;

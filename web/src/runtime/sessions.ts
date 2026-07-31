// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as ort from "onnxruntime-web/webgpu";

import {
  REQUIRED_WEBGPU_FEATURE,
  type BrowserGraphSpecs,
  type BrowserRequiredWebGpuFeatures,
  type GraphSpec,
} from "./manifest";
import type { ModelAssets } from "./model-assets";

export { ort };

export interface RuntimeSessions {
  textEncoder: ort.InferenceSession;
  denoiser: ort.InferenceSession;
  decoder: ort.InferenceSession;
}

export type SessionProgressCallback = (
  completed: number,
  total: number,
  message: string,
) => void;

interface WebGpuApi {
  requestAdapter(): Promise<GPUAdapter | null>;
}

let webGpuAdapter: GPUAdapter | undefined;

function configureOrtAdapter(adapter: GPUAdapter): void {
  if (ort.env.webgpu.adapter !== adapter) {
    ort.env.webgpu.adapter = adapter;
  }
}

async function resolveWebGpuAdapter(): Promise<GPUAdapter> {
  let adapter = webGpuAdapter;
  if (adapter !== undefined) {
    return adapter;
  }
  if (globalThis.isSecureContext === false) {
    throw new Error(
      "WebGPU requires HTTPS or localhost. Open this demo in a secure context and try again.",
    );
  }
  const navigatorWithGpu = globalThis.navigator as
    | (Navigator & { gpu?: WebGpuApi })
    | undefined;
  if (!navigatorWithGpu?.gpu) {
    throw new Error(
      "WebGPU is required. Use a WebGPU-capable browser and device.",
    );
  }
  let requestedAdapter: GPUAdapter | null;
  try {
    requestedAdapter = await navigatorWithGpu.gpu.requestAdapter();
  } catch (error) {
    throw new Error("WebGPU adapter initialization failed.", { cause: error });
  }
  if (requestedAdapter === null) {
    throw new Error(
      "WebGPU is required, but no compatible GPU adapter is available.",
    );
  }
  adapter = requestedAdapter;
  webGpuAdapter = adapter;
  return adapter;
}

export interface WebGpuCapabilities {
  shaderF16: boolean;
}

export async function getWebGpuCapabilities(): Promise<WebGpuCapabilities> {
  const adapter = await resolveWebGpuAdapter();
  return {
    shaderF16: adapter.features.has(REQUIRED_WEBGPU_FEATURE),
  };
}

export async function assertWebGpuAvailable(
  requiredFeatures: BrowserRequiredWebGpuFeatures = [],
): Promise<void> {
  const adapter = await resolveWebGpuAdapter();
  for (const feature of requiredFeatures) {
    if (!adapter.features.has(feature)) {
      throw new Error(
        `The selected model requires the WebGPU feature ${feature}, but the selected GPU adapter does not provide it.`,
      );
    }
  }
  configureOrtAdapter(adapter);
}

function runtimeAssetBaseUrl(): string {
  if (typeof globalThis.location === "undefined") return "/ort/";
  return new URL("../ort/", globalThis.location.href).href;
}

function configureOrt(): void {
  // ORT reports benign WebGPU CPU-fallback assignments at warning severity
  // through console.error. Keep actionable runtime failures visible without
  // presenting expected shape-op placement as an application error.
  ort.env.logLevel = "error";
  // ONNX Runtime's WebGPU execution provider is hosted by its WebAssembly
  // runtime. These settings initialize that host; they do not enable the
  // WebAssembly execution provider or a CPU fallback session.
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? 0 : 1;
  ort.env.wasm.wasmPaths = runtimeAssetBaseUrl();
}

async function createSession(
  assets: ModelAssets,
  graph: GraphSpec<
    Record<string, string>,
    Record<string, string | undefined>
  >,
): Promise<ort.InferenceSession> {
  try {
    const [model, externalData] = await Promise.all([
      assets.read(graph.model),
      Promise.all(
        (graph.external_data ?? []).map(async (spec) => ({
          path: spec.path,
          data: await assets.read(spec.file),
        })),
      ),
    ]);
    return await ort.InferenceSession.create(model, {
      executionProviders: ["webgpu"],
      logSeverityLevel: 3,
      ...(externalData.length === 0 ? {} : { externalData }),
    });
  } finally {
    assets.release(graph.model);
    for (const external of graph.external_data ?? []) {
      assets.release(external.file);
    }
  }
}

async function releaseSessions(sessions: ort.InferenceSession[]): Promise<void> {
  await Promise.allSettled(sessions.map((session) => session.release()));
}

async function createAll(
  assets: ModelAssets,
  graphs: BrowserGraphSpecs,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  const created: ort.InferenceSession[] = [];
  const total = 3;
  try {
    const decoder = await createSession(assets, graphs.decoder);
    created.push(decoder);
    onProgress?.(1, total, "decoder.onnx");
    const textEncoder = await createSession(assets, graphs.text_encoder);
    created.push(textEncoder);
    onProgress?.(2, total, "text_encoder.onnx");
    const denoiser = await createSession(assets, graphs.denoiser);
    created.push(denoiser);
    onProgress?.(total, total, "denoiser.onnx");
    return {
      textEncoder,
      denoiser,
      decoder,
    };
  } catch (error) {
    await releaseSessions(created);
    throw error;
  }
}

export async function createRuntimeSessions(
  assets: ModelAssets,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  await assertWebGpuAvailable(
    assets.manifest.runtime.required_webgpu_features,
  );
  configureOrt();
  return createAll(assets, assets.manifest.graphs, onProgress);
}

export async function disposeRuntimeSessions(
  sessions: RuntimeSessions,
): Promise<void> {
  await releaseSessions([
    sessions.textEncoder,
    sessions.denoiser,
    sessions.decoder,
  ]);
}

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as ort from "onnxruntime-web/webgpu";

import type {
  BrowserGraphSpecs,
  GraphSpec,
} from "./manifest";
import type { ModelPack } from "./model-pack";

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
  requestAdapter(): Promise<unknown | null>;
}

let webGpuReady = false;

export async function assertWebGpuAvailable(): Promise<void> {
  if (webGpuReady) return;
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
  let adapter: unknown | null;
  try {
    adapter = await navigatorWithGpu.gpu.requestAdapter();
  } catch (error) {
    throw new Error("WebGPU adapter initialization failed.", { cause: error });
  }
  if (adapter === null) {
    throw new Error(
      "WebGPU is required, but no compatible GPU adapter is available.",
    );
  }
  webGpuReady = true;
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
  pack: ModelPack,
  graph: GraphSpec<
    Record<string, string>,
    Record<string, string | undefined>
  >,
): Promise<ort.InferenceSession> {
  const [model, externalData] = await Promise.all([
    pack.read(graph.model),
    Promise.all(
      (graph.external_data ?? []).map(async (spec) => ({
        path: spec.path,
        data: await pack.read(spec.file),
      })),
    ),
  ]);
  return ort.InferenceSession.create(model, {
    executionProviders: ["webgpu"],
    logSeverityLevel: 3,
    ...(externalData.length === 0 ? {} : { externalData }),
  });
}

async function releaseSessions(sessions: ort.InferenceSession[]): Promise<void> {
  await Promise.allSettled(sessions.map((session) => session.release()));
}

async function createAll(
  pack: ModelPack,
  graphs: BrowserGraphSpecs,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  const created: ort.InferenceSession[] = [];
  const total = 3;
  try {
    const textEncoder = await createSession(pack, graphs.text_encoder);
    created.push(textEncoder);
    onProgress?.(1, total, "text_encoder.onnx");
    const denoiser = await createSession(pack, graphs.denoiser);
    created.push(denoiser);
    onProgress?.(2, total, "denoiser.onnx");
    const decoder = await createSession(pack, graphs.decoder);
    created.push(decoder);
    onProgress?.(total, total, "decoder.onnx");
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

function releaseGraphAssets(pack: ModelPack, graphs: BrowserGraphSpecs): void {
  for (const graph of [
    graphs.text_encoder,
    graphs.denoiser,
    graphs.decoder,
  ]) {
    pack.release(graph.model);
    for (const external of graph.external_data ?? []) {
      pack.release(external.file);
    }
  }
}

export async function createRuntimeSessions(
  pack: ModelPack,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  await assertWebGpuAvailable();
  configureOrt();
  const graphs = pack.manifest.graphs;
  try {
    return await createAll(pack, graphs, onProgress);
  } finally {
    releaseGraphAssets(pack, graphs);
  }
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

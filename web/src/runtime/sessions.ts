// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as ort from "onnxruntime-web/webgpu";

import type {
  BrowserGraphSpecs,
  GraphSpec,
} from "./manifest";
import type { ModelPack } from "./model-pack";
import type {
  RuntimeBackend,
  RuntimeBackendPreference,
} from "./protocol";

export { ort };

export interface RuntimeSessions {
  textEncoder: ort.InferenceSession;
  denoiser: ort.InferenceSession;
  constraintDenoiser?: ort.InferenceSession;
  decoder: ort.InferenceSession;
  backend: RuntimeBackend;
}

export type SessionProgressCallback = (
  completed: number,
  total: number,
  message: string,
) => void;

function webGpuAvailable(): boolean {
  const navigatorWithGpu = globalThis.navigator as
    | (Navigator & { gpu?: unknown })
    | undefined;
  return navigatorWithGpu?.gpu !== undefined && globalThis.isSecureContext !== false;
}

function configureOrt(wasmPaths = "/ort/"): void {
  // ORT reports benign WebGPU CPU-fallback assignments at warning severity
  // through console.error. Keep actionable runtime failures visible without
  // presenting expected shape-op placement as an application error.
  ort.env.logLevel = "error";
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? 0 : 1;
  ort.env.wasm.wasmPaths = wasmPaths;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function createSession(
  pack: ModelPack,
  graph: GraphSpec<
    Record<string, string>,
    Record<string, string | undefined>
  >,
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
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
    executionProviders,
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
  providers: ort.InferenceSession.ExecutionProviderConfig[],
  backend: RuntimeBackend,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  const created: ort.InferenceSession[] = [];
  const total = graphs.constraint_denoiser === undefined ? 3 : 4;
  try {
    const textEncoder = await createSession(pack, graphs.text_encoder, providers);
    created.push(textEncoder);
    onProgress?.(1, total, "text_encoder.onnx");
    const denoiser = await createSession(pack, graphs.denoiser, providers);
    created.push(denoiser);
    onProgress?.(2, total, "denoiser.onnx");
    let constraintDenoiser: ort.InferenceSession | undefined;
    if (graphs.constraint_denoiser !== undefined) {
      constraintDenoiser = await createSession(
        pack,
        graphs.constraint_denoiser,
        providers,
      );
      created.push(constraintDenoiser);
      onProgress?.(3, total, "denoiser_constraints.onnx");
    }
    const decoder = await createSession(pack, graphs.decoder, providers);
    created.push(decoder);
    onProgress?.(total, total, "decoder.onnx");
    return {
      textEncoder,
      denoiser,
      ...(constraintDenoiser === undefined ? {} : { constraintDenoiser }),
      decoder,
      backend,
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
    graphs.constraint_denoiser,
    graphs.decoder,
  ]) {
    if (graph === undefined) {
      continue;
    }
    pack.release(graph.model);
    for (const external of graph.external_data ?? []) {
      pack.release(external.file);
    }
  }
}

export async function createRuntimeSessions(
  pack: ModelPack,
  preference: RuntimeBackendPreference = "auto",
  wasmPaths?: string,
  onProgress?: SessionProgressCallback,
): Promise<RuntimeSessions> {
  configureOrt(wasmPaths);
  const graphs = pack.manifest.graphs;
  const shouldTryWebGpu = preference !== "wasm" && webGpuAvailable();
  try {
    if (shouldTryWebGpu) {
      try {
        return await createAll(
          pack,
          graphs,
          ["webgpu"],
          "webgpu",
          onProgress,
        );
      } catch (webGpuError) {
        if (isAbortError(webGpuError)) {
          throw webGpuError;
        }
        console.warn("WebGPU initialization failed; retrying with WASM.", webGpuError);
      }
    }
    return await createAll(pack, graphs, ["wasm"], "wasm", onProgress);
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
    ...(sessions.constraintDenoiser === undefined
      ? []
      : [sessions.constraintDenoiser]),
    sessions.decoder,
  ]);
}

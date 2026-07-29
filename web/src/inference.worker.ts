// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  BrowserArdyRuntime,
  RuntimeCancelledError,
  type BrowserArdyGenerationSession,
  type RuntimeContinuationState,
  type RuntimeGenerationChunk,
  type RuntimeGenerationResult,
  type RuntimeProgress,
} from "./runtime/engine";
import { loadModelPackFromTarGzip } from "./runtime/model-pack";
import {
  parseWorkerCommand,
  serializeWorkerError,
  WORKER_PROTOCOL_VERSION,
  type GenerateCommand,
  type GenerationMode,
  type RuntimeCapabilities,
  type WorkerCommand,
  type WorkerEvent,
} from "./runtime/protocol";

interface WorkerPort {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
}

type ActiveKind = "loading" | "generating";

interface ActiveOperation {
  requestId: string;
  kind: ActiveKind;
  controller: AbortController;
  promise: Promise<void>;
  cancelRequestId?: string;
}

const port = self as unknown as WorkerPort;
let runtime: BrowserArdyRuntime | null = null;
let generationSession: BrowserArdyGenerationSession | null = null;
let active: ActiveOperation | null = null;

function post(event: WorkerEvent, transfer?: Transferable[]): void {
  port.postMessage(event, transfer);
}

function requestIdFrom(value: unknown): string {
  if (
    typeof value === "object" &&
    value !== null &&
    "requestId" in value &&
    typeof value.requestId === "string"
  ) {
    return value.requestId;
  }
  return "unknown";
}

function postProgress(requestId: string, progress: RuntimeProgress): void {
  post({
    type: "progress",
    requestId,
    ...progress,
  });
}

function isCancellation(error: unknown, signal: AbortSignal): boolean {
  return (
    signal.aborted ||
    error instanceof RuntimeCancelledError ||
    (error instanceof DOMException && error.name === "AbortError")
  );
}

function postCancelled(operation: ActiveOperation): void {
  if (operation.cancelRequestId !== undefined) {
    post({
      type: "cancelled",
      requestId: operation.cancelRequestId,
      targetRequestId: operation.requestId,
    });
  }
}

function startOperation(
  requestId: string,
  kind: ActiveKind,
  task: (operation: ActiveOperation) => Promise<void>,
): void {
  if (active !== null) {
    post(
      serializeWorkerError(
        requestId,
        new Error(`Worker is already ${active.kind}`),
      ),
    );
    return;
  }
  const controller = new AbortController();
  const operation = {
    requestId,
    kind,
    controller,
    promise: Promise.resolve(),
  } as ActiveOperation;
  active = operation;
  operation.promise = task(operation)
    .catch((error: unknown) => {
      if (isCancellation(error, controller.signal)) {
        postCancelled(operation);
      } else {
        post(serializeWorkerError(requestId, error));
      }
    })
    .finally(() => {
      if (active === operation) {
        active = null;
      }
    });
}

function capabilitiesFor(selectedRuntime: BrowserArdyRuntime): RuntimeCapabilities {
  const manifest = selectedRuntime.manifest;
  const decoderOutputs = manifest.graphs.decoder.outputs;
  const richMotionOutputs =
    decoderOutputs.localRotations !== undefined &&
    decoderOutputs.globalRotations !== undefined &&
    decoderOutputs.rootPositions !== undefined &&
    decoderOutputs.footContacts !== undefined &&
    decoderOutputs.globalRootHeading !== undefined;
  return {
    streamingChunks: true,
    sessionContinuation: true,
    branching: true,
    richMotionOutputs,
    // The UI applies a deterministic browser-native correction pass after
    // each completed decode request. This is distinct from ARDY's optional native C++
    // correction library, which is intentionally not embedded in the pack.
    motionCorrection: true,
  };
}

function translationTuple(
  value: Float32Array | undefined,
): [number, number, number] | undefined {
  return value === undefined ? undefined : [value[0], value[1], value[2]];
}

function cloneContinuation(
  state: RuntimeContinuationState,
): RuntimeContinuationState {
  return {
    frameCount: state.frameCount,
    hybridTokens: new Float32Array(state.hybridTokens),
    hybridDim: state.hybridDim,
    random: { ...state.random },
    initialTranslation: [...state.initialTranslation],
    initialHeading: state.initialHeading,
  };
}

function cloneChunk(chunk: RuntimeGenerationChunk): RuntimeGenerationChunk {
  return {
    ...chunk,
    motion: new Float32Array(chunk.motion),
    joints: new Float32Array(chunk.joints),
    hybridTokens: new Float32Array(chunk.hybridTokens),
    ...(chunk.localRotations === undefined
      ? {}
      : { localRotations: new Float32Array(chunk.localRotations) }),
    ...(chunk.globalRotations === undefined
      ? {}
      : { globalRotations: new Float32Array(chunk.globalRotations) }),
    ...(chunk.rootPositions === undefined
      ? {}
      : { rootPositions: new Float32Array(chunk.rootPositions) }),
    ...(chunk.footContacts === undefined
      ? {}
      : { footContacts: new Uint8Array(chunk.footContacts) }),
    ...(chunk.globalRootHeading === undefined
      ? {}
      : { globalRootHeading: new Float32Array(chunk.globalRootHeading) }),
    timingsMs: { ...chunk.timingsMs },
  };
}

function cloneResult(result: RuntimeGenerationResult): RuntimeGenerationResult {
  return {
    ...result,
    motion: new Float32Array(result.motion),
    joints: new Float32Array(result.joints),
    continuation: cloneContinuation(result.continuation),
    ...(result.localRotations === undefined
      ? {}
      : { localRotations: new Float32Array(result.localRotations) }),
    ...(result.globalRotations === undefined
      ? {}
      : { globalRotations: new Float32Array(result.globalRotations) }),
    ...(result.rootPositions === undefined
      ? {}
      : { rootPositions: new Float32Array(result.rootPositions) }),
    ...(result.footContacts === undefined
      ? {}
      : { footContacts: new Uint8Array(result.footContacts) }),
    ...(result.globalRootHeading === undefined
      ? {}
      : { globalRootHeading: new Float32Array(result.globalRootHeading) }),
    timingsMs: { ...result.timingsMs },
  };
}

function motionTransfers(
  payload: RuntimeGenerationChunk | RuntimeGenerationResult,
): Transferable[] {
  const buffers = new Set<ArrayBuffer>();
  const add = (
    value: Float32Array | Uint8Array | undefined,
  ): void => {
    if (value !== undefined) {
      buffers.add(value.buffer as ArrayBuffer);
    }
  };
  add(payload.motion);
  add(payload.joints);
  add(payload.localRotations);
  add(payload.globalRotations);
  add(payload.rootPositions);
  add(payload.footContacts);
  add(payload.globalRootHeading);
  if ("hybridTokens" in payload) {
    add(payload.hybridTokens);
  }
  if ("continuation" in payload) {
    add(payload.continuation.hybridTokens);
  }
  return [...buffers];
}

function load(command: Extract<WorkerCommand, { type: "loadModelPack" }>): void {
  startOperation(command.requestId, "loading", async (operation) => {
    const pack = await loadModelPackFromTarGzip(
      command.archive,
      (progress) => postProgress(command.requestId, progress),
      operation.controller.signal,
    );
    if (runtime !== null) {
      await runtime.dispose();
      runtime = null;
      generationSession = null;
    }
    const loaded = await BrowserArdyRuntime.create(pack, {
      backend: command.backend,
      wasmPaths: command.wasmPaths,
      signal: operation.controller.signal,
      onProgress: (progress) => postProgress(command.requestId, progress),
    });
    runtime = loaded;
    const capabilities = capabilitiesFor(loaded);
    post({
      type: "modelLoaded",
      requestId: command.requestId,
      model: {
        id: loaded.manifest.model.id,
        variant: loaded.manifest.model.variant,
        backend: loaded.backend,
        fps: loaded.manifest.dimensions.fps,
        minFrames: loaded.manifest.generation.min_frames,
        maxFrames: loaded.manifest.generation.max_frames,
        generationFrames: loaded.manifest.dimensions.generation_frames,
        capabilities,
        manifest: loaded.manifest,
      },
    });
  });
}

function frameCountFor(
  selectedRuntime: BrowserArdyRuntime,
  command: GenerateCommand,
): number {
  return command.durationFrames ??
    Math.round(
      command.durationSeconds! * selectedRuntime.manifest.dimensions.fps,
    );
}

function prepareGenerationSession(
  selectedRuntime: BrowserArdyRuntime,
  command: GenerateCommand,
  mode: GenerationMode,
): BrowserArdyGenerationSession {
  if (mode === "replace") {
    const options = {
      seed: command.seed,
      initialTranslation: translationTuple(command.initialTranslation),
      initialHeading: command.initialHeading,
    };
    if (generationSession === null) {
      generationSession = selectedRuntime.createGenerationSession(options);
    } else {
      generationSession.reset(options);
    }
    return generationSession;
  }
  if (generationSession === null) {
    throw new Error(
      `${mode} generation requires an existing or restored session`,
    );
  }
  if (mode === "branch") {
    generationSession.branch(command.branchFrame!);
  }
  return generationSession;
}

function generate(command: Extract<WorkerCommand, { type: "generate" }>): void {
  if (runtime === null) {
    post(
      serializeWorkerError(
        command.requestId,
        new Error("Load a model pack before generating motion"),
      ),
    );
    return;
  }
  const selectedRuntime = runtime;
  startOperation(command.requestId, "generating", async (operation) => {
    const mode = command.mode ?? "replace";
    const session = prepareGenerationSession(selectedRuntime, command, mode);
    const result = await session.generate({
      prompt: command.prompt,
      durationFrames: frameCountFor(selectedRuntime, command),
      cfgWeight: command.cfgWeight,
      historyFrames: command.historyFrames,
      signal: operation.controller.signal,
      onProgress: (progress) => postProgress(command.requestId, progress),
      onChunk: (chunk) => {
        // The engine retains its chunk arrays until it assembles the final
        // result. Transfer a snapshot so streaming cannot detach engine state.
        const snapshot = cloneChunk(chunk);
        post(
          {
            type: "generationChunk",
            requestId: command.requestId,
            mode,
            chunk: snapshot,
          },
          motionTransfers(snapshot),
        );
      },
    });

    // Protocol-v1 clients need the old terminal event. Clone the v2 completion
    // payload first so its transfer does not detach the legacy result.
    const completionResult =
      command.mode === undefined ? cloneResult(result) : result;
    post(
      {
        type: "generationComplete",
        requestId: command.requestId,
        mode,
        generatedFrameCount: result.frameCount,
        sessionFrameCount: session.frameCount,
        result: completionResult,
      },
      motionTransfers(completionResult),
    );
    if (command.mode === undefined) {
      post(
        {
          type: "generationResult",
          requestId: command.requestId,
          result,
        },
        motionTransfers(result),
      );
    }
  });
}

function resetSession(
  command: Extract<WorkerCommand, { type: "resetSession" }>,
): void {
  if (active !== null) {
    throw new Error(`Worker is already ${active.kind}`);
  }
  if (runtime === null) {
    throw new Error("Load a model pack before resetting the generation session");
  }
  const options = {
    seed: command.seed,
    initialTranslation: translationTuple(command.initialTranslation),
    initialHeading: command.initialHeading,
  };
  if (generationSession === null) {
    generationSession = runtime.createGenerationSession(options);
  } else {
    generationSession.reset(options);
  }
  post({
    type: "sessionReset",
    requestId: command.requestId,
    seed: generationSession.seed,
    frameCount: 0,
  });
}

function restoreContinuation(
  command: Extract<WorkerCommand, { type: "restoreContinuation" }>,
): void {
  if (active !== null) {
    throw new Error(`Worker is already ${active.kind}`);
  }
  if (runtime === null) {
    throw new Error("Load a model pack before restoring a continuation");
  }
  if (generationSession === null) {
    generationSession = runtime.createGenerationSession({
      seed: command.continuation.random.seed,
      initialTranslation: command.continuation.initialTranslation,
      initialHeading: command.continuation.initialHeading,
    });
  }
  generationSession.restore(command.continuation);
  post({
    type: "continuationRestored",
    requestId: command.requestId,
    frameCount: generationSession.frameCount,
  });
}

function cancel(command: Extract<WorkerCommand, { type: "cancel" }>): void {
  if (active?.requestId === command.targetRequestId) {
    active.cancelRequestId = command.requestId;
    active.controller.abort(new RuntimeCancelledError());
    return;
  }
  post({
    type: "cancelled",
    requestId: command.requestId,
    targetRequestId: command.targetRequestId,
  });
}

async function dispose(
  command: Extract<WorkerCommand, { type: "dispose" }>,
): Promise<void> {
  const current = active;
  if (current !== null) {
    current.controller.abort(new RuntimeCancelledError("Runtime disposed"));
    await current.promise;
  }
  generationSession = null;
  if (runtime !== null) {
    await runtime.dispose();
    runtime = null;
  }
  post({ type: "disposed", requestId: command.requestId });
}

function status(command: Extract<WorkerCommand, { type: "getStatus" }>): void {
  if (active?.kind === "loading") {
    post({
      type: "status",
      requestId: command.requestId,
      status: { state: "loading", activeRequestId: active.requestId },
    });
  } else if (runtime === null) {
    post({
      type: "status",
      requestId: command.requestId,
      status: { state: "empty" },
    });
  } else {
    post({
      type: "status",
      requestId: command.requestId,
      status: {
        state: active?.kind === "generating" ? "generating" : "ready",
        ...(active === null ? {} : { activeRequestId: active.requestId }),
        modelId: runtime.manifest.model.id,
        backend: runtime.backend,
        generatedFrameCount: generationSession?.frameCount ?? 0,
        capabilities: capabilitiesFor(runtime),
      },
    });
  }
}

function runSynchronous(requestId: string, task: () => void): void {
  try {
    task();
  } catch (error) {
    post(serializeWorkerError(requestId, error));
  }
}

port.addEventListener("message", (message) => {
  let command: WorkerCommand;
  try {
    command = parseWorkerCommand(message.data);
  } catch (error) {
    post(serializeWorkerError(requestIdFrom(message.data), error));
    return;
  }
  switch (command.type) {
    case "loadModelPack":
      load(command);
      break;
    case "generate":
      generate(command);
      break;
    case "resetSession":
      runSynchronous(command.requestId, () => resetSession(command));
      break;
    case "restoreContinuation":
      runSynchronous(command.requestId, () =>
        restoreContinuation(command),
      );
      break;
    case "cancel":
      cancel(command);
      break;
    case "dispose":
      void dispose(command).catch((error) =>
        post(serializeWorkerError(command.requestId, error)),
      );
      break;
    case "getStatus":
      status(command);
      break;
  }
});

post({
  type: "workerReady",
  protocolVersion: WORKER_PROTOCOL_VERSION,
});

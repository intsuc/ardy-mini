// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  BrowserArdyRuntime,
  RuntimeCancelledError,
  loadModelPackFromFiles,
  parseWorkerCommand,
  serializeWorkerError,
  WORKER_PROTOCOL_VERSION,
  type RuntimeProgress,
  type WorkerCommand,
  type WorkerEvent,
} from "./runtime";

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

function load(command: Extract<WorkerCommand, { type: "loadModelPack" }>): void {
  startOperation(command.requestId, "loading", async (operation) => {
    const pack = await loadModelPackFromFiles(
      command.files,
      (progress) => postProgress(command.requestId, progress),
      operation.controller.signal,
    );
    if (runtime !== null) {
      await runtime.dispose();
      runtime = null;
    }
    const loaded = await BrowserArdyRuntime.create(pack, {
      backend: command.backend,
      wasmPaths: command.wasmPaths,
      signal: operation.controller.signal,
      onProgress: (progress) => postProgress(command.requestId, progress),
    });
    runtime = loaded;
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
      },
    });
  });
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
    const result = await selectedRuntime.generate({
      prompt: command.prompt,
      seed: command.seed,
      durationFrames: command.durationFrames,
      durationSeconds: command.durationSeconds,
      cfgWeight: command.cfgWeight,
      signal: operation.controller.signal,
      onProgress: (progress) => postProgress(command.requestId, progress),
    });
    post(
      {
        type: "generationResult",
        requestId: command.requestId,
        result,
      },
      [result.motion.buffer, result.joints.buffer],
    );
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
      },
    });
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

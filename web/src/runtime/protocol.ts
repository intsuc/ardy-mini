// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const WORKER_PROTOCOL_VERSION = 1;

export type RuntimeBackendPreference = "auto" | "webgpu" | "wasm";
export type RuntimeBackend = "webgpu" | "wasm";

export type RuntimeProgressStage =
  | "reading-pack"
  | "hashing-pack"
  | "loading-tokenizer"
  | "loading-sessions"
  | "encoding-text"
  | "denoising"
  | "decoding";

interface RequestMessage {
  requestId: string;
}

export interface LoadModelPackCommand extends RequestMessage {
  type: "loadModelPack";
  /** Files selected with an `<input webkitdirectory>` or drag-and-drop directory. */
  files: File[];
  backend?: RuntimeBackendPreference;
  /** URL prefix containing ORT's version-matched .mjs/.wasm files. */
  wasmPaths?: string;
}

export interface GenerateCommand extends RequestMessage {
  type: "generate";
  prompt: string;
  seed: number | string;
  /** Either durationFrames or durationSeconds must be supplied, but not both. */
  durationFrames?: number;
  durationSeconds?: number;
  cfgWeight?: number;
}

export interface CancelCommand extends RequestMessage {
  type: "cancel";
  targetRequestId: string;
}

export interface DisposeCommand extends RequestMessage {
  type: "dispose";
}

export interface GetStatusCommand extends RequestMessage {
  type: "getStatus";
}

export type WorkerCommand =
  | LoadModelPackCommand
  | GenerateCommand
  | CancelCommand
  | DisposeCommand
  | GetStatusCommand;

export interface WorkerReadyEvent {
  type: "workerReady";
  protocolVersion: typeof WORKER_PROTOCOL_VERSION;
}

export interface ProgressEvent extends RequestMessage {
  type: "progress";
  stage: RuntimeProgressStage;
  completed: number;
  total: number;
  message?: string;
}

export interface ModelLoadedEvent extends RequestMessage {
  type: "modelLoaded";
  model: {
    id: string;
    variant: string;
    backend: RuntimeBackend;
    fps: number;
    minFrames: number;
    maxFrames: number;
    generationFrames: number;
  };
}

export interface GenerationResultEvent extends RequestMessage {
  type: "generationResult";
  result: {
    seed: number;
    prompt: string;
    backend: RuntimeBackend;
    fps: number;
    frameCount: number;
    motion: Float32Array;
    motionShape: [1, number, number];
    joints: Float32Array;
    jointsShape: [1, number, number, 3];
    timingsMs: {
      total: number;
      text: number;
      denoising: number;
      decoding: number;
    };
  };
}

export interface CancelledEvent extends RequestMessage {
  type: "cancelled";
  targetRequestId: string;
}

export interface DisposedEvent extends RequestMessage {
  type: "disposed";
}

export interface RuntimeStatusEvent extends RequestMessage {
  type: "status";
  status:
    | { state: "empty" }
    | { state: "loading"; activeRequestId: string }
    | {
        state: "ready" | "generating";
        activeRequestId?: string;
        modelId: string;
        backend: RuntimeBackend;
      };
}

export interface WorkerErrorEvent extends RequestMessage {
  type: "error";
  error: {
    name: string;
    message: string;
    stack?: string;
  };
}

export type WorkerEvent =
  | WorkerReadyEvent
  | ProgressEvent
  | ModelLoadedEvent
  | GenerationResultEvent
  | CancelledEvent
  | DisposedEvent
  | RuntimeStatusEvent
  | WorkerErrorEvent;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requestId(value: unknown): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError("Worker command requestId must be a non-empty string");
  }
  return value;
}

function optionalBackend(value: unknown): RuntimeBackendPreference | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value !== "auto" && value !== "webgpu" && value !== "wasm") {
    throw new TypeError("backend must be 'auto', 'webgpu', or 'wasm'");
  }
  return value;
}

function isFile(value: unknown): value is File {
  if (typeof File !== "undefined") {
    return value instanceof File;
  }
  return (
    isRecord(value) &&
    typeof value.name === "string" &&
    typeof value.size === "number" &&
    typeof value.arrayBuffer === "function"
  );
}

/** Validate a structured-cloned message before it reaches allocation or model code. */
export function parseWorkerCommand(value: unknown): WorkerCommand {
  if (!isRecord(value)) {
    throw new TypeError("Worker command must be an object");
  }
  const id = requestId(value.requestId);
  switch (value.type) {
    case "loadModelPack": {
      if (!Array.isArray(value.files) || !value.files.every(isFile)) {
        throw new TypeError("loadModelPack.files must be an array of File objects");
      }
      if (value.files.length === 0) {
        throw new TypeError("loadModelPack.files must not be empty");
      }
      if (value.wasmPaths !== undefined && typeof value.wasmPaths !== "string") {
        throw new TypeError("loadModelPack.wasmPaths must be a string");
      }
      return {
        type: "loadModelPack",
        requestId: id,
        files: value.files,
        backend: optionalBackend(value.backend),
        wasmPaths: value.wasmPaths,
      };
    }
    case "generate": {
      if (typeof value.prompt !== "string" || value.prompt.trim().length === 0) {
        throw new TypeError("generate.prompt must be a non-empty string");
      }
      if (
        (typeof value.seed !== "string" && typeof value.seed !== "number") ||
        (typeof value.seed === "number" && !Number.isFinite(value.seed))
      ) {
        throw new TypeError("generate.seed must be a finite number or string");
      }
      const frames = value.durationFrames;
      const seconds = value.durationSeconds;
      if ((frames === undefined) === (seconds === undefined)) {
        throw new TypeError(
          "generate requires exactly one of durationFrames or durationSeconds",
        );
      }
      if (
        frames !== undefined &&
        (typeof frames !== "number" || !Number.isSafeInteger(frames) || frames <= 0)
      ) {
        throw new TypeError("generate.durationFrames must be a positive integer");
      }
      if (
        seconds !== undefined &&
        (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0)
      ) {
        throw new TypeError("generate.durationSeconds must be positive");
      }
      if (
        value.cfgWeight !== undefined &&
        (typeof value.cfgWeight !== "number" ||
          !Number.isFinite(value.cfgWeight) ||
          value.cfgWeight <= 0)
      ) {
        throw new TypeError("generate.cfgWeight must be positive");
      }
      return {
        type: "generate",
        requestId: id,
        prompt: value.prompt,
        seed: value.seed as number | string,
        durationFrames: frames as number | undefined,
        durationSeconds: seconds as number | undefined,
        cfgWeight: value.cfgWeight as number | undefined,
      };
    }
    case "cancel":
      return {
        type: "cancel",
        requestId: id,
        targetRequestId: requestId(value.targetRequestId),
      };
    case "dispose":
      return { type: "dispose", requestId: id };
    case "getStatus":
      return { type: "getStatus", requestId: id };
    default:
      throw new TypeError(`Unknown worker command type ${String(value.type)}`);
  }
}

export function serializeWorkerError(requestIdValue: string, error: unknown): WorkerErrorEvent {
  if (error instanceof Error) {
    return {
      type: "error",
      requestId: requestIdValue,
      error: {
        name: error.name,
        message: error.message,
        ...(error.stack === undefined ? {} : { stack: error.stack }),
      },
    };
  }
  return {
    type: "error",
    requestId: requestIdValue,
    error: {
      name: "Error",
      message: String(error),
    },
  };
}

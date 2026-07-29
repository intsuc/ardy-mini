// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type {
  MotionConstraint,
  MotionConstraintKind,
} from "./constraints";
import type {
  RuntimeContinuationState,
  RuntimeGenerationChunk,
  RuntimeGenerationResult,
} from "./engine";
import type { PortableRandomState } from "./random";
import type { BrowserModelPackManifest } from "./manifest";

export const WORKER_PROTOCOL_VERSION = 2;

export const MAX_WORKER_REQUEST_ID_LENGTH = 256;
export const MAX_WORKER_PROMPT_LENGTH = 4_096;
export const MAX_WORKER_SEED_LENGTH = 512;
export const MAX_WORKER_FRAME_COUNT = 1_000_000;
export const MAX_WORKER_CONSTRAINTS = 10_000;
export const MAX_WORKER_CONSTRAINT_FEATURES = 65_536;
export const MAX_WORKER_CONTINUATION_VALUES = 100_000_000;

const MAX_MODEL_PACK_FILES = 10_000;
const MAX_MODEL_PACK_BYTES = 8 * 1024 * 1024 * 1024;
const MAX_WASM_PATH_LENGTH = 4_096;
const MAX_CFG_WEIGHT = 100;
const MAX_DURATION_SECONDS = 3_600;
const UINT32_RANGE = 0x1_0000_0000;

const CONSTRAINT_KINDS = new Set<MotionConstraintKind>([
  "root",
  "full-body",
  "left-hand",
  "right-hand",
  "left-foot",
  "right-foot",
]);

export type RuntimeBackendPreference = "auto" | "webgpu" | "wasm";
export type RuntimeBackend = "webgpu" | "wasm";
export type GenerationMode = "replace" | "append" | "branch";

export type RuntimeProgressStage =
  | "reading-pack"
  | "hashing-pack"
  | "loading-tokenizer"
  | "loading-sessions"
  | "encoding-text"
  | "denoising"
  | "decoding";

export interface RuntimeCapabilities {
  streamingChunks: boolean;
  sessionContinuation: boolean;
  branching: boolean;
  constraints: boolean;
  futureConstraints: boolean;
  richMotionOutputs: boolean;
  motionCorrection: boolean;
}

export type HybridContinuation = RuntimeContinuationState;
export type GenerationChunkPayload = RuntimeGenerationChunk;
export type GenerationResultPayload = RuntimeGenerationResult;

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
  /**
   * Omitted by protocol-v1 clients. The worker treats an omitted mode as
   * "replace" and emits the legacy generationResult event after completion.
   */
  mode?: GenerationMode;
  branchFrame?: number;
  prompt: string;
  seed: number | string;
  /** Either durationFrames or durationSeconds must be supplied, but not both. */
  durationFrames?: number;
  durationSeconds?: number;
  /** Protocol-v1 alias for textCfgWeight. */
  cfgWeight?: number;
  textCfgWeight?: number;
  constraintCfgWeight?: number;
  historyFrames?: number;
  futureFrames?: number;
  initialTranslation?: Float32Array;
  initialHeading?: number;
  constraints?: MotionConstraint[];
}

export interface ResetSessionCommand extends RequestMessage {
  type: "resetSession";
  seed: number | string;
  initialTranslation?: Float32Array;
  initialHeading?: number;
}

export interface RestoreContinuationCommand extends RequestMessage {
  type: "restoreContinuation";
  continuation: HybridContinuation;
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
  | ResetSessionCommand
  | RestoreContinuationCommand
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
    constraintMaxFrames: number;
    capabilities: RuntimeCapabilities;
    /** Validated conditioning/skeleton metadata needed by the browser editor. */
    manifest: BrowserModelPackManifest;
  };
}

export interface GenerationChunkEvent extends RequestMessage {
  type: "generationChunk";
  mode: GenerationMode;
  chunk: GenerationChunkPayload;
}

export interface GenerationCompleteEvent extends RequestMessage {
  type: "generationComplete";
  mode: GenerationMode;
  generatedFrameCount: number;
  sessionFrameCount: number;
  result: GenerationResultPayload;
}

/** Protocol-v1 terminal event, emitted only when GenerateCommand.mode is omitted. */
export interface GenerationResultEvent extends RequestMessage {
  type: "generationResult";
  result: GenerationResultPayload;
}

export interface SessionResetEvent extends RequestMessage {
  type: "sessionReset";
  seed: number;
  frameCount: 0;
}

export interface ContinuationRestoredEvent extends RequestMessage {
  type: "continuationRestored";
  frameCount: number;
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
        generatedFrameCount: number;
        capabilities: RuntimeCapabilities;
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
  | GenerationChunkEvent
  | GenerationCompleteEvent
  | GenerationResultEvent
  | SessionResetEvent
  | ContinuationRestoredEvent
  | CancelledEvent
  | DisposedEvent
  | RuntimeStatusEvent
  | WorkerErrorEvent;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function boundedString(
  value: unknown,
  label: string,
  maximumLength: number,
): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0 ||
    value.length > maximumLength
  ) {
    throw new TypeError(
      `${label} must be a non-empty string of at most ${maximumLength} characters`,
    );
  }
  return value;
}

function requestId(value: unknown): string {
  return boundedString(
    value,
    "Worker command requestId",
    MAX_WORKER_REQUEST_ID_LENGTH,
  );
}

function seed(value: unknown, label: string): number | string {
  if (typeof value === "string") {
    return boundedString(value, label, MAX_WORKER_SEED_LENGTH);
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number or non-empty string`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function boundedInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum = MAX_WORKER_FRAME_COUNT,
): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new RangeError(
      `${label} must be an integer in the range ${minimum}–${maximum}`,
    );
  }
  return value;
}

function optionalFrameCount(value: unknown, label: string): number | undefined {
  return value === undefined ? undefined : boundedInteger(value, label, 0);
}

function cfgWeight(value: unknown, label: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  const weight = finiteNumber(value, label);
  if (weight < 0 || weight > MAX_CFG_WEIGHT) {
    throw new RangeError(`${label} must be between 0 and ${MAX_CFG_WEIGHT}`);
  }
  return weight;
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
    Number.isFinite(value.size) &&
    value.size >= 0 &&
    typeof value.arrayBuffer === "function"
  );
}

function initialTranslation(
  value: unknown,
  label: string,
): Float32Array | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new TypeError(`${label} must contain exactly three finite numbers`);
  }
  const components = Array.from(value as ArrayLike<unknown>);
  if (components.length !== 3) {
    throw new RangeError(`${label} must contain exactly three finite numbers`);
  }
  return Float32Array.of(
    finiteNumber(components[0], `${label}[0]`),
    finiteNumber(components[1], `${label}[1]`),
    finiteNumber(components[2], `${label}[2]`),
  );
}

function finiteFloat32Array(
  value: unknown,
  label: string,
  maximumLength: number,
  allowEmpty = false,
): Float32Array {
  if (!(value instanceof Float32Array)) {
    throw new TypeError(`${label} must be a Float32Array`);
  }
  if (
    (!allowEmpty && value.length === 0) ||
    value.length > maximumLength
  ) {
    throw new RangeError(
      `${label} must contain ${allowEmpty ? "at most" : "1–"} ${maximumLength} values`,
    );
  }
  const result = new Float32Array(value);
  for (let index = 0; index < result.length; index += 1) {
    if (!Number.isFinite(result[index])) {
      throw new RangeError(`${label}[${index}] must be finite`);
    }
  }
  return result;
}

function parseConstraint(value: unknown, index: number): MotionConstraint {
  const label = `generate.constraints[${index}]`;
  if (!isRecord(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  const id = boundedString(value.id, `${label}.id`, 256).trim();
  if (
    typeof value.kind !== "string" ||
    !CONSTRAINT_KINDS.has(value.kind as MotionConstraintKind)
  ) {
    throw new TypeError(`${label}.kind is not supported`);
  }
  const frame = boundedInteger(value.frame, `${label}.frame`, 0);
  const endFrame =
    value.endFrame === undefined
      ? undefined
      : boundedInteger(value.endFrame, `${label}.endFrame`, frame);
  const values = finiteFloat32Array(
    value.values,
    `${label}.values`,
    MAX_WORKER_CONSTRAINT_FEATURES,
  );
  const mask = finiteFloat32Array(
    value.mask,
    `${label}.mask`,
    MAX_WORKER_CONSTRAINT_FEATURES,
  );
  if (values.length !== mask.length) {
    throw new RangeError(`${label}.values and mask must have equal lengths`);
  }
  let hasObservedFeature = false;
  for (let feature = 0; feature < mask.length; feature += 1) {
    if (mask[feature] < 0 || mask[feature] > 1) {
      throw new RangeError(`${label}.mask[${feature}] must be between 0 and 1`);
    }
    hasObservedFeature ||= mask[feature] > 0.5;
  }
  if (!hasObservedFeature) {
    throw new RangeError(`${label}.mask must observe at least one feature`);
  }
  return {
    id,
    kind: value.kind as MotionConstraintKind,
    frame,
    ...(endFrame === undefined ? {} : { endFrame }),
    values,
    mask,
  };
}

function constraints(value: unknown): MotionConstraint[] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new TypeError("generate.constraints must be an array");
  }
  if (value.length > MAX_WORKER_CONSTRAINTS) {
    throw new RangeError(
      `generate.constraints must contain at most ${MAX_WORKER_CONSTRAINTS} entries`,
    );
  }
  const result = value.map(parseConstraint);
  if (new Set(result.map((constraint) => constraint.id)).size !== result.length) {
    throw new RangeError("generate.constraints ids must be unique");
  }
  return result;
}

function uint32(value: unknown, label: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value >= UINT32_RANGE
  ) {
    throw new RangeError(`${label} must be an unsigned 32-bit integer`);
  }
  return value;
}

function randomState(value: unknown): PortableRandomState {
  if (!isRecord(value)) {
    throw new TypeError("continuation.random must be an object");
  }
  const spareNormal =
    value.spareNormal === undefined
      ? undefined
      : finiteNumber(value.spareNormal, "continuation.random.spareNormal");
  return {
    seed: uint32(value.seed, "continuation.random.seed"),
    state: uint32(value.state, "continuation.random.state"),
    ...(spareNormal === undefined ? {} : { spareNormal }),
  };
}

function continuation(value: unknown): HybridContinuation {
  if (!isRecord(value)) {
    throw new TypeError("restoreContinuation.continuation must be an object");
  }
  const hybridDim = boundedInteger(
    value.hybridDim,
    "continuation.hybridDim",
    1,
    MAX_WORKER_CONSTRAINT_FEATURES,
  );
  const hybridTokens = finiteFloat32Array(
    value.hybridTokens,
    "continuation.hybridTokens",
    MAX_WORKER_CONTINUATION_VALUES,
    true,
  );
  if (hybridTokens.length % hybridDim !== 0) {
    throw new RangeError(
      "continuation.hybridTokens length must be divisible by hybridDim",
    );
  }
  const transform = initialTranslation(
    value.initialTranslation,
    "continuation.initialTranslation",
  );
  if (transform === undefined) {
    throw new TypeError(
      "continuation.initialTranslation must contain exactly three finite numbers",
    );
  }
  return {
    frameCount: boundedInteger(
      value.frameCount,
      "continuation.frameCount",
      0,
    ),
    hybridTokens,
    hybridDim,
    random: randomState(value.random),
    initialTranslation: [transform[0], transform[1], transform[2]],
    initialHeading: finiteNumber(
      value.initialHeading,
      "continuation.initialHeading",
    ),
  };
}

function parseGenerate(
  value: Record<string, unknown>,
  id: string,
): GenerateCommand {
  const prompt = boundedString(
    value.prompt,
    "generate.prompt",
    MAX_WORKER_PROMPT_LENGTH,
  );
  const parsedSeed = seed(value.seed, "generate.seed");
  const frames = value.durationFrames;
  const seconds = value.durationSeconds;
  if ((frames === undefined) === (seconds === undefined)) {
    throw new TypeError(
      "generate requires exactly one of durationFrames or durationSeconds",
    );
  }
  const durationFrames =
    frames === undefined
      ? undefined
      : boundedInteger(
          frames,
          "generate.durationFrames",
          1,
          MAX_WORKER_FRAME_COUNT,
        );
  let durationSeconds: number | undefined;
  if (seconds !== undefined) {
    durationSeconds = finiteNumber(seconds, "generate.durationSeconds");
    if (durationSeconds <= 0 || durationSeconds > MAX_DURATION_SECONDS) {
      throw new RangeError(
        `generate.durationSeconds must be in the range 0–${MAX_DURATION_SECONDS}`,
      );
    }
  }

  let mode: GenerationMode | undefined;
  if (value.mode !== undefined) {
    if (
      value.mode !== "replace" &&
      value.mode !== "append" &&
      value.mode !== "branch"
    ) {
      throw new TypeError(
        "generate.mode must be 'replace', 'append', or 'branch'",
      );
    }
    mode = value.mode;
  }
  const effectiveMode = mode ?? "replace";
  let branchFrame: number | undefined;
  if (effectiveMode === "branch") {
    branchFrame = boundedInteger(
      value.branchFrame,
      "generate.branchFrame",
      0,
    );
  } else if (value.branchFrame !== undefined) {
    throw new TypeError("generate.branchFrame is only valid in branch mode");
  }

  if (value.cfgWeight !== undefined && value.textCfgWeight !== undefined) {
    throw new TypeError(
      "generate.cfgWeight and textCfgWeight must not both be supplied",
    );
  }
  const legacyCfgWeight = cfgWeight(
    value.cfgWeight,
    "generate.cfgWeight",
  );
  const parsedTextCfgWeight = cfgWeight(
    value.textCfgWeight,
    "generate.textCfgWeight",
  );
  const parsedConstraintCfgWeight = cfgWeight(
    value.constraintCfgWeight,
    "generate.constraintCfgWeight",
  );
  const parsedTranslation = initialTranslation(
    value.initialTranslation,
    "generate.initialTranslation",
  );
  const parsedHeading =
    value.initialHeading === undefined
      ? undefined
      : finiteNumber(value.initialHeading, "generate.initialHeading");
  if (
    effectiveMode !== "replace" &&
    (parsedTranslation !== undefined || parsedHeading !== undefined)
  ) {
    throw new TypeError(
      "generate initial transform is only valid in replace mode",
    );
  }

  return {
    type: "generate",
    requestId: id,
    ...(mode === undefined ? {} : { mode }),
    ...(branchFrame === undefined ? {} : { branchFrame }),
    prompt,
    seed: parsedSeed,
    ...(durationFrames === undefined ? {} : { durationFrames }),
    ...(durationSeconds === undefined ? {} : { durationSeconds }),
    ...(legacyCfgWeight === undefined ? {} : { cfgWeight: legacyCfgWeight }),
    ...(parsedTextCfgWeight === undefined
      ? {}
      : { textCfgWeight: parsedTextCfgWeight }),
    ...(parsedConstraintCfgWeight === undefined
      ? {}
      : { constraintCfgWeight: parsedConstraintCfgWeight }),
    ...(value.historyFrames === undefined
      ? {}
      : {
          historyFrames: optionalFrameCount(
            value.historyFrames,
            "generate.historyFrames",
          )!,
        }),
    ...(value.futureFrames === undefined
      ? {}
      : {
          futureFrames: optionalFrameCount(
            value.futureFrames,
            "generate.futureFrames",
          )!,
        }),
    ...(parsedTranslation === undefined
      ? {}
      : { initialTranslation: parsedTranslation }),
    ...(parsedHeading === undefined ? {} : { initialHeading: parsedHeading }),
    ...(value.constraints === undefined
      ? {}
      : { constraints: constraints(value.constraints)! }),
  };
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
        throw new TypeError(
          "loadModelPack.files must be an array of File objects",
        );
      }
      if (
        value.files.length === 0 ||
        value.files.length > MAX_MODEL_PACK_FILES
      ) {
        throw new RangeError(
          `loadModelPack.files must contain 1–${MAX_MODEL_PACK_FILES} files`,
        );
      }
      const totalBytes = value.files.reduce(
        (total, file) => total + file.size,
        0,
      );
      if (!Number.isSafeInteger(totalBytes) || totalBytes > MAX_MODEL_PACK_BYTES) {
        throw new RangeError(
          `loadModelPack files must total at most ${MAX_MODEL_PACK_BYTES} bytes`,
        );
      }
      if (
        value.wasmPaths !== undefined &&
        (typeof value.wasmPaths !== "string" ||
          value.wasmPaths.length > MAX_WASM_PATH_LENGTH)
      ) {
        throw new TypeError(
          `loadModelPack.wasmPaths must be a string of at most ${MAX_WASM_PATH_LENGTH} characters`,
        );
      }
      return {
        type: "loadModelPack",
        requestId: id,
        files: [...value.files],
        backend: optionalBackend(value.backend),
        wasmPaths: value.wasmPaths,
      };
    }
    case "generate":
      return parseGenerate(value, id);
    case "resetSession": {
      const parsedTranslation = initialTranslation(
        value.initialTranslation,
        "resetSession.initialTranslation",
      );
      return {
        type: "resetSession",
        requestId: id,
        seed: seed(value.seed, "resetSession.seed"),
        ...(parsedTranslation === undefined
          ? {}
          : { initialTranslation: parsedTranslation }),
        ...(value.initialHeading === undefined
          ? {}
          : {
              initialHeading: finiteNumber(
                value.initialHeading,
                "resetSession.initialHeading",
              ),
            }),
      };
    }
    case "restoreContinuation":
      return {
        type: "restoreContinuation",
        requestId: id,
        continuation: continuation(value.continuation),
      };
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

export function serializeWorkerError(
  requestIdValue: string,
  error: unknown,
): WorkerErrorEvent {
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

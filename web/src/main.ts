// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  type RuntimeContinuationState,
  type RuntimeGenerationChunk,
  type RuntimeGenerationResult,
} from "./runtime/engine";
import {
  validateModelPackManifest,
  type BrowserModelPackManifest,
} from "./runtime/manifest";
import {
  type GenerationCompleteEvent,
  type GenerationMode,
  type LoadModelPackCommand,
  type ModelLoadedEvent,
  type ProgressEvent,
  type RuntimeBackendPreference,
  type WorkerCommand,
  type WorkerEvent,
} from "./runtime/protocol";
import {
  normalizeStructuredMotion,
  type RotationTrack,
  type StructuredMotionResult,
} from "./motion-data";
import {
  isContinuationModelCompatible,
  type MotionSessionProvenance,
} from "./session-format";
import {
  DEFAULT_EDITOR_STATE,
  type MotionEditorState,
  type ViewerOutputVisibility,
} from "./editor-state";
import {
  SkeletonViewer,
  type PlaybackState,
  type VrmModelInfo,
} from "./viewer";
import { PROMPT_EXAMPLE_EVENT } from "./prompt-examples";

const UINT32_MAX = 0xffff_ffff;
const ALLOWED_DURATIONS = new Set([2, 4, 6, 8, 10]);
const CACHE_ROOT = "ardy-mini-model-cache";
const CACHE_PACK = "active-pack";
const CACHE_INDEX = "index.json";
const PACK_MANIFEST = "manifest.json";
const ORT_WASM_PATH = new URL("ort/", document.baseURI).href;
const DEFAULT_TEXT_CFG_WEIGHT = 3.5;
const DEFAULT_CONSTRAINT_CFG_WEIGHT = 1;
const DEFAULT_HISTORY_FRAMES = 40;
const DEFAULT_FUTURE_CROP_FRAMES = 80;
const DEFAULT_REPLAN_BUFFER_FRAMES = 20;
const DEFAULT_REPLAN_THRESHOLD_FRAMES = 10;

type FileWithRelativePath = File & { readonly webkitRelativePath?: string };

interface CacheIndex {
  schemaVersion: 1;
  files: Array<{ path: string; size: number; lastModified: number; type: string }>;
}

interface FormValues {
  prompt: string;
  durationSeconds: number;
  seed: number;
}

interface FormValidation {
  values?: FormValues;
  promptError?: string;
  seedError?: string;
}

interface DirectoryPickerWindow {
  showDirectoryPicker?: (options?: { mode?: "read" }) => Promise<FileSystemDirectoryHandle>;
}

interface ActiveGeneration {
  id: string;
  mode: GenerationMode;
  background: boolean;
  receivedChunk: boolean;
  resumePlayback: boolean;
}

export function validateGenerationForm(
  promptValue: string,
  durationValue: string,
  seedValue: string,
): FormValidation {
  const prompt = promptValue.trim();
  const durationSeconds = Number(durationValue);
  const seed = Number(seedValue);
  const validation: FormValidation = {};

  if (!prompt) {
    validation.promptError = "Describe the motion you want to generate.";
  } else if (prompt.length > 280) {
    validation.promptError = "Keep the prompt to 280 characters or fewer.";
  }

  if (!Number.isInteger(seed) || seed < 0 || seed > UINT32_MAX) {
    validation.seedError = `Enter a whole-number seed from 0 to ${UINT32_MAX}.`;
  }

  if (!ALLOWED_DURATIONS.has(durationSeconds)) {
    validation.promptError ??= "Choose a duration from 2 to 10 seconds in 2-second steps.";
  }

  if (!validation.promptError && !validation.seedError) {
    validation.values = { prompt, durationSeconds, seed };
  }
  return validation;
}

export function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

function canonicalPackPath(file: FileWithRelativePath): string {
  return (file.webkitRelativePath || file.name)
    .replaceAll("\\", "/")
    .replace(/^\.\/+/, "");
}

/**
 * Strip the directory selected by `<input webkitdirectory>` while retaining all
 * paths below the pack's manifest.
 */
export function canonicalizePackFiles(input: readonly File[]): File[] {
  if (input.length === 0) {
    throw new Error("The selected model-pack folder is empty.");
  }
  const entries = input.map((file) => ({ file, path: canonicalPackPath(file) }));
  const manifestPaths = entries.filter(
    ({ path }) => path === PACK_MANIFEST || path.endsWith(`/${PACK_MANIFEST}`),
  );
  if (manifestPaths.length !== 1) {
    throw new Error(
      manifestPaths.length === 0
        ? "No manifest.json was found in the selected folder."
        : "The selected folder contains more than one manifest.json.",
    );
  }

  const manifestPath = manifestPaths[0].path;
  const prefix = manifestPath.slice(0, manifestPath.length - PACK_MANIFEST.length);
  const normalized = entries.map(({ file, path }) => {
    if (prefix && !path.startsWith(prefix)) {
      throw new Error(
        "Every model-pack file must be inside the folder containing manifest.json.",
      );
    }
    const relativePath = prefix ? path.slice(prefix.length) : path;
    const parts = relativePath.split("/");
    if (
      !relativePath ||
      relativePath.startsWith("/") ||
      relativePath.endsWith("/") ||
      parts.some((part) => !part || part === "." || part === "..")
    ) {
      throw new Error(`Unsafe model-pack path: ${relativePath || path}`);
    }
    return new File([file], relativePath, {
      type: file.type,
      lastModified: file.lastModified,
    });
  });

  normalized.sort((left, right) => {
    if (left.name === PACK_MANIFEST) return -1;
    if (right.name === PACK_MANIFEST) return 1;
    return left.name.localeCompare(right.name);
  });
  return normalized;
}

export function shouldAutoplayMotion(reducedMotion: boolean): boolean {
  return !reducedMotion;
}

/**
 * Only the first visual update of a replace operation starts a new motion
 * presentation. Append, branch, and later streamed chunks must keep playback
 * and VRM secondary-motion continuity.
 */
export function shouldResetMotionPresentation(
  mode: GenerationMode,
  initialVisualUpdate: boolean,
): boolean {
  return mode === "replace" && initialVisualUpdate;
}

export function canAttemptGeneration(
  promptValue: string,
  runtimeReady: boolean,
  modelReady: boolean,
  modelLoading: boolean,
  generating: boolean,
): boolean {
  return (
    promptValue.trim().length > 0 &&
    runtimeReady &&
    modelReady &&
    !modelLoading &&
    !generating
  );
}

export function canContinueGeneration(
  modelReady: boolean,
  generating: boolean,
  hasMotion: boolean,
  hasContinuation: boolean,
): boolean {
  return (
    modelReady &&
    !generating &&
    hasMotion &&
    hasContinuation
  );
}

function requiredElement<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required UI element #${id}.`);
  return element as unknown as T;
}

function requiredDescendant<T extends Element>(
  root: Element,
  selector: string,
  label: string,
): T {
  const element = root.querySelector(selector);
  if (!element) throw new Error(`Missing required UI element ${label}.`);
  return element as T;
}

function requestId(prefix: string): string {
  const suffix =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random()}`;
  return `${prefix}-${suffix}`;
}

function humanizeStage(stage: string): string {
  const labels: Record<string, string> = {
    "reading-pack": "Reading model pack",
    "hashing-pack": "Verifying model files",
    "loading-tokenizer": "Loading tokenizer",
    "loading-sessions": "Preparing inference sessions",
    "encoding-text": "Encoding motion prompt",
    "preparing-constraints": "Preparing constraints",
    denoising: "Generating motion",
    decoding: "Decoding skeleton",
    postprocessing: "Correcting motion",
  };
  return labels[stage] ?? stage;
}

function progressFraction(event: ProgressEvent): number {
  if (!(event.total > 0)) return 0;
  return Math.max(0, Math.min(1, event.completed / event.total));
}

function generationProgress(event: ProgressEvent): number {
  const local = progressFraction(event);
  const stage = event.stage as string;
  if (stage === "encoding-text") return local * 0.06;
  if (stage === "preparing-constraints") return 0.06 + local * 0.04;
  if (stage === "denoising") return 0.1 + local * 0.78;
  if (stage === "decoding") return 0.88 + local * 0.1;
  if (stage === "postprocessing") return 0.98 + local * 0.02;
  return local;
}

function modelLoadProgress(event: ProgressEvent): number {
  const local = progressFraction(event);
  const ranges: Partial<
    Record<ProgressEvent["stage"], readonly [number, number]>
  > = {
    "reading-pack": [0, 0.18],
    "hashing-pack": [0.18, 0.38],
    "loading-tokenizer": [0.38, 0.48],
    "loading-sessions": [0.48, 1],
  };
  const range = ranges[event.stage];
  return range ? range[0] + local * (range[1] - range[0]) : local;
}

async function collectDirectoryFiles(
  directory: FileSystemDirectoryHandle,
  prefix = "",
): Promise<File[]> {
  const files: File[] = [];
  for await (const [name, handle] of directory.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === "directory") {
      files.push(
        ...(await collectDirectoryFiles(
          handle as FileSystemDirectoryHandle,
          path,
        )),
      );
    } else {
      const source = await (handle as FileSystemFileHandle).getFile();
      files.push(
        new File([source], path, {
          type: source.type,
          lastModified: source.lastModified,
        }),
      );
    }
  }
  return files;
}

function supportsPersistentPackCache(): boolean {
  return Boolean(navigator.storage && "getDirectory" in navigator.storage);
}

async function getCacheRoot(
  create: boolean,
): Promise<FileSystemDirectoryHandle | null> {
  if (!supportsPersistentPackCache()) return null;
  const opfs = await navigator.storage.getDirectory();
  try {
    return await opfs.getDirectoryHandle(CACHE_ROOT, { create });
  } catch (error) {
    if (
      !create &&
      error instanceof DOMException &&
      error.name === "NotFoundError"
    ) {
      return null;
    }
    throw error;
  }
}

async function ensureDirectory(
  root: FileSystemDirectoryHandle,
  parts: readonly string[],
): Promise<FileSystemDirectoryHandle> {
  let directory = root;
  for (const part of parts) {
    directory = await directory.getDirectoryHandle(part, { create: true });
  }
  return directory;
}

async function writeTextFile(
  directory: FileSystemDirectoryHandle,
  name: string,
  text: string,
): Promise<void> {
  const handle = await directory.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  await writable.write(text);
  await writable.close();
}

async function cacheModelPack(
  files: readonly File[],
  onProgress: (completedBytes: number, totalBytes: number) => void,
): Promise<void> {
  const root = await getCacheRoot(true);
  if (!root) throw new Error("Persistent browser storage is unavailable.");

  const totalBytes = files.reduce((total, file) => total + file.size, 0);
  const estimate = await navigator.storage.estimate();
  const available = (estimate.quota ?? 0) - (estimate.usage ?? 0);
  if (estimate.quota !== undefined && available < totalBytes * 1.05) {
    throw new Error(
      `The model pack needs ${formatBytes(totalBytes)}, but only about ${formatBytes(Math.max(0, available))} is available.`,
    );
  }

  await navigator.storage.persist?.();
  try {
    await root.removeEntry(CACHE_PACK, { recursive: true });
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) {
      throw error;
    }
  }
  const packDirectory = await root.getDirectoryHandle(CACHE_PACK, {
    create: true,
  });
  let completedBytes = 0;
  const index: CacheIndex = { schemaVersion: 1, files: [] };

  for (const file of files) {
    const parts = file.name.split("/");
    const filename = parts.pop();
    if (!filename) throw new Error(`Invalid model-pack path ${file.name}.`);
    const directory = await ensureDirectory(packDirectory, parts);
    const handle = await directory.getFileHandle(filename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(file);
    await writable.close();
    completedBytes += file.size;
    index.files.push({
      path: file.name,
      size: file.size,
      lastModified: file.lastModified,
      type: file.type,
    });
    onProgress(completedBytes, totalBytes);
  }

  await writeTextFile(root, CACHE_INDEX, JSON.stringify(index));
}

async function readCachedModelPack(): Promise<File[] | null> {
  const root = await getCacheRoot(false);
  if (!root) return null;
  try {
    const indexHandle = await root.getFileHandle(CACHE_INDEX);
    const index = JSON.parse(
      await (await indexHandle.getFile()).text(),
    ) as CacheIndex;
    if (
      index.schemaVersion !== 1 ||
      !Array.isArray(index.files) ||
      index.files.length === 0
    ) {
      return null;
    }
    const packDirectory = await root.getDirectoryHandle(CACHE_PACK);
    const files: File[] = [];
    for (const entry of index.files) {
      if (!entry || typeof entry.path !== "string") return null;
      const parts = entry.path.split("/");
      const filename = parts.pop();
      if (
        !filename ||
        parts.some((part) => !part || part === "." || part === "..")
      ) {
        return null;
      }
      let directory = packDirectory;
      for (const part of parts) {
        directory = await directory.getDirectoryHandle(part);
      }
      const stored = await (
        await directory.getFileHandle(filename)
      ).getFile();
      if (stored.size !== entry.size) return null;
      files.push(
        new File([stored], entry.path, {
          type: entry.type,
          lastModified: entry.lastModified,
        }),
      );
    }
    return canonicalizePackFiles(files);
  } catch {
    return null;
  }
}

async function removeCachedModelPack(): Promise<void> {
  const opfs = supportsPersistentPackCache()
    ? await navigator.storage.getDirectory()
    : null;
  if (!opfs) return;
  try {
    await opfs.removeEntry(CACHE_ROOT, { recursive: true });
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) {
      throw error;
    }
  }
}

async function manifestFromFiles(
  files: readonly File[],
): Promise<BrowserModelPackManifest> {
  const file = files.find((candidate) => candidate.name === PACK_MANIFEST);
  if (!file) throw new Error("The canonical model pack has no manifest.");
  let value: unknown;
  try {
    value = JSON.parse(await file.text()) as unknown;
  } catch (error) {
    throw new Error(
      `manifest.json is invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return validateModelPackManifest(value);
}

function sameSkeleton(
  left: StructuredMotionResult["skeleton"],
  right: StructuredMotionResult["skeleton"],
): boolean {
  return (
    left.parents.length === right.parents.length &&
    left.parents.every((parent, index) => parent === right.parents[index])
  );
}

function mergeFloatTrack(
  previous: Float32Array | undefined,
  incoming: Float32Array | undefined,
  prefixFrames: number,
  previousStride: number,
  incomingStride: number,
): Float32Array | undefined {
  if (!incoming || (prefixFrames > 0 && !previous)) return undefined;
  if (previousStride !== incomingStride) {
    throw new RangeError("Motion tracks use incompatible frame strides.");
  }
  const result = new Float32Array(
    prefixFrames * previousStride + incoming.length,
  );
  if (prefixFrames > 0 && previous) {
    result.set(previous.subarray(0, prefixFrames * previousStride));
  }
  result.set(incoming, prefixFrames * previousStride);
  return result;
}

function mergeByteTrack(
  previous: Uint8Array | undefined,
  incoming: Uint8Array | undefined,
  prefixFrames: number,
  previousStride: number,
  incomingStride: number,
): Uint8Array | undefined {
  if (!incoming || (prefixFrames > 0 && !previous)) return undefined;
  if (previousStride !== incomingStride) {
    throw new RangeError("Motion contact tracks use incompatible frame strides.");
  }
  const result = new Uint8Array(
    prefixFrames * previousStride + incoming.length,
  );
  if (prefixFrames > 0 && previous) {
    result.set(previous.subarray(0, prefixFrames * previousStride));
  }
  result.set(incoming, prefixFrames * previousStride);
  return result;
}

function mergeRotationTrack(
  previous: RotationTrack | undefined,
  incoming: RotationTrack | undefined,
  prefixFrames: number,
): RotationTrack | undefined {
  if (!incoming || (prefixFrames > 0 && !previous)) return undefined;
  const previousStride = previous
    ? previous.shape[1] * previous.shape[2]
    : incoming.shape[1] * incoming.shape[2];
  const incomingStride = incoming.shape[1] * incoming.shape[2];
  if (
    previous &&
    (previous.format !== incoming.format ||
      previous.shape[1] !== incoming.shape[1] ||
      previous.shape[2] !== incoming.shape[2])
  ) {
    throw new RangeError("Motion rotation tracks are incompatible.");
  }
  const values = mergeFloatTrack(
    previous?.values,
    incoming.values,
    prefixFrames,
    previousStride,
    incomingStride,
  )!;
  return {
    values,
    shape: [
      prefixFrames + incoming.shape[0],
      incoming.shape[1],
      incoming.shape[2],
    ],
    format: incoming.format,
  };
}

function mergeMotion(
  previous: StructuredMotionResult | null,
  incoming: StructuredMotionResult,
  startFrame: number,
): StructuredMotionResult {
  if (!Number.isSafeInteger(startFrame) || startFrame < 0) {
    throw new RangeError("Chunk start frame must be a non-negative integer.");
  }
  if (!previous) {
    if (startFrame !== 0) {
      throw new RangeError("The first generation chunk must start at frame zero.");
    }
    return incoming;
  }
  if (startFrame > previous.frameCount) {
    throw new RangeError(
      `Chunk starts at ${startFrame}, after the ${previous.frameCount}-frame session.`,
    );
  }
  if (previous.fps !== incoming.fps || !sameSkeleton(previous.skeleton, incoming.skeleton)) {
    throw new RangeError("Generated chunks use incompatible motion metadata.");
  }

  const prefixFrames = startFrame;
  const jointCount = incoming.positionsShape[1];
  const totalFrames = prefixFrames + incoming.frameCount;
  const positions = mergeFloatTrack(
    previous.positions,
    incoming.positions,
    prefixFrames,
    previous.positionsShape[1] * 3,
    jointCount * 3,
  )!;
  const normalizedStride =
    incoming.normalizedMotionShape?.[1] ??
    previous.normalizedMotionShape?.[1] ??
    0;
  const normalizedMotion =
    normalizedStride > 0
      ? mergeFloatTrack(
          previous.normalizedMotion,
          incoming.normalizedMotion,
          prefixFrames,
          previous.normalizedMotionShape?.[1] ?? normalizedStride,
          incoming.normalizedMotionShape?.[1] ?? normalizedStride,
        )
      : undefined;
  const rootStride =
    incoming.rootsShape?.[1] ?? previous.rootsShape?.[1] ?? 0;
  const roots =
    rootStride > 0
      ? mergeFloatTrack(
          previous.roots,
          incoming.roots,
          prefixFrames,
          previous.rootsShape?.[1] ?? rootStride,
          incoming.rootsShape?.[1] ?? rootStride,
        )
      : undefined;
  const contactStride =
    incoming.contactsShape?.[1] ?? previous.contactsShape?.[1] ?? 0;
  const contacts =
    contactStride > 0
      ? mergeByteTrack(
          previous.contacts,
          incoming.contacts,
          prefixFrames,
          previous.contactsShape?.[1] ?? contactStride,
          incoming.contactsShape?.[1] ?? contactStride,
        )
      : undefined;

  return {
    skeleton: incoming.skeleton,
    positions,
    positionsShape: [totalFrames, jointCount, 3],
    frameCount: totalFrames,
    fps: incoming.fps,
    normalizedMotion,
    normalizedMotionShape: normalizedMotion
      ? [totalFrames, normalizedStride]
      : undefined,
    localRotations: mergeRotationTrack(
      previous.localRotations,
      incoming.localRotations,
      prefixFrames,
    ),
    globalRotations: mergeRotationTrack(
      previous.globalRotations,
      incoming.globalRotations,
      prefixFrames,
    ),
    roots,
    rootsShape: roots ? [totalFrames, rootStride] : undefined,
    contacts,
    contactsShape: contacts ? [totalFrames, contactStride] : undefined,
  };
}

function motionFromRuntime(
  payload: RuntimeGenerationChunk | RuntimeGenerationResult,
  manifest: BrowserModelPackManifest | null,
): StructuredMotionResult {
  return normalizeStructuredMotion(
    {
      positions: payload.joints,
      frameCount: payload.frameCount,
      fps: payload.fps,
      normalizedMotion: payload.motion,
      localRotations: payload.localRotations,
      localRotationFormat: payload.localRotations
        ? "matrix3x3-row-major"
        : undefined,
      globalRotations: payload.globalRotations,
      globalRotationFormat: payload.globalRotations
        ? "matrix3x3-row-major"
        : undefined,
      roots: payload.rootPositions,
      contacts: payload.footContacts,
    },
    {
      skeleton: manifest?.skeleton,
      defaultFps: payload.fps,
    },
  );
}

function cloneEditorState(state: MotionEditorState): MotionEditorState {
  return {
    initialTransform: {
      position: [...state.initialTransform.position] as [
        number,
        number,
        number,
      ],
      headingRadians: state.initialTransform.headingRadians,
    },
    waypoints: state.waypoints.map((waypoint) => ({
      ...waypoint,
      position: [...waypoint.position] as [number, number, number],
    })),
    constraints: state.constraints.map((constraint) => ({
      ...constraint,
      position: constraint.position
        ? ([...constraint.position] as [number, number, number])
        : undefined,
      orientation: constraint.orientation
        ? ([
            ...constraint.orientation,
          ] as [number, number, number, number])
        : undefined,
    })),
    outputVisibility: { ...state.outputVisibility },
  };
}

export function bootstrap(): () => void {
  if (!document.getElementById("app")) return () => {};
  const lifecycle = new AbortController();
  let disposed = false;

  const form = requiredElement<HTMLFormElement>("generation-form");
  const prompt = requiredElement<HTMLTextAreaElement>("prompt");
  const promptCount = requiredElement<HTMLElement>("prompt-count");
  const promptError = requiredElement<HTMLElement>("prompt-error");
  const duration = requiredElement<HTMLInputElement>("duration");
  const durationOutput = requiredElement<HTMLOutputElement>("duration-output");
  const seed = requiredElement<HTMLInputElement>("seed");
  const seedError = requiredElement<HTMLElement>("seed-error");
  const randomizeSeed = requiredElement<HTMLButtonElement>("randomize-seed");
  const backend = requiredElement<HTMLSelectElement>("backend");
  const backendHelp = requiredElement<HTMLElement>("backend-help");
  const importModel = requiredElement<HTMLButtonElement>("import-model");
  const importModelLabel = requiredElement<HTMLElement>("import-model-label");
  const removeModel = requiredElement<HTMLButtonElement>("remove-model");
  const fileInput = requiredElement<HTMLInputElement>("model-file-input");
  const modelCard = requiredElement<HTMLElement>("model-card");
  const modelTitle = requiredElement<HTMLElement>("model-title");
  const modelDetail = requiredElement<HTMLElement>("model-detail");
  const modelSetupHelp = requiredElement<HTMLElement>("model-setup-help");
  const modelState = requiredElement<HTMLElement>("model-state");
  const modelProgress = requiredElement<HTMLElement>("model-progress");
  const modelProgressbar = requiredElement<HTMLElement>("model-progressbar");
  const modelProgressFill = requiredDescendant<HTMLElement>(
    modelProgressbar,
    '[data-slot="progress-indicator"]',
    "model progress indicator",
  );
  const modelProgressLabel = requiredElement<HTMLElement>("model-progress-label");
  const modelErrorBanner = requiredElement<HTMLElement>("model-error-banner");
  const modelErrorTitle = requiredElement<HTMLElement>("model-error-title");
  const modelErrorMessage = requiredElement<HTMLElement>("model-error-message");
  const dismissModelError =
    requiredElement<HTMLButtonElement>("dismiss-model-error");
  const generate = requiredElement<HTMLButtonElement>("generate");
  const generateSpinner =
    requiredElement<SVGSVGElement>("generate-spinner");
  const generateLabel = requiredElement<HTMLElement>("generate-label");
  const generateHelp = requiredElement<HTMLElement>("generate-help");
  const buttonShortcut = requiredElement<HTMLElement>("button-shortcut");
  const restartGeneration =
    requiredElement<HTMLButtonElement>("restart-generation");
  const restartFromNow =
    requiredElement<HTMLButtonElement>("restart-from-now");
  const applyPrompt = requiredElement<HTMLButtonElement>("apply-prompt");
  const cancelGeneration =
    requiredElement<HTMLButtonElement>("cancel-generation");
  const streamGeneration =
    requiredElement<HTMLInputElement>("stream-generation");
  const targetBuffer = requiredElement<HTMLInputElement>("target-buffer");
  const generationProgressElement =
    requiredElement<HTMLElement>("generation-progress");
  const generationStage = requiredElement<HTMLElement>("generation-stage");
  const generationPercent = requiredElement<HTMLElement>("generation-percent");
  const generationProgressbar =
    requiredElement<HTMLElement>("generation-progressbar");
  const generationProgressFill =
    requiredDescendant<HTMLElement>(
      generationProgressbar,
      '[data-slot="progress-indicator"]',
      "generation progress indicator",
    );
  const errorBanner = requiredElement<HTMLElement>("error-banner");
  const errorTitle = requiredElement<HTMLElement>("error-title");
  const errorMessage = requiredElement<HTMLElement>("error-message");
  const dismissError = requiredElement<HTMLButtonElement>("dismiss-error");
  const canvas = requiredElement<HTMLCanvasElement>("motion-canvas");
  const emptyState = requiredElement<HTMLElement>("empty-state");
  const motionBadge = requiredElement<HTMLElement>("motion-badge");
  const runtimeMetric = requiredElement<HTMLElement>("runtime-metric");
  const runtimeValue = requiredElement<HTMLElement>("runtime-value");
  const playPause = requiredElement<HTMLButtonElement>("play-pause");
  const timeline = requiredElement<HTMLInputElement>("timeline");
  const currentTime = requiredElement<HTMLElement>("current-time");
  const totalTime = requiredElement<HTMLElement>("total-time");
  const playbackSpeed =
    requiredElement<HTMLSelectElement>("playback-speed");
  const loopToggle = requiredElement<HTMLButtonElement>("loop-toggle");
  const resetCamera = requiredElement<HTMLButtonElement>("reset-camera");
  const gpuDot = requiredElement<HTMLElement>("gpu-dot");
  const gpuLabel = requiredElement<HTMLElement>("gpu-label");
  const isolationDot = requiredElement<HTMLElement>("isolation-dot");
  const isolationLabel = requiredElement<HTMLElement>("isolation-label");
  const appStatus = requiredElement<HTMLElement>("app-status");
  const viewportPanel = requiredElement<HTMLElement>("viewport-panel");
  const viewport = requiredElement<HTMLElement>("viewport");
  const modelRuntimeStatus =
    requiredElement<HTMLElement>("model-runtime-status");
  const modelRuntimeDetail =
    requiredElement<HTMLElement>("model-runtime-detail");

  const showSkeleton = requiredElement<HTMLInputElement>("show-skeleton");
  const showContacts = requiredElement<HTMLInputElement>("show-contacts");
  const showOrientations =
    requiredElement<HTMLInputElement>("show-orientations");
  const showTrajectory =
    requiredElement<HTMLInputElement>("show-trajectory");
  const previewSettings =
    requiredElement<HTMLDetailsElement>("preview-settings");
  const vrmCard = requiredElement<HTMLElement>("vrm-card");
  const vrmName = requiredElement<HTMLElement>("vrm-name");
  const vrmDetail = requiredElement<HTMLElement>("vrm-detail");
  const vrmState = requiredElement<HTMLElement>("vrm-state");
  const importVrm = requiredElement<HTMLButtonElement>("import-vrm");
  const importVrmLabel = requiredElement<HTMLElement>("import-vrm-label");
  const vrmFileInput = requiredElement<HTMLInputElement>("vrm-file-input");
  const removeVrm = requiredElement<HTMLButtonElement>("remove-vrm");
  const showVrm = requiredElement<HTMLInputElement>("show-vrm");
  const vrmErrorBanner =
    requiredElement<HTMLElement>("vrm-error-banner");
  const vrmErrorMessage =
    requiredElement<HTMLElement>("vrm-error-message");
  const dismissVrmError =
    requiredElement<HTMLButtonElement>("dismiss-vrm-error");

  fileInput.setAttribute("webkitdirectory", "");
  fileInput.setAttribute("directory", "");
  restartFromNow.disabled = true;

  const hasWebGpu = "gpu" in navigator;
  const webGpuOption =
    backend.querySelector<HTMLOptionElement>('option[value="webgpu"]');
  if (webGpuOption) webGpuOption.disabled = !hasWebGpu;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  buttonShortcut.textContent = /Mac|iPhone|iPad|iPod/.test(navigator.platform)
    ? "⌘ ↵"
    : "Ctrl ↵";
  gpuDot.dataset.state = hasWebGpu ? "available" : "unavailable";
  gpuLabel.textContent = hasWebGpu
    ? "WebGPU available"
    : "WebGPU unavailable";
  isolationDot.dataset.state = crossOriginIsolated
    ? "available"
    : "unavailable";
  isolationLabel.textContent = crossOriginIsolated
    ? "WASM threads ready"
    : "WASM single-thread";

  let errorReturnFocus: HTMLElement | null = null;
  let modelErrorReturnFocus: HTMLElement | null = null;
  let vrmErrorReturnFocus: HTMLElement | null = null;
  let viewer: SkeletonViewer | null = null;
  try {
    viewer = new SkeletonViewer(canvas);
    viewer.setReducedMotion(reducedMotion.matches);
  } catch (error) {
    showError(
      "3D preview unavailable",
      error instanceof Error ? error.message : String(error),
    );
  }

  const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {
    type: "module",
    name: "ardy-inference",
  });

  let workerReady = false;
  let modelReady = false;
  let modelLoading = false;
  let modelCaching = false;
  let activeLoadRequest: string | null = null;
  let activeGeneration: ActiveGeneration | null = null;
  let activeRestoreRequest: string | null = null;
  let announceContinuationRestore = true;
  let lastPackFiles: File[] | null = null;
  let packPendingCache = false;
  let cachedPack = false;
  let modelLabel = "";
  let modelBackend = "";
  let modelInfo: ModelLoadedEvent["model"] | null = null;
  let activeManifest: BrowserModelPackManifest | null = null;
  let pendingManifest: BrowserModelPackManifest | null = null;
  let generationProgressValue = 0;
  let modelProgressValue = 0;
  let generationReturnFocus: HTMLElement | null = null;
  let currentMotion: StructuredMotionResult | null = null;
  let currentContinuation: RuntimeContinuationState | null = null;
  let currentProvenance: MotionSessionProvenance = {};
  let editorState = cloneEditorState(DEFAULT_EDITOR_STATE);
  let activeVrmLoad = 0;
  let currentVrmInfo: VrmModelInfo | null = null;

  const postCommand = (command: WorkerCommand): void =>
    worker.postMessage(command);

  function announce(message: string): void {
    appStatus.textContent = "";
    window.requestAnimationFrame(() => {
      appStatus.textContent = message;
    });
  }

  function setProgress(
    fill: HTMLElement,
    progressbar: HTMLElement,
    fraction: number,
  ): number {
    const safeFraction = Math.max(0, Math.min(1, fraction));
    const percent = Math.round(safeFraction * 100);
    fill.style.transform = `translateX(-${100 - percent}%)`;
    progressbar.setAttribute("aria-valuenow", String(percent));
    return percent;
  }

  function setFieldInvalid(
    control: HTMLInputElement | HTMLTextAreaElement,
    invalid: boolean,
  ): void {
    if (invalid) {
      control.setAttribute("aria-invalid", "true");
    } else {
      control.removeAttribute("aria-invalid");
    }
    const field = control.closest<HTMLElement>('[data-slot="field"]');
    if (!field) return;
    if (invalid) {
      field.setAttribute("data-invalid", "true");
    } else {
      field.removeAttribute("data-invalid");
    }
  }

  function finiteInput(
    input: HTMLInputElement,
    fallback: number,
    minimum = -Number.MAX_VALUE,
    maximum = Number.MAX_VALUE,
  ): number {
    const value = Number(input.value);
    return Number.isFinite(value)
      ? Math.max(minimum, Math.min(maximum, value))
      : fallback;
  }

  function integerInput(
    input: HTMLInputElement,
    fallback: number,
    minimum: number,
    maximum: number,
  ): number {
    return Math.round(finiteInput(input, fallback, minimum, maximum));
  }

  function setModelStatus(
    state: "missing" | "loading" | "ready",
    title: string,
    detail: string,
    label: string,
  ): void {
    modelCard.dataset.state = state;
    modelState.dataset.state = state;
    modelTitle.textContent = title;
    modelDetail.textContent = detail;
    modelState.textContent = label;
    modelSetupHelp.hidden = state === "ready";
    importModelLabel.textContent =
      state === "ready" ? "Replace model pack" : "Choose model pack";
    form.dataset.modelState = state;
    modelRuntimeStatus.dataset.state = state;
    modelRuntimeDetail.textContent = `${title}. ${detail}`;
  }

  function resetRuntimeBadge(): void {
    gpuDot.dataset.state = hasWebGpu ? "available" : "unavailable";
    gpuLabel.textContent = hasWebGpu
      ? "WebGPU available"
      : "WebGPU unavailable";
  }

  function showError(title: string, message: string): void {
    errorReturnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorBanner.hidden = false;
    errorBanner.focus();
    announce(`${title}: ${message}`);
  }

  function clearError(restoreFocus = false): void {
    errorBanner.hidden = true;
    if (restoreFocus) errorReturnFocus?.focus();
    errorReturnFocus = null;
  }

  function showModelError(
    title: string,
    message: string,
    focus = true,
  ): void {
    modelErrorReturnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    modelErrorTitle.textContent = title;
    modelErrorMessage.textContent = message;
    modelErrorBanner.hidden = false;
    if (focus) modelErrorBanner.focus();
    announce(`${title}: ${message}`);
  }

  function clearModelError(restoreFocus = false): void {
    modelErrorBanner.hidden = true;
    if (restoreFocus) modelErrorReturnFocus?.focus();
    modelErrorReturnFocus = null;
  }

  function vrmDetailText(info: VrmModelInfo): string {
    const details = [
      info.metaVersion === "0" ? "VRM 0.x" : "VRM 1.0",
      info.version ? `model ${info.version}` : "",
      info.authors.length > 0 ? `by ${info.authors.join(", ")}` : "",
      "local preview",
    ].filter(Boolean);
    return details.join(" · ");
  }

  function setVrmStatus(info: VrmModelInfo | null): void {
    currentVrmInfo = info;
    vrmCard.dataset.state = info ? "ready" : "missing";
    vrmState.dataset.state = info ? "ready" : "missing";
    vrmName.textContent = info?.name ?? "No avatar loaded";
    vrmDetail.textContent = info
      ? vrmDetailText(info)
      : "Load a VRM 0.x or 1.0 file for local preview.";
    vrmState.textContent = info
      ? info.metaVersion === "0"
        ? "VRM 0.x"
        : "VRM 1.0"
      : "Optional";
    importVrmLabel.textContent = info ? "Replace VRM" : "Load VRM";
    removeVrm.disabled = !info;
    showVrm.disabled = !info;
  }

  function setVrmLoading(loading: boolean): void {
    vrmCard.toggleAttribute("aria-busy", loading);
    vrmState.dataset.state = loading
      ? "loading"
      : currentVrmInfo
        ? "ready"
        : "missing";
    vrmState.textContent = loading
      ? "Loading"
      : currentVrmInfo
        ? currentVrmInfo.metaVersion === "0"
          ? "VRM 0.x"
          : "VRM 1.0"
        : "Optional";
    importVrm.disabled = loading || viewer === null;
    removeVrm.disabled = loading || currentVrmInfo === null;
    showVrm.disabled = loading || currentVrmInfo === null;
    importVrmLabel.textContent = loading
      ? "Loading VRM…"
      : currentVrmInfo
        ? "Replace VRM"
        : "Load VRM";
  }

  function showVrmError(message: string): void {
    vrmErrorReturnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : importVrm;
    previewSettings.open = true;
    vrmErrorMessage.textContent = message;
    vrmErrorBanner.hidden = false;
    vrmErrorBanner.focus();
    announce(`VRM import failed: ${message}`);
  }

  function clearVrmError(restoreFocus = false): void {
    vrmErrorBanner.hidden = true;
    if (restoreFocus) vrmErrorReturnFocus?.focus();
    vrmErrorReturnFocus = null;
  }

  function updatePrompt(): void {
    promptCount.textContent = `${prompt.value.length} / 280`;
    if (prompt.dataset.validated === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        duration.value,
        seed.value,
      );
      promptError.textContent = validation.promptError ?? "";
      setFieldInvalid(prompt, Boolean(validation.promptError));
    }
    updateGenerateAvailability();
  }

  function updateDuration(): void {
    durationOutput.value = `${duration.value} seconds`;
    duration.setAttribute("aria-valuetext", `${duration.value} seconds`);
    const progress = ((Number(duration.value) - 2) / 8) * 100;
    duration.style.setProperty("--range-progress", `${progress}%`);
  }

  function updateSeed(): void {
    if (seed.dataset.validated === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        duration.value,
        seed.value,
      );
      seedError.textContent = validation.seedError ?? "";
      setFieldInvalid(seed, Boolean(validation.seedError));
    }
  }

  function updateTargetBuffer(): void {
    const output = targetBuffer
      .closest(".field-group")
      ?.querySelector<HTMLOutputElement>("output");
    if (output) output.value = `${targetBuffer.value} frames`;
    const min = Number(targetBuffer.min);
    const max = Number(targetBuffer.max);
    const progress = ((Number(targetBuffer.value) - min) / (max - min)) * 100;
    targetBuffer.style.setProperty("--range-progress", `${progress}%`);
  }

  function updateGenerateAvailability(): void {
    const generating = activeGeneration !== null;
    const generationBusy =
      generating || activeRestoreRequest !== null;
    const continuationAvailable = canContinueGeneration(
      modelReady,
      generationBusy,
      currentMotion !== null,
      currentContinuation !== null,
    );
    generate.disabled = !canAttemptGeneration(
      prompt.value,
      workerReady,
      modelReady,
      modelLoading || modelCaching,
      generationBusy,
    );
    applyPrompt.disabled =
      !modelReady ||
      generationBusy ||
      prompt.value.trim().length === 0 ||
      (currentMotion !== null && currentContinuation === null);
    restartGeneration.disabled = !modelReady || generationBusy;
    restartFromNow.disabled = !continuationAvailable;
    streamGeneration.disabled =
      generationBusy ||
      (currentMotion !== null && currentContinuation === null);
    importModel.disabled = modelLoading || modelCaching;
    removeModel.disabled = modelLoading || modelCaching;

    if (generationBusy) {
      generateHelp.textContent = activeRestoreRequest
        ? "Restoring the saved generation state."
        : "Generating locally. Received frames remain available if you cancel.";
    } else if (modelLoading || modelCaching) {
      generateHelp.textContent = "Preparing the local model pack.";
    } else if (!modelReady) {
      generateHelp.textContent = "Load a model pack to enable generation.";
    } else if (prompt.value.trim().length === 0) {
      generateHelp.textContent = "Describe a motion to enable generation.";
    } else if (currentMotion !== null && currentContinuation === null) {
      generateHelp.textContent =
        "Restart to create a new continuable motion session.";
    } else {
      generateHelp.textContent = "Ready to generate entirely in this browser.";
    }

    if (
      !generationBusy &&
      generationProgressElement.dataset.state !== "complete"
    ) {
      generationStage.textContent = modelReady
        ? "Ready to generate"
        : modelLoading || modelCaching
          ? "Preparing model"
          : "Waiting for model";
      generationPercent.textContent = "—";
    }
  }

  function markCurrentMotionPlaybackOnly(): void {
    currentContinuation = null;
    if (currentMotion) {
      streamGeneration.checked = false;
      motionBadge.removeAttribute("data-state");
      motionBadge.textContent =
        `${currentMotion.frameCount} frames · ${currentMotion.fps} FPS · playback only`;
    }
    updateGenerateAvailability();
  }

  function setGenerationBusy(active: boolean, background = false): void {
    const returnFocus =
      !active && document.activeElement === cancelGeneration
        ? generationReturnFocus
        : null;
    generationProgressElement.dataset.state = active
      ? "active"
      : generationProgressValue >= 1
        ? "complete"
        : "idle";
    generationProgressElement.setAttribute("aria-busy", String(active));
    cancelGeneration.dataset.state = active ? "active" : "idle";
    cancelGeneration.disabled = !active;
    cancelGeneration.setAttribute("aria-hidden", String(!active));
    cancelGeneration.tabIndex = active ? 0 : -1;
    cancelGeneration.textContent = "Cancel";
    generateLabel.textContent = active
      ? background
        ? "Extending motion"
        : "Generating motion"
      : "Generate motion";
    generateSpinner.classList.toggle("hidden", !active);
    buttonShortcut.hidden = active;
    generate.setAttribute("aria-busy", String(active));
    updateGenerateAvailability();
    if (!active) {
      generationReturnFocus = null;
      if (returnFocus?.isConnected) {
        window.requestAnimationFrame(() => returnFocus.focus());
      }
    }
  }

  function setEditor(next: MotionEditorState): void {
    editorState = cloneEditorState(next);
    try {
      viewer?.applyEditorState(editorState);
    } catch (error) {
      showError(
        "Could not update editor",
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  function outputVisibilityFromControls(): ViewerOutputVisibility {
    return {
      ...editorState.outputVisibility,
      skeleton: showSkeleton.checked,
      mesh: false,
      reference: false,
      contacts: showContacts.checked,
      orientationAxes: showOrientations.checked,
      trajectory: showTrajectory.checked,
    };
  }

  function syncOutputVisibility(): void {
    const visibility = outputVisibilityFromControls();
    editorState = { ...editorState, outputVisibility: visibility };
    viewer?.setOutputVisibility(visibility);
  }

  function updatePlayback(state: PlaybackState): void {
    const maxFrame = Math.max(0, state.frameCount - 1);
    timeline.max = String(maxFrame);
    timeline.value = String(Math.min(maxFrame, state.frame));
    timeline.disabled = state.frameCount === 0;
    playPause.disabled = state.frameCount < 2;
    playbackSpeed.disabled = state.frameCount === 0;
    loopToggle.disabled = state.frameCount < 2;
    playPause.dataset.playing = String(state.playing);
    playPause.setAttribute(
      "aria-label",
      state.playing ? "Pause motion" : "Play motion",
    );
    const progress = maxFrame > 0 ? (state.frame / maxFrame) * 100 : 0;
    timeline.style.setProperty("--range-progress", `${progress}%`);
    const elapsed = formatTime(state.frame / state.fps);
    const total = formatTime(maxFrame / state.fps);
    currentTime.textContent = elapsed;
    totalTime.textContent = total;
    timeline.setAttribute("aria-valuetext", `${elapsed} of ${total}`);
    maybeAutoReplan(state);
  }

  if (viewer) viewer.onPlaybackChange = updatePlayback;

  async function chooseModelPack(): Promise<File[] | null> {
    const picker = (window as unknown as DirectoryPickerWindow)
      .showDirectoryPicker;
    if (!picker) {
      fileInput.click();
      return null;
    }
    try {
      const directory = await picker({ mode: "read" });
      return canonicalizePackFiles(await collectDirectoryFiles(directory));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return null;
      }
      throw error;
    }
  }

  async function loadModelPack(
    files: File[],
    shouldCache: boolean,
  ): Promise<void> {
    clearError();
    clearModelError();
    const canonicalFiles = canonicalizePackFiles(files);
    const manifest = await manifestFromFiles(canonicalFiles);
    const totalBytes = canonicalFiles.reduce(
      (total, file) => total + file.size,
      0,
    );
    activeLoadRequest = requestId("load");
    lastPackFiles = canonicalFiles;
    pendingManifest = manifest;
    packPendingCache = shouldCache;
    modelLoading = true;
    modelReady = false;
    modelCard.setAttribute("aria-busy", "true");
    gpuDot.removeAttribute("data-state");
    gpuLabel.textContent =
      backend.value === "wasm"
        ? "Preparing WebAssembly"
        : backend.value === "webgpu"
          ? "Preparing WebGPU"
          : "Selecting runtime";
    modelProgress.hidden = false;
    modelProgressValue = 0;
    setProgress(modelProgressFill, modelProgressbar, 0);
    modelProgressLabel.textContent = "0%";
    setModelStatus(
      "loading",
      "Loading Core40 model",
      `${canonicalFiles.length} files · ${formatBytes(totalBytes)}`,
      "Loading",
    );
    updateGenerateAvailability();
    announce(
      `Loading model pack, ${formatBytes(totalBytes)}. Files stay on this device.`,
    );
    const command: LoadModelPackCommand = {
      type: "loadModelPack",
      requestId: activeLoadRequest,
      files: canonicalFiles,
      backend: backend.value as RuntimeBackendPreference,
      wasmPaths: ORT_WASM_PATH,
    };
    postCommand(command);
  }

  async function persistLoadedPack(files: File[]): Promise<void> {
    if (!supportsPersistentPackCache()) {
      modelDetail.textContent = `${modelLabel} · ${modelBackend} · ready for this tab`;
      return;
    }
    modelCaching = true;
    modelCard.setAttribute("aria-busy", "true");
    modelState.dataset.state = "loading";
    modelState.textContent = "Saving";
    modelProgress.hidden = false;
    modelProgressValue = 0;
    setProgress(modelProgressFill, modelProgressbar, 0);
    modelProgressLabel.textContent = "Caching";
    updateGenerateAvailability();
    try {
      await cacheModelPack(files, (completed, total) => {
        const percent =
          total > 0 ? Math.round((completed / total) * 100) : 0;
        modelProgressValue = Math.max(modelProgressValue, percent / 100);
        setProgress(
          modelProgressFill,
          modelProgressbar,
          modelProgressValue,
        );
        modelProgressLabel.textContent = `Saving ${percent}%`;
      });
      cachedPack = true;
      removeModel.hidden = false;
      modelDetail.textContent = `${modelLabel} · ${modelBackend} · saved on this device`;
    } catch (error) {
      cachedPack = false;
      removeModel.hidden = true;
      modelDetail.textContent = `${modelLabel} · ${modelBackend} · ready for this tab`;
      showModelError(
        "Model loaded, but not cached",
        error instanceof Error
          ? error.message
          : "The browser could not persist this model pack.",
        false,
      );
    } finally {
      modelCaching = false;
      modelCard.removeAttribute("aria-busy");
      modelState.dataset.state = "ready";
      modelState.textContent = "Ready";
      modelProgress.hidden = true;
      updateGenerateAvailability();
    }
  }

  function handleProgress(event: ProgressEvent): void {
    if (event.requestId === activeLoadRequest) {
      modelProgressValue = Math.max(
        modelProgressValue,
        modelLoadProgress(event),
      );
      const percent = setProgress(
        modelProgressFill,
        modelProgressbar,
        modelProgressValue,
      );
      modelProgress.hidden = false;
      modelProgressLabel.textContent = `${percent}%`;
      modelTitle.textContent = event.message || humanizeStage(event.stage);
      return;
    }
    if (event.requestId === activeGeneration?.id) {
      generationProgressValue = Math.max(
        generationProgressValue,
        generationProgress(event),
      );
      const percent = setProgress(
        generationProgressFill,
        generationProgressbar,
        generationProgressValue,
      );
      generationStage.textContent =
        event.message || humanizeStage(event.stage);
      generationPercent.textContent = `${percent}%`;
    }
  }

  function restoreWorkerContinuation(announceRestore = true): void {
    if (!currentContinuation || !modelReady) return;
    activeRestoreRequest = requestId("restore");
    announceContinuationRestore = announceRestore;
    updateGenerateAvailability();
    postCommand({
      type: "restoreContinuation",
      requestId: activeRestoreRequest,
      continuation: currentContinuation,
    });
  }

  function resetWorkerSession(): void {
    if (!modelReady) return;
    postCommand({
      type: "resetSession",
      requestId: requestId("reset"),
      seed: integerInput(seed, 2, 0, UINT32_MAX),
      initialTranslation: new Float32Array(
        DEFAULT_EDITOR_STATE.initialTransform.position,
      ),
      initialHeading: DEFAULT_EDITOR_STATE.initialTransform.headingRadians,
    });
  }

  function handleModelLoaded(event: ModelLoadedEvent): void {
    if (event.requestId !== activeLoadRequest) return;
    modelLoading = false;
    modelReady = true;
    activeLoadRequest = null;
    activeManifest = event.model.manifest ?? pendingManifest;
    pendingManifest = null;
    modelInfo = event.model;
    modelLabel = event.model.variant || event.model.id;
    modelBackend =
      event.model.backend === "webgpu" ? "WebGPU" : "WebAssembly";
    gpuDot.dataset.state = "available";
    gpuLabel.textContent = `Running on ${modelBackend}`;
    modelCard.removeAttribute("aria-busy");
    modelProgress.hidden = true;
    setModelStatus(
      "ready",
      event.model.variant || "ARDY Mini Core40",
      `${event.model.id} · ${modelBackend} · ${event.model.fps} FPS`,
      "Ready",
    );
    const incompatibleContinuation =
      currentContinuation !== null &&
      !isContinuationModelCompatible(currentProvenance, event.model);
    if (incompatibleContinuation) {
      markCurrentMotionPlaybackOnly();
    }
    removeModel.hidden = !cachedPack && !packPendingCache;
    updateGenerateAvailability();
    announce(
      incompatibleContinuation
        ? `${event.model.variant || "ARDY Mini Core40"} is ready on ${modelBackend}. The displayed motion remains playback-only because its continuation belongs to a different model pack.`
        : `${event.model.variant || "ARDY Mini Core40"} is ready on ${modelBackend}.`,
    );
    if (currentContinuation) {
      restoreWorkerContinuation();
    } else {
      resetWorkerSession();
    }
    if (packPendingCache && lastPackFiles) {
      packPendingCache = false;
      void persistLoadedPack(lastPackFiles);
    }
  }

  function refreshViewer(
    preserveFrame: number,
    preservePlaying: boolean,
    resetPresentation = false,
  ): void {
    if (!currentMotion || !viewer) return;
    viewer.setMotion(
      currentMotion,
      resetPresentation
        ? {
            frame: Math.min(
              preserveFrame,
              currentMotion.frameCount - 1,
            ),
            playing: preservePlaying,
            resetCamera: true,
          }
        : {
            playing: preservePlaying,
            resetCamera: false,
            preserveContinuity: true,
          },
    );
    setEditor(editorState);
    syncOutputVisibility();
    if (resetPresentation) {
      const loop = !streamGeneration.checked && !reducedMotion.matches;
      viewer.setLoop(loop);
      loopToggle.setAttribute("aria-pressed", String(loop));
      loopToggle.classList.toggle("is-active", loop);
    }
    emptyState.hidden = true;
    motionBadge.dataset.state = "ready";
    motionBadge.textContent = `${currentMotion.frameCount} frames · ${currentMotion.fps} FPS`;
    updateGenerateAvailability();
  }

  function applyChunk(chunk: RuntimeGenerationChunk): void {
    const active = activeGeneration;
    if (!active) return;
    const previousPlayback = viewer?.getPlaybackState();
    const incoming = motionFromRuntime(chunk, activeManifest);
    currentMotion = mergeMotion(currentMotion, incoming, chunk.startFrame);
    active.receivedChunk = true;
    currentProvenance = {
      prompt: chunk.prompt,
      seed: chunk.seed,
      backend: chunk.backend,
      modelId: modelInfo?.id,
      modelVariant: modelInfo?.variant,
      createdAt: new Date().toISOString(),
    };
    const preserveFrame =
      active.mode === "replace" && chunk.startFrame === 0
        ? 0
        : (previousPlayback?.frame ?? 0);
    const preservePlaying =
      active.mode === "replace"
        ? shouldAutoplayMotion(reducedMotion.matches)
        : Boolean(previousPlayback?.playing || active.resumePlayback);
    const resetPresentation = shouldResetMotionPresentation(
      active.mode,
      chunk.startFrame === 0,
    );
    refreshViewer(
      preserveFrame,
      preservePlaying,
      resetPresentation,
    );
    generationStage.textContent = `Received ${chunk.frameCount} frames`;
    announce(
      `Generated frames ${chunk.startFrame} through ${chunk.startFrame + chunk.frameCount - 1}.`,
    );
  }

  function finishGeneration(event: GenerationCompleteEvent): void {
    if (event.requestId !== activeGeneration?.id) return;
    const active = activeGeneration;
    try {
      if (!active.receivedChunk) {
        const incoming = motionFromRuntime(event.result, activeManifest);
        currentMotion = mergeMotion(
          currentMotion,
          incoming,
          event.result.startFrame,
        );
      }
      currentContinuation = event.result.continuation;
      const playback = viewer?.getPlaybackState();
      const resetPresentation = shouldResetMotionPresentation(
        active.mode,
        !active.receivedChunk,
      );
      refreshViewer(
        playback?.frame ?? 0,
        Boolean(playback?.playing || active.resumePlayback),
        resetPresentation,
      );
      generationProgressValue = 1;
      generationPercent.textContent = "100%";
      setProgress(generationProgressFill, generationProgressbar, 1);
      runtimeMetric.hidden = false;
      runtimeValue.textContent =
        event.result.timingsMs.total >= 1000
          ? `${(event.result.timingsMs.total / 1000).toFixed(2)} s`
          : `${Math.round(event.result.timingsMs.total)} ms`;
      activeGeneration = null;
      setGenerationBusy(false);
      announce(
        `Generation complete. The session contains ${event.sessionFrameCount} frames.`,
      );
      updateGenerateAvailability();
      if (
        window.matchMedia("(max-width: 760px)").matches &&
        active.mode === "replace"
      ) {
        viewportPanel.scrollIntoView({
          behavior: reducedMotion.matches ? "auto" : "smooth",
          block: "start",
        });
      }
    } catch (error) {
      activeGeneration = null;
      setGenerationBusy(false);
      showError(
        "Could not assemble generated motion",
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  function validateFormForGeneration(): FormValues | null {
    const validation = validateGenerationForm(
      prompt.value,
      duration.value,
      seed.value,
    );
    promptError.textContent = validation.promptError ?? "";
    seedError.textContent = validation.seedError ?? "";
    prompt.dataset.validated = "true";
    seed.dataset.validated = "true";
    setFieldInvalid(prompt, Boolean(validation.promptError));
    setFieldInvalid(seed, Boolean(validation.seedError));
    if (!validation.values) {
      if (validation.promptError) {
        prompt.focus();
      } else if (validation.seedError) {
        window.requestAnimationFrame(() => seed.focus());
      }
      return null;
    }
    return validation.values;
  }

  function startGeneration(
    mode: GenerationMode,
    options: {
      durationFrames?: number;
      durationSeconds?: number;
      branchFrame?: number;
      background?: boolean;
    } = {},
  ): void {
    if (
      !modelReady ||
      activeGeneration ||
      activeRestoreRequest ||
      !modelInfo
    ) {
      return;
    }
    if (
      mode !== "replace" &&
      !canContinueGeneration(
        modelReady,
        false,
        currentMotion !== null,
        currentContinuation !== null,
      )
    ) {
      updateGenerateAvailability();
      announce(
        "This motion is playback-only. Restart generation to create a new continuable session.",
      );
      return;
    }
    const values = validateFormForGeneration();
    if (!values) return;
    clearError();
    const background = Boolean(options.background);
    const id = requestId(mode);
    const playback = viewer?.getPlaybackState();
    activeGeneration = {
      id,
      mode,
      background,
      receivedChunk: false,
      resumePlayback: Boolean(
        playback?.playing || (background && mode === "append"),
      ),
    };
    generationProgressValue = 0;
    setProgress(generationProgressFill, generationProgressbar, 0);
    generationPercent.textContent = "0%";
    generationStage.textContent =
      mode === "append"
        ? "Extending session"
        : mode === "branch"
          ? "Replanning future"
          : "Starting session";
    if (!background) {
      generationReturnFocus =
        document.activeElement instanceof HTMLElement &&
        document.activeElement !== document.body
          ? document.activeElement
          : generate;
    }
    setGenerationBusy(true, background);
    if (!background) {
      window.requestAnimationFrame(() => {
        if (
          activeGeneration &&
          cancelGeneration.dataset.state === "active"
        ) {
          cancelGeneration.focus();
        }
      });
    }
    const requestedFuture = Math.min(
      DEFAULT_FUTURE_CROP_FRAMES,
      modelInfo.constraintMaxFrames,
    );
    const command: WorkerCommand = {
      type: "generate",
      requestId: id,
      mode,
      ...(mode === "branch"
        ? {
            branchFrame:
              options.branchFrame ??
              Math.floor(playback?.frame ?? currentMotion?.frameCount ?? 0),
          }
        : {}),
      prompt: values.prompt,
      seed: values.seed,
      ...(options.durationFrames !== undefined
        ? { durationFrames: options.durationFrames }
        : {
            durationSeconds:
              options.durationSeconds ?? values.durationSeconds,
          }),
      textCfgWeight: DEFAULT_TEXT_CFG_WEIGHT,
      constraintCfgWeight: DEFAULT_CONSTRAINT_CFG_WEIGHT,
      historyFrames: Math.min(
        DEFAULT_HISTORY_FRAMES,
        activeManifest?.dimensions.history_frames ?? DEFAULT_HISTORY_FRAMES,
      ),
      futureFrames: requestedFuture,
      ...(mode === "replace"
        ? {
            initialTranslation: new Float32Array(
              DEFAULT_EDITOR_STATE.initialTransform.position,
            ),
            initialHeading:
              DEFAULT_EDITOR_STATE.initialTransform.headingRadians,
          }
        : {}),
    };
    postCommand(command);
    announce(
      background
        ? "Extending the playback buffer locally."
        : mode === "branch"
          ? "Replanning the motion locally."
          : "Generation started locally.",
    );
  }

  function maybeAutoReplan(state: PlaybackState): void {
    if (
      !streamGeneration.checked ||
      !modelReady ||
      activeGeneration ||
      !currentMotion ||
      !currentContinuation ||
      state.frameCount === 0 ||
      (!state.playing && state.frame < state.frameCount - 1)
    ) {
      return;
    }
    const remaining = state.frameCount - state.frame - 1;
    const threshold = Math.min(
      DEFAULT_REPLAN_THRESHOLD_FRAMES,
      Math.max(1, Number(targetBuffer.value)),
    );
    if (remaining > threshold) return;
    const desired = integerInput(targetBuffer, 80, 1, 1000);
    const needed = Math.max(
      modelInfo?.generationFrames ?? 40,
      desired - remaining,
    );
    startGeneration("append", {
      durationFrames: needed,
      background: true,
    });
  }

  worker.addEventListener(
    "message",
    (message: MessageEvent<WorkerEvent>) => {
      const event = message.data;
      switch (event.type) {
        case "workerReady":
          workerReady = true;
          updateGenerateAvailability();
          postCommand({ type: "getStatus", requestId: requestId("status") });
          break;
        case "progress":
          handleProgress(event);
          break;
        case "modelLoaded":
          handleModelLoaded(event);
          break;
        case "generationChunk":
          if (event.requestId === activeGeneration?.id) {
            try {
              applyChunk(event.chunk);
            } catch (error) {
              showError(
                "Could not display generation chunk",
                error instanceof Error ? error.message : String(error),
              );
            }
          }
          break;
        case "generationComplete":
          finishGeneration(event);
          break;
        case "generationResult":
          finishGeneration({
            type: "generationComplete",
            requestId: event.requestId,
            mode: "replace",
            generatedFrameCount: event.result.frameCount,
            sessionFrameCount: event.result.continuation.frameCount,
            result: event.result,
          });
          break;
        case "sessionReset":
          markCurrentMotionPlaybackOnly();
          break;
        case "continuationRestored":
          if (event.requestId === activeRestoreRequest) {
            activeRestoreRequest = null;
            if (announceContinuationRestore) {
              announce(
                `Generation continuation restored at frame ${event.frameCount}.`,
              );
            }
            announceContinuationRestore = true;
            updateGenerateAvailability();
          }
          break;
        case "cancelled":
          if (event.targetRequestId === activeGeneration?.id) {
            activeGeneration = null;
            markCurrentMotionPlaybackOnly();
            setGenerationBusy(false);
            announce(
              currentMotion
                ? "Generation cancelled. Received frames remain available for playback, but continuation and replanning are disabled."
                : "Generation cancelled.",
            );
          }
          break;
        case "status":
          if (event.status.state === "empty" && !modelLoading) {
            modelReady = false;
          } else if (
            event.status.state === "ready" ||
            event.status.state === "generating"
          ) {
            modelReady = true;
          }
          updateGenerateAvailability();
          break;
        case "error": {
          const wasLoading = event.requestId === activeLoadRequest;
          const wasGenerating = event.requestId === activeGeneration?.id;
          const wasRestoring = event.requestId === activeRestoreRequest;
          if (wasLoading) {
            activeLoadRequest = null;
            pendingManifest = null;
            modelLoading = false;
            modelReady = false;
            modelProgress.hidden = true;
            modelCard.removeAttribute("aria-busy");
            resetRuntimeBadge();
            setModelStatus(
              "missing",
              "Model pack could not be loaded",
              "Check the pack and try importing it again.",
              "Error",
            );
          }
          if (wasGenerating) {
            activeGeneration = null;
            markCurrentMotionPlaybackOnly();
            setGenerationBusy(false);
          }
          if (wasRestoring) {
            activeRestoreRequest = null;
            announceContinuationRestore = true;
            markCurrentMotionPlaybackOnly();
            resetWorkerSession();
            announce(
              "The saved continuation is incompatible with this model pack. The motion remains available for playback only.",
            );
          }
          if (wasLoading) {
            showModelError("Model import failed", event.error.message);
          } else if (wasRestoring) {
            showError(
              "Continuation unavailable",
              `${event.error.message} The imported motion is still available for playback.`,
            );
          } else {
            showError("Inference failed", event.error.message);
          }
          updateGenerateAvailability();
          break;
        }
        case "disposed":
          modelReady = false;
          resetRuntimeBadge();
          updateGenerateAvailability();
          break;
        default:
          break;
      }
    },
    { signal: lifecycle.signal },
  );

  worker.addEventListener(
    "error",
    (event) => {
      modelLoading = false;
      modelReady = false;
      activeGeneration = null;
      setGenerationBusy(false);
      modelCard.removeAttribute("aria-busy");
      resetRuntimeBadge();
      setModelStatus(
        "missing",
        "Inference worker unavailable",
        "Reload the page to try again.",
        "Error",
      );
      showModelError(
        "Inference worker stopped",
        event.message || "An unexpected worker error occurred.",
      );
    },
    { signal: lifecycle.signal },
  );

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startGeneration("replace");
  });

  prompt.addEventListener("input", updatePrompt);
  duration.addEventListener("input", updateDuration);
  seed.addEventListener("input", updateSeed);
  targetBuffer.addEventListener("input", updateTargetBuffer);

  document.addEventListener(PROMPT_EXAMPLE_EVENT, (event) => {
    if (!(event instanceof CustomEvent) || typeof event.detail !== "string") {
      return;
    }
    prompt.value = event.detail;
    updatePrompt();
    prompt.focus();
  }, { signal: lifecycle.signal });

  randomizeSeed.addEventListener("click", () => {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    seed.value = String(random[0]);
    updateSeed();
  });

  backend.addEventListener("change", () => {
    const selected = backend.value as RuntimeBackendPreference;
    if (selected === "webgpu" && !hasWebGpu) {
      backendHelp.textContent =
        "WebGPU is unavailable. Select Auto or WebAssembly.";
    } else if (selected === "wasm") {
      backendHelp.textContent = crossOriginIsolated
        ? "WebAssembly uses local SIMD and multi-threaded CPU inference."
        : "WebAssembly is single-threaded without COOP/COEP headers.";
    } else {
      backendHelp.textContent =
        "Auto prefers WebGPU and falls back to WebAssembly.";
    }
    if (lastPackFiles && modelReady) {
      void loadModelPack(lastPackFiles, false).catch((error) =>
        showModelError(
          "Could not reload model",
          error instanceof Error ? error.message : String(error),
        ),
      );
    }
  });

  importModel.addEventListener("click", async () => {
    try {
      const files = await chooseModelPack();
      if (files) await loadModelPack(files, true);
    } catch (error) {
      showModelError(
        "Could not open model pack",
        error instanceof Error ? error.message : String(error),
      );
    }
  });

  fileInput.addEventListener("change", () => {
    const selected = fileInput.files?.length
      ? Array.from(fileInput.files)
      : null;
    fileInput.value = "";
    if (!selected) return;
    void loadModelPack(selected, true).catch((error) =>
      showModelError(
        "Could not open model pack",
        error instanceof Error ? error.message : String(error),
      ),
    );
  });

  removeModel.addEventListener("click", async () => {
    if (
      !window.confirm(
        "Unload the model and remove its saved browser copy? Generated motion will remain visible.",
      )
    ) {
      return;
    }
    try {
      await removeCachedModelPack();
      cachedPack = false;
      lastPackFiles = null;
      activeManifest = null;
      currentContinuation = null;
      modelReady = false;
      removeModel.hidden = true;
      postCommand({ type: "dispose", requestId: requestId("dispose") });
      resetRuntimeBadge();
      setModelStatus(
        "missing",
        "Model pack required",
        "Choose the exported Core40 browser-pack folder.",
        "Not loaded",
      );
      updateGenerateAvailability();
      announce("The model and its saved browser copy were removed.");
    } catch (error) {
      showModelError(
        "Could not remove cached pack",
        error instanceof Error ? error.message : String(error),
      );
    }
  });

  cancelGeneration.addEventListener("click", () => {
    if (!activeGeneration) return;
    cancelGeneration.disabled = true;
    cancelGeneration.textContent = "Cancelling…";
    generateLabel.textContent = "Cancelling…";
    postCommand({
      type: "cancel",
      requestId: requestId("cancel"),
      targetRequestId: activeGeneration.id,
    });
    announce("Cancelling after the current inference step.");
  });

  restartGeneration.addEventListener("click", () =>
    startGeneration("replace"),
  );
  restartFromNow.addEventListener("click", () => {
    const frame = viewer?.getPlaybackState().frame ?? 0;
    startGeneration("branch", {
      branchFrame: frame,
      durationFrames: integerInput(targetBuffer, 80, 1, 1000),
    });
  });
  applyPrompt.addEventListener("click", () => {
    if (!currentMotion) {
      startGeneration("replace");
      return;
    }
    const playback = viewer?.getPlaybackState();
    const buffer = DEFAULT_REPLAN_BUFFER_FRAMES;
    const branchFrame = Math.min(
      currentMotion.frameCount,
      (playback?.frame ?? 0) + buffer,
    );
    startGeneration("branch", {
      branchFrame,
      durationFrames: integerInput(targetBuffer, 80, 1, 1000),
      background: Boolean(playback?.playing),
    });
  });

  const outputControls: Array<
    [HTMLInputElement, keyof ViewerOutputVisibility]
  > = [
    [showSkeleton, "skeleton"],
    [showContacts, "contacts"],
    [showOrientations, "orientationAxes"],
    [showTrajectory, "trajectory"],
  ];
  for (const [control] of outputControls) {
    control.addEventListener("change", syncOutputVisibility);
  }
  importVrm.addEventListener("click", () => vrmFileInput.click());
  vrmFileInput.addEventListener("change", () => {
    const file = vrmFileInput.files?.[0];
    vrmFileInput.value = "";
    if (!file) return;
    const request = ++activeVrmLoad;
    clearVrmError();
    setVrmLoading(true);
    void (async () => {
      try {
        if (!viewer) {
          throw new Error("The 3D preview is unavailable in this browser.");
        }
        const info = await viewer.loadVrm(file);
        if (request !== activeVrmLoad) return;
        showVrm.checked = true;
        viewer.setVrmVisible(true);
        setVrmStatus(info);
        announce(`Loaded VRM avatar ${info.name}.`);
      } catch (error) {
        if (request !== activeVrmLoad) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        setVrmStatus(currentVrmInfo);
        showVrmError(error instanceof Error ? error.message : String(error));
      } finally {
        if (request === activeVrmLoad) setVrmLoading(false);
      }
    })();
  });
  removeVrm.addEventListener("click", () => {
    activeVrmLoad += 1;
    viewer?.clearVrm();
    showVrm.checked = true;
    clearVrmError();
    setVrmStatus(null);
    setVrmLoading(false);
    announce("Removed the VRM avatar.");
  });
  showVrm.addEventListener("change", () => {
    viewer?.setVrmVisible(showVrm.checked);
    announce(showVrm.checked ? "VRM avatar shown." : "VRM avatar hidden.");
  });

  dismissError.addEventListener("click", () => clearError(true));
  dismissModelError.addEventListener("click", () => clearModelError(true));
  dismissVrmError.addEventListener("click", () => clearVrmError(true));
  playPause.addEventListener("click", () => viewer?.togglePlaying());
  timeline.addEventListener("input", () =>
    viewer?.seek(Number(timeline.value)),
  );
  playbackSpeed.addEventListener("change", () =>
    viewer?.setSpeed(Number(playbackSpeed.value)),
  );
  loopToggle.addEventListener("click", () => {
    const loop = loopToggle.getAttribute("aria-pressed") !== "true";
    loopToggle.setAttribute("aria-pressed", String(loop));
    loopToggle.classList.toggle("is-active", loop);
    viewer?.setLoop(loop);
  });
  resetCamera.addEventListener("click", () => viewer?.resetCamera());
  streamGeneration.addEventListener("change", () => {
    if (streamGeneration.checked) {
      viewer?.setLoop(false);
      loopToggle.setAttribute("aria-pressed", "false");
      loopToggle.classList.remove("is-active");
      const state = viewer?.getPlaybackState();
      if (state) maybeAutoReplan(state);
    }
  });

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    viewer?.setReducedMotion(event.matches);
    if (event.matches) {
      viewer?.setPlaying(false);
      viewer?.setLoop(false);
      loopToggle.setAttribute("aria-pressed", "false");
      loopToggle.classList.remove("is-active");
      announce(
        "Reduced motion enabled. Playback is paused and looping is off.",
      );
    }
  };
  reducedMotion.addEventListener("change", handleReducedMotionChange, {
    signal: lifecycle.signal,
  });

  document.addEventListener(
    "keydown",
    (event) => {
      const target = event.target as HTMLElement | null;
      if (event.isComposing || event.repeat) return;
      if (
        (event.metaKey || event.ctrlKey) &&
        !event.altKey &&
        event.key === "Enter"
      ) {
        event.preventDefault();
        if (!generate.disabled) form.requestSubmit();
        return;
      }
      if (event.key === "Escape" && activeGeneration) {
        cancelGeneration.click();
        return;
      }
      if (
        target !== canvas ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey
      ) {
        return;
      }
      if (event.shiftKey && event.key.startsWith("Arrow")) {
        event.preventDefault();
        const horizontal =
          event.key === "ArrowLeft" ? 1 : event.key === "ArrowRight" ? -1 : 0;
        const vertical =
          event.key === "ArrowUp" ? 1 : event.key === "ArrowDown" ? -1 : 0;
        viewer?.orbit(horizontal, vertical);
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        viewer?.togglePlaying();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        const state = viewer?.getPlaybackState();
        if (state) viewer?.seek(state.frame - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        const state = viewer?.getPlaybackState();
        if (state) viewer?.seek(state.frame + 1);
      } else if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        viewer?.zoom("in");
      } else if (event.key === "-") {
        event.preventDefault();
        viewer?.zoom("out");
      } else if (event.key === "Home") {
        event.preventDefault();
        viewer?.resetCamera();
      }
    },
    { signal: lifecycle.signal },
  );

  const cleanup = (): void => {
    if (disposed) return;
    disposed = true;
    activeVrmLoad += 1;
    try {
      postCommand({ type: "dispose", requestId: requestId("dispose") });
    } catch {
      // The worker may already have stopped after a fatal runtime error.
    } finally {
      worker.terminate();
      lifecycle.abort();
      viewer?.dispose();
    }
  };
  window.addEventListener("beforeunload", cleanup, {
    signal: lifecycle.signal,
  });

  updatePrompt();
  updateDuration();
  updateSeed();
  updateTargetBuffer();
  syncOutputVisibility();
  setVrmStatus(null);
  setVrmLoading(false);
  modelCard.setAttribute("aria-busy", "true");
  setModelStatus(
    "loading",
    "Checking local model cache",
    "No network request is made.",
    "Checking",
  );
  void readCachedModelPack()
    .then(async (files) => {
      if (disposed) return;
      if (files) {
        cachedPack = true;
        removeModel.hidden = false;
        await loadModelPack(files, false);
      } else {
        modelCard.removeAttribute("aria-busy");
        setModelStatus(
          "missing",
          "Model pack required",
          "Choose the exported Core40 browser-pack folder.",
          "Not loaded",
        );
      }
    })
    .catch((error) => {
      if (disposed) return;
      modelCard.removeAttribute("aria-busy");
      setModelStatus(
        "missing",
        "Model pack required",
        "Persistent cache could not be read.",
        "Not loaded",
      );
      showModelError(
        "Could not read model cache",
        error instanceof Error ? error.message : String(error),
      );
    });
  return cleanup;
}

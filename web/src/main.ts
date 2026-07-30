// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  type RuntimeContinuationState,
  type RuntimeGenerationChunk,
  type RuntimeGenerationResult,
} from "./runtime/engine";
import { type BrowserModelPackManifest } from "./runtime/manifest";
import {
  type GenerationCompleteEvent,
  type GenerationMode,
  type LoadModelPackCommand,
  type ModelLoadedEvent,
  type ProgressEvent,
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
import {
  continuousGenerationControl,
  durationControl,
  generationProgressControl,
  loopControl,
  modelProgressControl,
  previewSettingsControl,
  removeSavedModelAction,
  showContactsControl,
  showOrientationsControl,
  showSkeletonControl,
  showTrajectoryControl,
  showVrmControl,
  targetBufferControl,
  timelineControl,
  type PressedControlState,
} from "./ui-control-store";

const UINT32_MAX = 0xffff_ffff;
const ALLOWED_DURATIONS = new Set([2, 4, 6, 8, 10]);
const CACHE_ROOT = "ardy-mini-model-cache";
const CACHE_ARCHIVE = "active-pack.tar.gz";
const DEFAULT_TEXT_CFG_WEIGHT = 3.5;
const DEFAULT_HISTORY_FRAMES = 40;
const DEFAULT_REPLAN_BUFFER_FRAMES = 20;
const DEFAULT_REPLAN_THRESHOLD_FRAMES = 10;

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

type WebGpuState = "checking" | "ready" | "unavailable";

interface WebGpuApi {
  requestAdapter(): Promise<unknown | null>;
}

async function webGpuUnavailableReason(): Promise<string | null> {
  if (globalThis.isSecureContext === false) {
    return "Open this demo over HTTPS or localhost, then reload the page.";
  }
  const gpu = (navigator as Navigator & { gpu?: WebGpuApi }).gpu;
  if (!gpu) {
    return "Use a browser and device that support WebGPU, then reload the page.";
  }
  try {
    if ((await gpu.requestAdapter()) === null) {
      return "No compatible GPU adapter is available. Check browser GPU settings or use another device.";
    }
  } catch {
    return "WebGPU could not initialize. Check browser GPU settings or use another device.";
  }
  return null;
}

export function cameraMoveForCode(
  code: string,
): readonly [forwardSteps: number, rightSteps: number] | null {
  if (code === "KeyW") return [1, 0];
  if (code === "KeyA") return [0, -1];
  if (code === "KeyS") return [-1, 0];
  if (code === "KeyD") return [0, 1];
  return null;
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

export function isModelPackArchive(file: File): boolean {
  return file.size > 0 && file.name.toLowerCase().endsWith(".tar.gz");
}

export function isVrmFile(file: File): boolean {
  return file.name.toLocaleLowerCase("en-US").endsWith(".vrm");
}

export function resolveGenerationProgressState(
  active: boolean,
  playbackOnly: boolean,
  progress: number,
): "active" | "complete" | "idle" | "playback-only" {
  if (active) return "active";
  if (playbackOnly) return "playback-only";
  return progress >= 1 ? "complete" : "idle";
}

export function shouldShowIdleGenerationStatus(
  generationBusy: boolean,
  state: string | undefined,
): boolean {
  return (
    !generationBusy &&
    state !== "complete" &&
    state !== "playback-only"
  );
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
    denoising: "Generating motion",
    decoding: "Decoding skeleton",
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
  if (stage === "encoding-text") return local * 0.1;
  if (stage === "denoising") return 0.1 + local * 0.78;
  if (stage === "decoding") return 0.88 + local * 0.12;
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

async function cacheModelPack(
  archive: File,
  onProgress: (completedBytes: number, totalBytes: number) => void,
): Promise<void> {
  const root = await getCacheRoot(true);
  if (!root) throw new Error("Persistent browser storage is unavailable.");

  const totalBytes = archive.size;
  const estimate = await navigator.storage.estimate();
  const available = (estimate.quota ?? 0) - (estimate.usage ?? 0);
  if (estimate.quota !== undefined && available < totalBytes * 1.05) {
    throw new Error(
      `The model pack needs ${formatBytes(totalBytes)}, but only about ${formatBytes(Math.max(0, available))} is available.`,
    );
  }

  await navigator.storage.persist?.();
  const handle = await root.getFileHandle(CACHE_ARCHIVE, {
    create: true,
  });
  const writable = await handle.createWritable();
  const reader = archive.stream().getReader();
  let completedBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
      completedBytes += value.byteLength;
      onProgress(completedBytes, totalBytes);
    }
    await writable.close();
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    await writable.abort(error).catch(() => {});
    throw error;
  }
}

async function readCachedModelPack(): Promise<File | null> {
  const root = await getCacheRoot(false);
  if (!root) return null;
  try {
    const stored = await (
      await root.getFileHandle(CACHE_ARCHIVE)
    ).getFile();
    if (stored.size === 0) {
      await removeCachedModelPack();
      return null;
    }
    return new File([stored], CACHE_ARCHIVE, {
      type: "application/gzip",
      lastModified: stored.lastModified,
    });
  } catch (error) {
    if (
      error instanceof DOMException &&
      (error.name === "NotFoundError" || error.name === "TypeMismatchError")
    ) {
      // The former directory-pack cache used this same root. It is not a
      // supported input and would otherwise consume quota beside the new
      // archive-only cache.
      await removeCachedModelPack();
      return null;
    }
    throw error;
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
  const durationOutput = requiredElement<HTMLOutputElement>("duration-output");
  const seed = requiredElement<HTMLInputElement>("seed");
  const seedError = requiredElement<HTMLElement>("seed-error");
  const randomizeSeed = requiredElement<HTMLButtonElement>("randomize-seed");
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
  const restartGeneration =
    requiredElement<HTMLButtonElement>("restart-generation");
  const restartFromNow =
    requiredElement<HTMLButtonElement>("restart-from-now");
  const applyPrompt = requiredElement<HTMLButtonElement>("apply-prompt");
  const cancelGeneration =
    requiredElement<HTMLButtonElement>("cancel-generation");
  const targetBufferOutput =
    requiredElement<HTMLOutputElement>("target-buffer-output");
  const generationProgressElement =
    requiredElement<HTMLElement>("generation-progress");
  const generationStage = requiredElement<HTMLElement>("generation-stage");
  const generationPercent = requiredElement<HTMLElement>("generation-percent");
  const errorBanner = requiredElement<HTMLElement>("error-banner");
  const errorTitle = requiredElement<HTMLElement>("error-title");
  const errorMessage = requiredElement<HTMLElement>("error-message");
  const dismissError = requiredElement<HTMLButtonElement>("dismiss-error");
  const canvas = requiredElement<HTMLCanvasElement>("motion-canvas");
  const playPause = requiredElement<HTMLButtonElement>("play-pause");
  const currentTime = requiredElement<HTMLElement>("current-time");
  const totalTime = requiredElement<HTMLElement>("total-time");
  const playbackSpeed =
    requiredElement<HTMLSelectElement>("playback-speed");
  const resetCamera = requiredElement<HTMLButtonElement>("reset-camera");
  const appStatus = requiredElement<HTMLElement>("app-status");
  const viewportPanel = requiredElement<HTMLElement>("viewport-panel");
  const viewport = requiredElement<HTMLElement>("viewport");

  const vrmCard = requiredElement<HTMLElement>("vrm-card");
  const vrmName = requiredElement<HTMLElement>("vrm-name");
  const vrmDetail = requiredElement<HTMLElement>("vrm-detail");
  const importVrm = requiredElement<HTMLButtonElement>("import-vrm");
  const importVrmLabel = requiredElement<HTMLElement>("import-vrm-label");
  const vrmFileInput = requiredElement<HTMLInputElement>("vrm-file-input");
  const removeVrm = requiredElement<HTMLButtonElement>("remove-vrm");
  const vrmErrorBanner =
    requiredElement<HTMLElement>("vrm-error-banner");
  const vrmErrorMessage =
    requiredElement<HTMLElement>("vrm-error-message");
  const dismissVrmError =
    requiredElement<HTMLButtonElement>("dismiss-vrm-error");
  const vrmDropTarget = requiredElement<HTMLElement>("vrm-drop-target");

  restartFromNow.disabled = true;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

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
  let lastPackArchive: File | null = null;
  let packPendingCache = false;
  let cachedPack = false;
  let modelLabel = "";
  let modelInfo: ModelLoadedEvent["model"] | null = null;
  let webGpuState: WebGpuState = "checking";
  let activeManifest: BrowserModelPackManifest | null = null;
  let generationProgressValue = 0;
  let modelProgressValue = 0;
  let generationReturnFocus: HTMLElement | null = null;
  let currentMotion: StructuredMotionResult | null = null;
  let currentContinuation: RuntimeContinuationState | null = null;
  let playbackOnlyStatus = false;
  let currentProvenance: MotionSessionProvenance = {};
  let editorState = cloneEditorState(DEFAULT_EDITOR_STATE);
  let activeVrmLoad = 0;
  let currentVrmInfo: VrmModelInfo | null = null;
  let vrmLoading = false;
  let vrmDragDepth = 0;

  const postCommand = (command: WorkerCommand): void =>
    worker.postMessage(command);

  function announce(message: string): void {
    appStatus.textContent = "";
    window.requestAnimationFrame(() => {
      appStatus.textContent = message;
    });
  }

  function updateLoopControl(next: Partial<PressedControlState>): void {
    loopControl.setState(next);
  }

  function setProgress(
    control: typeof modelProgressControl,
    fraction: number,
  ): number {
    const safeFraction = Math.max(0, Math.min(1, fraction));
    const percent = Math.round(safeFraction * 100);
    control.commit(percent);
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

  function integerValue(
    value: number,
    fallback: number,
    minimum: number,
    maximum: number,
  ): number {
    return Number.isFinite(value)
      ? Math.round(Math.max(minimum, Math.min(maximum, value)))
      : fallback;
  }

  function setModelStatus(
    state: "missing" | "loading" | "ready" | "unavailable",
    title: string,
    detail: string,
    label: string,
  ): void {
    modelCard.dataset.state = state;
    modelState.dataset.state = state;
    modelTitle.textContent = title;
    modelDetail.textContent = detail;
    modelState.textContent = label;
    modelSetupHelp.hidden = state === "ready" || state === "unavailable";
    importModelLabel.textContent =
      state === "ready" ? "Replace model pack" : "Choose model pack";
    form.dataset.modelState = state;
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
    ].filter(Boolean);
    return details.join(" · ");
  }

  function setVrmStatus(info: VrmModelInfo | null): void {
    currentVrmInfo = info;
    vrmCard.dataset.state = info ? "ready" : "missing";
    vrmName.textContent = info?.name ?? "No avatar loaded";
    vrmDetail.textContent = info
      ? vrmDetailText(info)
      : "Load a VRM 0.x or 1.0 file.";
    importVrmLabel.textContent = info ? "Replace VRM" : "Load VRM";
    removeVrm.disabled = !info;
    showVrmControl.setState({ disabled: !info });
  }

  function setVrmLoading(loading: boolean): void {
    vrmLoading = loading;
    vrmCard.toggleAttribute("aria-busy", loading);
    importVrm.disabled = loading || viewer === null;
    removeVrm.disabled = loading || currentVrmInfo === null;
    showVrmControl.setState({
      disabled: loading || currentVrmInfo === null,
    });
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
    previewSettingsControl.setState({ open: true });
    vrmErrorMessage.textContent = message;
    vrmErrorBanner.hidden = false;
    window.requestAnimationFrame(() => {
      if (!vrmErrorBanner.hidden) vrmErrorBanner.focus();
    });
    announce(`VRM import failed: ${message}`);
  }

  function clearVrmError(restoreFocus = false): void {
    vrmErrorBanner.hidden = true;
    if (restoreFocus) vrmErrorReturnFocus?.focus();
    vrmErrorReturnFocus = null;
  }

  async function loadVrmFile(file: File): Promise<void> {
    if (vrmLoading) {
      showVrmError("Wait for the current VRM file to finish loading.");
      return;
    }
    if (!isVrmFile(file)) {
      showVrmError("Choose a .vrm file.");
      return;
    }

    const request = ++activeVrmLoad;
    clearVrmError();
    setVrmLoading(true);
    try {
      if (!viewer) {
        throw new Error("The 3D preview is unavailable in this browser.");
      }
      const info = await viewer.loadVrm(file);
      if (request !== activeVrmLoad) return;
      clearVrmError();
      showVrmControl.setState({ checked: true });
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
  }

  function setVrmDropTargetVisible(visible: boolean): void {
    vrmDropTarget.hidden = !visible;
  }

  function resetVrmDropTarget(): void {
    vrmDragDepth = 0;
    setVrmDropTargetVisible(false);
  }

  function hasDraggedFiles(event: DragEvent): boolean {
    return Array.from(event.dataTransfer?.types ?? []).includes("Files");
  }

  function updatePrompt(): void {
    promptCount.textContent = `${prompt.value.length} / 280`;
    if (prompt.dataset.validated === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        String(durationControl.getSnapshot().value),
        seed.value,
      );
      promptError.textContent = validation.promptError ?? "";
      setFieldInvalid(prompt, Boolean(validation.promptError));
    }
    updateGenerateAvailability();
  }

  function updateDuration(): void {
    const value = durationControl.getSnapshot().value;
    const valueText = `${value} seconds`;
    durationOutput.value = valueText;
    durationControl.setState({ ariaValueText: valueText });
  }

  function updateSeed(): void {
    if (seed.dataset.validated === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        String(durationControl.getSnapshot().value),
        seed.value,
      );
      seedError.textContent = validation.seedError ?? "";
      setFieldInvalid(seed, Boolean(validation.seedError));
    }
  }

  function updateTargetBuffer(): void {
    const value = targetBufferControl.getSnapshot().value;
    const valueText = `${value} frames`;
    targetBufferOutput.value = valueText;
    targetBufferControl.setState({ ariaValueText: valueText });
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
    continuousGenerationControl.setState({
      disabled:
        generationBusy ||
        (currentMotion !== null && currentContinuation === null),
    });
    importModel.disabled =
      webGpuState !== "ready" || modelLoading || modelCaching;
    fileInput.disabled = importModel.disabled;
    removeModel.disabled = modelLoading || modelCaching;

    if (generationBusy) {
      generateHelp.textContent = activeRestoreRequest
        ? "Restoring the saved generation state."
        : "Generating locally. Received frames remain available if you cancel.";
    } else if (webGpuState === "checking") {
      generateHelp.textContent = "Checking WebGPU support.";
    } else if (webGpuState === "unavailable") {
      generateHelp.textContent =
        "WebGPU is required to load the model and generate motion.";
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
      shouldShowIdleGenerationStatus(
        generationBusy,
        generationProgressElement.dataset.state,
      )
    ) {
      generationStage.textContent = modelReady
        ? "Ready to generate"
        : webGpuState === "checking"
          ? "Checking WebGPU"
          : webGpuState === "unavailable"
            ? "WebGPU unavailable"
            : modelLoading || modelCaching
              ? "Preparing model"
              : "Waiting for model";
      generationPercent.textContent = "—";
    }
  }

  function markCurrentMotionPlaybackOnly(): void {
    currentContinuation = null;
    playbackOnlyStatus = currentMotion !== null;
    if (currentMotion) {
      continuousGenerationControl.setState({ checked: false });
      generationProgressElement.dataset.state = "playback-only";
      generationStage.textContent =
        `${currentMotion.frameCount} frames · playback only`;
      generationPercent.textContent = "—";
    }
    updateGenerateAvailability();
  }

  function setGenerationBusy(active: boolean, background = false): void {
    const activeElement = document.activeElement;
    const returnFocus =
      !active &&
      (activeElement === cancelGeneration || activeElement === document.body)
        ? generationReturnFocus
        : null;
    generationProgressElement.dataset.state = resolveGenerationProgressState(
      active,
      playbackOnlyStatus,
      generationProgressValue,
    );
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
    generate.setAttribute("aria-busy", String(active));
    updateGenerateAvailability();
    if (!active) {
      generationReturnFocus = null;
      if (returnFocus?.isConnected) {
        returnFocus.focus({ preventScroll: true });
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
      skeleton: showSkeletonControl.getSnapshot().checked,
      mesh: false,
      reference: false,
      contacts: showContactsControl.getSnapshot().checked,
      orientationAxes: showOrientationsControl.getSnapshot().checked,
      trajectory: showTrajectoryControl.getSnapshot().checked,
      constraints: false,
      initialTransform: false,
      waypoints: false,
    };
  }

  function syncOutputVisibility(): void {
    const visibility = outputVisibilityFromControls();
    editorState = { ...editorState, outputVisibility: visibility };
    viewer?.setOutputVisibility(visibility);
  }

  function updatePlayback(state: PlaybackState): void {
    const maxFrame = Math.max(0, state.frameCount - 1);
    playPause.disabled = state.frameCount < 2;
    playbackSpeed.disabled = state.frameCount === 0;
    updateLoopControl({ disabled: state.frameCount < 2 });
    playPause.dataset.playing = String(state.playing);
    playPause.setAttribute(
      "aria-label",
      state.playing ? "Pause motion" : "Play motion",
    );
    const elapsed = formatTime(state.frame / state.fps);
    const total = formatTime(maxFrame / state.fps);
    currentTime.textContent = elapsed;
    totalTime.textContent = total;
    timelineControl.setState({
      max: Math.max(1, maxFrame),
      value: Math.min(maxFrame, state.frame),
      disabled: state.frameCount === 0,
      ariaValueText: `${elapsed} of ${total}`,
    });
    maybeAutoReplan(state);
  }

  if (viewer) viewer.onPlaybackChange = updatePlayback;

  async function loadModelPack(
    archive: File,
    shouldCache: boolean,
  ): Promise<void> {
    clearError();
    clearModelError();
    if (webGpuState !== "ready") {
      throw new Error(
        "WebGPU is required. Use a supported browser and device over HTTPS or localhost.",
      );
    }
    if (!isModelPackArchive(archive)) {
      throw new Error("Choose a non-empty .tar.gz model-pack file.");
    }
    const totalBytes = archive.size;
    activeLoadRequest = requestId("load");
    lastPackArchive = archive;
    packPendingCache = shouldCache;
    modelLoading = true;
    modelReady = false;
    modelCard.setAttribute("aria-busy", "true");
    modelProgress.hidden = false;
    modelProgressValue = 0;
    setProgress(modelProgressControl, 0);
    modelProgressLabel.textContent = "0%";
    setModelStatus(
      "loading",
      "Loading Core40 model",
      `${archive.name} · ${formatBytes(totalBytes)}`,
      "Loading",
    );
    updateGenerateAvailability();
    announce(
      `Loading model pack, ${formatBytes(totalBytes)}. The archive stays on this device.`,
    );
    const command: LoadModelPackCommand = {
      type: "loadModelPack",
      requestId: activeLoadRequest,
      archive,
    };
    postCommand(command);
  }

  async function persistLoadedPack(archive: File): Promise<void> {
    if (!supportsPersistentPackCache()) {
      modelDetail.textContent = modelInfo?.id ?? modelLabel;
      return;
    }
    modelCaching = true;
    modelCard.setAttribute("aria-busy", "true");
    modelState.dataset.state = "loading";
    modelState.textContent = "Saving";
    modelProgress.hidden = false;
    modelProgressValue = 0;
    setProgress(modelProgressControl, 0);
    modelProgressLabel.textContent = "Caching";
    updateGenerateAvailability();
    try {
      await cacheModelPack(archive, (completed, total) => {
        const percent =
          total > 0 ? Math.round((completed / total) * 100) : 0;
        modelProgressValue = Math.max(modelProgressValue, percent / 100);
        setProgress(modelProgressControl, modelProgressValue);
        modelProgressLabel.textContent = `Saving ${percent}%`;
      });
      cachedPack = true;
      removeModel.hidden = false;
      modelDetail.textContent = modelInfo?.id ?? modelLabel;
    } catch (error) {
      cachedPack = false;
      removeModel.hidden = true;
      modelDetail.textContent = modelInfo?.id ?? modelLabel;
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
        modelProgressControl,
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
        generationProgressControl,
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
    activeManifest = event.model.manifest;
    modelInfo = event.model;
    modelLabel = event.model.variant || event.model.id;
    modelCard.removeAttribute("aria-busy");
    modelProgress.hidden = true;
    setModelStatus(
      "ready",
      event.model.variant || "ARDY Mini Core40",
      event.model.id,
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
        ? `${event.model.variant || "ARDY Mini Core40"} is ready on WebGPU. The displayed motion remains playback-only because its continuation belongs to a different model pack.`
        : `${event.model.variant || "ARDY Mini Core40"} is ready on WebGPU.`,
    );
    if (currentContinuation) {
      restoreWorkerContinuation();
    } else {
      resetWorkerSession();
    }
    if (packPendingCache && lastPackArchive) {
      packPendingCache = false;
      void persistLoadedPack(lastPackArchive);
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
      const loop =
        !continuousGenerationControl.getSnapshot().checked &&
        !reducedMotion.matches;
      viewer.setLoop(loop);
      updateLoopControl({ pressed: loop });
    }
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
      playbackOnlyStatus = false;
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
      generationStage.textContent = `${event.sessionFrameCount} frames`;
      generationPercent.textContent =
        `${Math.round(event.result.timingsMs.total)} ms`;
      setProgress(generationProgressControl, 1);
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
      String(durationControl.getSnapshot().value),
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
      returnFocus?: HTMLElement;
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
    playbackOnlyStatus = false;
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
    setProgress(generationProgressControl, 0);
    generationPercent.textContent = "0%";
    generationStage.textContent =
      mode === "append"
        ? "Extending session"
        : mode === "branch"
          ? "Replanning future"
          : "Starting session";
    if (!background) {
      generationReturnFocus =
        options.returnFocus ??
        (document.activeElement instanceof HTMLElement &&
        document.activeElement !== document.body
          ? document.activeElement
          : generate);
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
      cfgWeight: DEFAULT_TEXT_CFG_WEIGHT,
      historyFrames: Math.min(
        DEFAULT_HISTORY_FRAMES,
        activeManifest?.dimensions.history_frames ?? DEFAULT_HISTORY_FRAMES,
      ),
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
      !continuousGenerationControl.getSnapshot().checked ||
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
      Math.max(1, targetBufferControl.getSnapshot().value),
    );
    if (remaining > threshold) return;
    const desired = integerValue(
      targetBufferControl.getSnapshot().value,
      80,
      1,
      1000,
    );
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
            modelLoading = false;
            modelReady = false;
            modelProgress.hidden = true;
            modelCard.removeAttribute("aria-busy");
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
    const submitter =
      event.submitter instanceof HTMLElement
        ? event.submitter
        : document.activeElement instanceof HTMLElement &&
            document.activeElement !== document.body
          ? document.activeElement
          : generate;
    startGeneration("replace", { returnFocus: submitter });
  });

  prompt.addEventListener("input", updatePrompt);
  seed.addEventListener("input", updateSeed);
  durationControl.onCommit(updateDuration, lifecycle.signal);
  targetBufferControl.onCommit(updateTargetBuffer, lifecycle.signal);

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

  importModel.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    const selected = fileInput.files?.item(0) ?? null;
    fileInput.value = "";
    if (!selected) return;
    void loadModelPack(selected, true).catch((error) =>
      showModelError(
        "Could not open model pack",
        error instanceof Error ? error.message : String(error),
      ),
    );
  });

  removeSavedModelAction.onTrigger(async () => {
    try {
      await removeCachedModelPack();
      cachedPack = false;
      lastPackArchive = null;
      activeManifest = null;
      currentContinuation = null;
      modelReady = false;
      removeModel.hidden = true;
      postCommand({ type: "dispose", requestId: requestId("dispose") });
      setModelStatus(
        "missing",
        "Model pack required",
        "Choose the exported Core40 .tar.gz model pack.",
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
  }, lifecycle.signal);

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
    startGeneration("replace", { returnFocus: restartGeneration }),
  );
  restartFromNow.addEventListener("click", () => {
    const frame = viewer?.getPlaybackState().frame ?? 0;
    startGeneration("branch", {
      returnFocus: restartFromNow,
      branchFrame: frame,
      durationFrames: integerValue(
        targetBufferControl.getSnapshot().value,
        80,
        1,
        1000,
      ),
    });
  });
  applyPrompt.addEventListener("click", () => {
    if (!currentMotion) {
      startGeneration("replace", { returnFocus: applyPrompt });
      return;
    }
    const playback = viewer?.getPlaybackState();
    const buffer = DEFAULT_REPLAN_BUFFER_FRAMES;
    const branchFrame = Math.min(
      currentMotion.frameCount,
      (playback?.frame ?? 0) + buffer,
    );
    startGeneration("branch", {
      returnFocus: applyPrompt,
      branchFrame,
      durationFrames: integerValue(
        targetBufferControl.getSnapshot().value,
        80,
        1,
        1000,
      ),
      background: Boolean(playback?.playing),
    });
  });

  for (const control of [
    showSkeletonControl,
    showContactsControl,
    showOrientationsControl,
    showTrajectoryControl,
  ]) {
    control.onCommit(syncOutputVisibility, lifecycle.signal);
  }
  importVrm.addEventListener("click", () => vrmFileInput.click());
  vrmFileInput.addEventListener("change", () => {
    const file = vrmFileInput.files?.[0];
    vrmFileInput.value = "";
    if (!file) return;
    void loadVrmFile(file);
  });
  window.addEventListener(
    "dragenter",
    (event) => {
      if (!hasDraggedFiles(event)) return;
      event.preventDefault();
      vrmDragDepth += 1;
      setVrmDropTargetVisible(true);
    },
    { signal: lifecycle.signal },
  );
  window.addEventListener(
    "dragover",
    (event) => {
      if (!hasDraggedFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      setVrmDropTargetVisible(true);
    },
    { signal: lifecycle.signal },
  );
  window.addEventListener(
    "dragleave",
    (event) => {
      if (!hasDraggedFiles(event)) return;
      vrmDragDepth = Math.max(0, vrmDragDepth - 1);
      if (vrmDragDepth === 0) setVrmDropTargetVisible(false);
    },
    { signal: lifecycle.signal },
  );
  window.addEventListener(
    "drop",
    (event) => {
      if (!hasDraggedFiles(event)) return;
      event.preventDefault();
      const files = Array.from(event.dataTransfer?.files ?? []);
      resetVrmDropTarget();
      if (files.length !== 1 || !isVrmFile(files[0])) {
        showVrmError("Drop a single .vrm file.");
        return;
      }
      void loadVrmFile(files[0]);
    },
    { signal: lifecycle.signal },
  );
  window.addEventListener("dragend", resetVrmDropTarget, {
    signal: lifecycle.signal,
  });
  window.addEventListener("blur", resetVrmDropTarget, {
    signal: lifecycle.signal,
  });
  removeVrm.addEventListener("click", () => {
    activeVrmLoad += 1;
    viewer?.clearVrm();
    showVrmControl.setState({ checked: true });
    clearVrmError();
    setVrmStatus(null);
    setVrmLoading(false);
    announce("Removed the VRM avatar.");
  });
  showVrmControl.onCommit((checked) => {
    viewer?.setVrmVisible(checked);
    announce(checked ? "VRM avatar shown." : "VRM avatar hidden.");
  }, lifecycle.signal);

  dismissError.addEventListener("click", () => clearError(true));
  dismissModelError.addEventListener("click", () => clearModelError(true));
  dismissVrmError.addEventListener("click", () => clearVrmError(true));
  playPause.addEventListener("click", () => viewer?.togglePlaying());
  timelineControl.onCommit((value) => viewer?.seek(value), lifecycle.signal);
  playbackSpeed.addEventListener("change", () =>
    viewer?.setSpeed(Number(playbackSpeed.value)),
  );
  loopControl.onCommit((pressed) => viewer?.setLoop(pressed), lifecycle.signal);
  resetCamera.addEventListener("click", () => viewer?.resetCamera());
  continuousGenerationControl.onCommit((checked) => {
    if (checked) {
      viewer?.setLoop(false);
      updateLoopControl({ pressed: false });
      const state = viewer?.getPlaybackState();
      if (state) maybeAutoReplan(state);
    }
  }, lifecycle.signal);

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    viewer?.setReducedMotion(event.matches);
    if (event.matches) {
      viewer?.setPlaying(false);
      viewer?.setLoop(false);
      updateLoopControl({ pressed: false });
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
      if (event.isComposing) return;
      const cameraMove =
        target === canvas &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.shiftKey
          ? cameraMoveForCode(event.code)
          : null;
      if (cameraMove) {
        event.preventDefault();
        viewer?.moveCamera(cameraMove[0], cameraMove[1]);
        return;
      }
      if (event.repeat) return;
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
    resetVrmDropTarget();
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

  async function initializeModelPack(): Promise<void> {
    const unavailableReason = await webGpuUnavailableReason();
    if (disposed) return;
    if (unavailableReason) {
      webGpuState = "unavailable";
      modelCard.removeAttribute("aria-busy");
      setModelStatus(
        "unavailable",
        "WebGPU required",
        unavailableReason,
        "Unavailable",
      );
      updateGenerateAvailability();
      announce(`WebGPU is unavailable. ${unavailableReason}`);
      return;
    }

    webGpuState = "ready";
    setModelStatus(
      "loading",
      "Checking local model cache",
      "No network request is made.",
      "Checking",
    );
    updateGenerateAvailability();
    try {
      const archive = await readCachedModelPack();
      if (disposed) return;
      if (archive) {
        cachedPack = true;
        removeModel.hidden = false;
        await loadModelPack(archive, false);
      } else {
        modelCard.removeAttribute("aria-busy");
        setModelStatus(
          "missing",
          "Model pack required",
          "Choose the exported Core40 .tar.gz model pack.",
          "Not loaded",
        );
        updateGenerateAvailability();
      }
    } catch (error) {
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
      updateGenerateAvailability();
    }
  }

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
    "Checking WebGPU",
    "A compatible GPU and secure context are required.",
    "Checking",
  );
  updateGenerateAvailability();
  void initializeModelPack();
  return cleanup;
}

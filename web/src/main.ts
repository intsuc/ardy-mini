// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  type RuntimeContinuationState,
  type RuntimeGenerationChunk,
  type RuntimeGenerationResult,
} from "./runtime/engine";
import { type BrowserModelManifest } from "./runtime/manifest";
import {
  clearModelCache,
  fetchModelManifest,
  inspectModelCache,
  modelTransportSize,
  type ModelCacheStatus,
  type ModelManifestSource,
} from "./runtime/model-assets";
import {
  modelVariantBaseUrl,
  preferredModelVariant,
  type BrowserModelVariant,
} from "./runtime/model-variant";
import {
  type GenerationCompleteEvent,
  type GenerationMode,
  type LoadModelCommand,
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
  clearModelCacheAction,
  generationActionsControl,
  modelDownloadAction,
  modelDownloadCancelAction,
  modelUiControl,
  playbackSpeedControl,
  playPauseControl,
  previewSettingsControl,
  previewSettingsTabControl,
  regenerateMotionAction,
  showContactsControl,
  showOrientationsControl,
  showSkeletonControl,
  showTrajectoryControl,
  showVrmControl,
  targetBufferControl,
  timelineControl,
  unsupportedDeviceControl,
  startNewMotionAction,
} from "./ui-control-store";

const UINT32_MAX = 0xffff_ffff;
const DEVELOPMENT_MODEL_FAMILY_BASE_URL =
  "/models/ardy-minilm-core40-browser-v1/";
const DEFAULT_TEXT_CFG_WEIGHT = 3.5;
const DEFAULT_HISTORY_FRAMES = 40;
const DEFAULT_REPLAN_BUFFER_FRAMES = 20;
const DEFAULT_REPLAN_THRESHOLD_FRAMES = 10;

interface FormValues {
  prompt: string;
  seed: number;
}

interface FormValidation {
  values?: FormValues;
  promptError?: string;
  seedError?: string;
}

type WebGpuState = "checking" | "ready" | "unavailable";

interface WebGpuApi {
  requestAdapter(): Promise<GPUAdapter | null>;
}

interface WebGpuPreflight {
  unavailableReason: string | null;
}

async function inspectWebGpuSupport(): Promise<WebGpuPreflight> {
  if (globalThis.isSecureContext === false) {
    return {
      unavailableReason:
        "Open this demo over HTTPS or localhost, then reload the page.",
    };
  }
  const gpu = (navigator as Navigator & { gpu?: WebGpuApi }).gpu;
  if (!gpu) {
    return {
      unavailableReason:
        "Use a browser and device that support WebGPU, then reload the page.",
    };
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    return {
      unavailableReason:
        "The browser could not initialize a WebGPU adapter. Check GPU acceleration, then reload the page.",
    };
  }
  if (!adapter) {
    return {
      unavailableReason:
        "No compatible WebGPU adapter is available on this device.",
    };
  }
  return {
    unavailableReason: null,
  };
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

export function cameraMovementForCodes(
  codes: Iterable<string>,
): readonly [forward: number, right: number] {
  let forward = 0;
  let right = 0;
  for (const code of codes) {
    const movement = cameraMoveForCode(code);
    if (!movement) continue;
    forward += movement[0];
    right += movement[1];
  }
  return [Math.sign(forward), Math.sign(right)];
}

interface ActiveGeneration {
  id: string;
  mode: GenerationMode;
  action:
    | "start"
    | "update"
    | "regenerate"
    | "new-motion"
    | "extend";
  receivedChunk: boolean;
  resumePlayback: boolean;
}

export function validateGenerationForm(
  promptValue: string,
  seedValue: string,
): FormValidation {
  const prompt = promptValue.trim();
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

  if (!validation.promptError && !validation.seedError) {
    validation.values = { prompt, seed };
  }
  return validation;
}

export interface PromptActionState {
  label: "Start motion" | "Update motion";
  dirty: boolean;
  canSubmit: boolean;
}

/**
 * The textarea is a draft. A live session keeps using its active prompt until
 * the user explicitly submits a different non-empty draft.
 */
export function resolvePromptActionState(
  hasMotion: boolean,
  hasContinuation: boolean,
  draftPrompt: string,
  activePrompt: string | null,
): PromptActionState {
  const draft = draftPrompt.trim();
  const dirty = activePrompt === null || draft !== activePrompt;
  return {
    label: hasMotion ? "Update motion" : "Start motion",
    dirty,
    canSubmit:
      draft.length > 0 &&
      draft.length <= 280 &&
      (!hasMotion || (hasContinuation && dirty)),
  };
}

export function livePromptBranchFrame(
  playhead: number,
  frameCount: number,
): number {
  const safeFrameCount = Math.max(0, Math.floor(frameCount));
  const safePlayhead = Math.max(0, Math.floor(playhead));
  return Math.min(safeFrameCount, safePlayhead + DEFAULT_REPLAN_BUFFER_FRAMES);
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

function configuredModelFamilyBaseUrl(): string | null {
  if (import.meta.env.DEV) {
    return new URL(
      DEVELOPMENT_MODEL_FAMILY_BASE_URL,
      globalThis.location.href,
    ).href;
  }
  const configured = import.meta.env.VITE_MODEL_BASE_URL?.trim();
  return configured
    ? new URL(
        configured.endsWith("/") ? configured : `${configured}/`,
        globalThis.location.href,
      ).href
    : null;
}

export function isVrmFile(file: File): boolean {
  return file.name.toLocaleLowerCase("en-US").endsWith(".vrm");
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
  manifest: BrowserModelManifest | null,
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
  unsupportedDeviceControl.setState({ open: false });
  modelUiControl.dispatch({ type: "reset" });

  const form = requiredElement<HTMLFormElement>("generation-form");
  const prompt = requiredElement<HTMLTextAreaElement>("prompt");
  const promptError = requiredElement<HTMLElement>("prompt-error");
  const seed = requiredElement<HTMLInputElement>("seed");
  const seedError = requiredElement<HTMLElement>("seed-error");
  const randomizeSeed = requiredElement<HTMLButtonElement>("randomize-seed");
  const generate = requiredElement<HTMLButtonElement>("generate");
  const generationMenuTrigger = requiredElement<HTMLButtonElement>(
    "generation-actions-menu",
  );
  const generateHelp = requiredElement<HTMLElement>("generate-help");
  const targetBufferOutput =
    requiredElement<HTMLOutputElement>("target-buffer-output");
  const canvas = requiredElement<HTMLCanvasElement>("motion-canvas");
  const previewDiagnostics =
    requiredElement<HTMLElement>("preview-diagnostics");
  const playPause = requiredElement<HTMLButtonElement>("play-pause");
  const currentTime = requiredElement<HTMLElement>("current-time");
  const totalTime = requiredElement<HTMLElement>("total-time");
  const resetCamera = requiredElement<HTMLButtonElement>("reset-camera");
  const appStatus = requiredElement<HTMLElement>("app-status");

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

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let vrmErrorReturnFocus: HTMLElement | null = null;
  let viewer: SkeletonViewer | null = null;
  let viewerInitialization: Promise<SkeletonViewer> | null = null;
  const pressedCameraKeys = new Set<string>();
  const syncCameraMovement = (): void => {
    const [forward, right] = cameraMovementForCodes(pressedCameraKeys);
    viewer?.setCameraMovement(forward, right);
  };
  const clearCameraMovement = (): void => {
    if (pressedCameraKeys.size === 0) {
      viewer?.setCameraMovement(0, 0);
      return;
    }
    pressedCameraKeys.clear();
    viewer?.setCameraMovement(0, 0);
  };

  const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {
    type: "module",
    name: "ardy-inference",
  });

  let workerReady = false;
  let workerFailure: Error | null = null;
  let modelReady = false;
  let modelLoading = false;
  let activeLoadRequest: string | null = null;
  let activeGeneration: ActiveGeneration | null = null;
  let activeRestoreRequest: string | null = null;
  let announceContinuationRestore = true;
  let modelInfo: ModelLoadedEvent["model"] | null = null;
  let webGpuState: WebGpuState = "checking";
  let modelVariant: BrowserModelVariant | null = null;
  let pendingCapabilitiesRequest: {
    requestId: string;
    resolve: (variant: BrowserModelVariant) => void;
    reject: (error: Error) => void;
  } | null = null;
  let modelSource: ModelManifestSource | null = null;
  let activeManifest: BrowserModelManifest | null = null;
  let modelProgressFiles = new Set<string>();
  let tokenizerPreparationSteps = 0;
  let lastUserGenerationMs: number | null = null;
  let currentMotion: StructuredMotionResult | null = null;
  let currentContinuation: RuntimeContinuationState | null = null;
  let activePrompt: string | null = null;
  let playbackIntent = false;
  let currentProvenance: MotionSessionProvenance = {};
  let editorState = cloneEditorState(DEFAULT_EDITOR_STATE);
  let activeVrmLoad = 0;
  let currentVrmInfo: VrmModelInfo | null = null;
  let vrmLoading = false;
  let vrmDragDepth = 0;

  const postCommand = (command: WorkerCommand): void =>
    worker.postMessage(command);

  function requestWorkerModelVariant(): Promise<BrowserModelVariant> {
    if (workerFailure !== null) {
      return Promise.reject(workerFailure);
    }
    if (pendingCapabilitiesRequest !== null) {
      throw new Error("WebGPU capability detection is already in progress.");
    }
    const capabilityRequestId = requestId("webgpu-capabilities");
    return new Promise((resolve, reject) => {
      pendingCapabilitiesRequest = {
        requestId: capabilityRequestId,
        resolve,
        reject,
      };
      try {
        postCommand({
          type: "getWebGpuCapabilities",
          requestId: capabilityRequestId,
        });
      } catch (error) {
        pendingCapabilitiesRequest = null;
        reject(
          error instanceof Error
            ? error
            : new Error(String(error)),
        );
      }
    });
  }

  function announce(message: string): void {
    appStatus.textContent = "";
    window.requestAnimationFrame(() => {
      appStatus.textContent = message;
    });
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

  function reportInternalError(title: string, error: unknown): void {
    console.error(`[ARDY] ${title}`, error);
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
    previewSettingsTabControl.commit("view");
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
    if (prompt.dataset.validated === "true") {
      const validation = validateGenerationForm(prompt.value, seed.value);
      promptError.textContent = validation.promptError ?? "";
      setFieldInvalid(prompt, Boolean(validation.promptError));
    }
    updateGenerateAvailability();
  }

  function updateSeed(): void {
    if (seed.dataset.validated === "true") {
      const validation = validateGenerationForm(prompt.value, seed.value);
      seedError.textContent = validation.seedError ?? "";
      setFieldInvalid(seed, Boolean(validation.seedError));
    }
    updateGenerateAvailability();
  }

  function updateTargetBuffer(): void {
    const value = targetBufferControl.getSnapshot().value;
    const seconds = value / (modelInfo?.fps ?? 20);
    const valueText = `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
    targetBufferOutput.value = valueText;
    targetBufferControl.setState({ ariaValueText: valueText });
  }

  function targetFrameCount(): number {
    return integerValue(targetBufferControl.getSnapshot().value, 80, 1, 1000);
  }

  function updateGenerateAvailability(): void {
    const generating = activeGeneration !== null;
    const generationBusy = generating || activeRestoreRequest !== null;
    const promptAction = resolvePromptActionState(
      currentMotion !== null,
      currentContinuation !== null,
      prompt.value,
      activePrompt,
    );
    const continuationAvailable = canContinueGeneration(
      modelReady,
      generationBusy,
      currentMotion !== null,
      currentContinuation !== null,
    );
    const canAttempt = canAttemptGeneration(
      prompt.value,
      workerReady,
      modelReady,
      modelLoading,
      generationBusy,
    );
    const seedIsValid =
      validateGenerationForm("valid prompt", seed.value).seedError ===
      undefined;
    const primaryDisabled = !canAttempt || !promptAction.canSubmit;
    const newMotionDisabled =
      currentMotion === null || !canAttempt || !seedIsValid;
    const activeLabel =
      activeGeneration?.action === "start" ||
      activeGeneration?.action === "new-motion"
        ? "Starting…"
        : activeGeneration?.action === "update"
          ? "Updating…"
          : activeGeneration?.action === "regenerate"
            ? "Regenerating…"
            : null;
    generationActionsControl.setState({
      primaryLabel: promptAction.label,
      activeLabel,
      primaryDisabled:
        primaryDisabled ||
        (currentMotion === null && !seedIsValid),
      menuDisabled:
        generationBusy ||
        (newMotionDisabled && !continuationAvailable),
      regenerateDisabled: !continuationAvailable,
      newMotionDisabled,
    });
    generate.disabled =
      primaryDisabled ||
      (currentMotion === null && !seedIsValid);
    generationMenuTrigger.disabled =
      generationBusy ||
      (newMotionDisabled && !continuationAvailable);
    generate.dataset.dirty = String(promptAction.dirty);
    form.dataset.promptAction =
      promptAction.label === "Start motion" ? "start" : "update";

    if (generationBusy) {
      generateHelp.textContent = activeRestoreRequest
        ? "Restoring the saved generation state."
        : "Generating locally.";
    } else if (webGpuState === "checking") {
      generateHelp.textContent = "Checking WebGPU support.";
    } else if (webGpuState === "unavailable") {
      generateHelp.textContent = "This device cannot run the model.";
    } else if (modelLoading) {
      generateHelp.textContent = "Preparing the model.";
    } else if (!modelReady) {
      generateHelp.textContent = "Download the model to enable generation.";
    } else if (prompt.value.trim().length === 0) {
      generateHelp.textContent = "Describe a motion to enable generation.";
    } else if (currentMotion === null && !seedIsValid) {
      generateHelp.textContent =
        "Enter a whole-number seed from 0 to 4294967295.";
    } else if (currentMotion !== null && currentContinuation === null) {
      generateHelp.textContent =
        "Start a new motion to create a continuable session.";
    } else if (currentMotion !== null && !promptAction.dirty) {
      generateHelp.textContent =
        "Edit the motion description to update the live session.";
    } else {
      generateHelp.textContent = currentMotion
        ? "Ready to update the live motion."
        : "Ready to start motion in this browser.";
    }
  }

  function markCurrentMotionPlaybackOnly(): void {
    currentContinuation = null;
    updateGenerateAvailability();
  }

  function setGenerationBusy(): void {
    updateGenerateAvailability();
  }

  function setEditor(next: MotionEditorState): void {
    editorState = cloneEditorState(next);
    try {
      viewer?.applyEditorState(editorState);
    } catch (error) {
      reportInternalError("Could not update editor", error);
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
    playPauseControl.setState({
      pressed: playbackIntent,
      disabled: state.frameCount < 2,
    });
    playbackSpeedControl.setState({
      value: String(state.speed),
      disabled: state.frameCount === 0,
    });
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

  async function initializeViewer(): Promise<boolean> {
    viewerInitialization ??= SkeletonViewer.create(canvas);
    const initializedViewer = await viewerInitialization;
    if (disposed) {
      initializedViewer.dispose();
      return false;
    }
    if (viewer === null) {
      viewer = initializedViewer;
      viewer.setReducedMotion(reducedMotion.matches);
      viewer.onPlaybackChange = updatePlayback;
      viewer.applyEditorState(editorState);
      viewer.setOutputVisibility(outputVisibilityFromControls());
      viewer.setLoop(false);
      setVrmLoading(false);
    }
    return true;
  }

  function loadModel(): void {
    if (webGpuState !== "ready") {
      throw new Error(
        "WebGPU is required. Use a supported browser and device over HTTPS or localhost.",
      );
    }
    if (!modelSource) {
      throw new Error("The model source is unavailable.");
    }
    activeLoadRequest = requestId("load");
    modelLoading = true;
    modelReady = false;
    modelProgressFiles = new Set();
    tokenizerPreparationSteps = 0;
    modelUiControl.dispatch({ type: "runtime-loading" });
    if (modelUiControl.getSnapshot().cache === "ready") {
      modelUiControl.dispatch({ type: "initialization-started" });
    }
    updateGenerateAvailability();
    announce(
      `Preparing ${formatBytes(modelTransportSize(modelSource.manifest))} of model files.`,
    );
    const command: LoadModelCommand = {
      type: "loadModel",
      requestId: activeLoadRequest,
      baseUrl: modelSource.baseUrl,
    };
    postCommand(command);
  }

  function handleProgress(event: ProgressEvent): void {
    if (event.requestId === activeLoadRequest) {
      if (event.stage === "downloading-model") {
        if (event.message) modelProgressFiles.add(event.message);
        const snapshot = modelUiControl.getSnapshot();
        modelUiControl.dispatch({
          type: "download-progress",
          cachedFiles: Math.min(
            snapshot.totalFiles,
            modelProgressFiles.size,
          ),
          totalFiles: snapshot.totalFiles,
          cachedBytes: event.completed,
          totalBytes: event.total,
        });
        if (event.total > 0 && event.completed >= event.total) {
          modelUiControl.dispatch({ type: "download-completed" });
        }
      } else if (event.stage === "verifying-model") {
        modelUiControl.dispatch({
          type: "verification-progress",
          completedFiles: event.completed,
          totalFiles: event.total,
        });
      } else if (event.stage === "loading-tokenizer") {
        tokenizerPreparationSteps = Math.max(
          0,
          Math.floor(event.total),
        );
        const sessionSteps = modelSource
          ? Object.keys(modelSource.manifest.graphs).length
          : 0;
        modelUiControl.dispatch({ type: "runtime-loading" });
        modelUiControl.dispatch({
          type: "initialization-progress",
          completedSteps: event.completed,
          totalSteps: tokenizerPreparationSteps + sessionSteps,
        });
      } else if (event.stage === "loading-sessions") {
        modelUiControl.dispatch({ type: "runtime-loading" });
        modelUiControl.dispatch({
          type: "initialization-progress",
          completedSteps: tokenizerPreparationSteps + event.completed,
          totalSteps: tokenizerPreparationSteps + event.total,
        });
      }
      return;
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
      seed: 2,
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
    modelUiControl.dispatch({ type: "runtime-ready" });
    if (modelSource) {
      void inspectModelCache(modelSource)
        .then((cache) => {
          if (disposed) return;
          modelUiControl.dispatch(
            cache.complete
              ? {
                  type: "cache-ready",
                  totalFiles: cache.fileCount,
                  totalBytes: cache.transportSizeBytes,
                }
              : {
                  type: "cache-missing",
                  cachedFiles: cache.cachedFileCount,
                  totalFiles: cache.fileCount,
                  cachedBytes: cache.cachedTransportSizeBytes,
                  totalBytes: cache.transportSizeBytes,
                },
          );
        })
        .catch((error) => {
          if (disposed) return;
          modelUiControl.dispatch({
            type: "cache-error",
            operation: "download",
          });
          reportInternalError(
            "Could not refresh model cache status",
            error,
          );
        });
    }
    const incompatibleContinuation =
      currentContinuation !== null &&
      !isContinuationModelCompatible(currentProvenance, event.model);
    if (incompatibleContinuation) {
      markCurrentMotionPlaybackOnly();
    }
    updateGenerateAvailability();
    announce(
      incompatibleContinuation
        ? `${event.model.variant || "ARDY Mini Core40"} is ready on WebGPU. The displayed motion remains playback-only because its continuation belongs to a different model revision.`
        : `${event.model.variant || "ARDY Mini Core40"} is ready on WebGPU.`,
    );
    if (currentContinuation) {
      restoreWorkerContinuation();
    } else {
      resetWorkerSession();
    }
  }

  async function reconcileCancelledModelLoad(): Promise<void> {
    if (!modelSource) {
      modelUiControl.dispatch({
        type: "cache-error",
        operation: "download",
      });
      return;
    }
    try {
      const cache = await inspectModelCache(modelSource);
      if (disposed) return;
      modelUiControl.dispatch({
        type: "cache-missing",
        cachedFiles: cache.cachedFileCount,
        totalFiles: cache.fileCount,
        cachedBytes: cache.cachedTransportSizeBytes,
        totalBytes: cache.transportSizeBytes,
      });
      announce(
        cache.cachedTransportSizeBytes > 0
          ? "Model download paused. Downloaded data was kept in this browser."
          : "Model download paused.",
      );
    } catch (error) {
      if (disposed) return;
      modelUiControl.dispatch({
        type: "cache-error",
        operation: "download",
      });
      reportInternalError(
        "Could not inspect the paused model download",
        error,
      );
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
    viewer.setLoop(false);
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
      modelRevision: modelInfo?.revision,
      modelVariant: modelInfo?.variant,
      createdAt: new Date().toISOString(),
    };
    const preserveFrame =
      active.mode === "replace" && chunk.startFrame === 0
        ? 0
        : (previousPlayback?.frame ?? 0);
    const preservePlaying =
      active.mode === "replace"
        ? active.resumePlayback
        : Boolean(previousPlayback?.playing || active.resumePlayback);
    if (active.mode === "replace") {
      playbackIntent = preservePlaying;
    }
    const resetPresentation = shouldResetMotionPresentation(
      active.mode,
      chunk.startFrame === 0,
    );
    refreshViewer(
      preserveFrame,
      preservePlaying,
      resetPresentation,
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
      if (active.action !== "extend") {
        lastUserGenerationMs = Math.round(event.result.timingsMs.total);
      }
      previewDiagnostics.textContent =
        `${event.sessionFrameCount} frames` +
        (lastUserGenerationMs === null
          ? ""
          : ` · ${lastUserGenerationMs} ms`);
      activeGeneration = null;
      setGenerationBusy();
      announce(
        `Generation complete. The session contains ${event.sessionFrameCount} frames.`,
      );
      updateGenerateAvailability();
    } catch (error) {
      activeGeneration = null;
      setGenerationBusy();
      reportInternalError("Could not assemble generated motion", error);
    }
  }

  function validateFormForGeneration(
    mode: GenerationMode,
    promptValue: string,
  ): FormValues | null {
    const continuationSeed =
      currentContinuation?.random.seed ?? currentProvenance.seed ?? 0;
    const validation = validateGenerationForm(
      promptValue,
      mode === "replace" ? seed.value : String(continuationSeed),
    );
    if (mode === "append") {
      return validation.values ?? null;
    }
    promptError.textContent = validation.promptError ?? "";
    seedError.textContent =
      mode === "replace" ? (validation.seedError ?? "") : "";
    prompt.dataset.validated = "true";
    if (mode === "replace") seed.dataset.validated = "true";
    setFieldInvalid(prompt, Boolean(validation.promptError));
    setFieldInvalid(seed, mode === "replace" && Boolean(validation.seedError));
    if (!validation.values) {
      if (validation.promptError) {
        prompt.focus();
      } else if (mode === "replace" && validation.seedError) {
        previewSettingsTabControl.commit("motion");
        previewSettingsControl.commit(true);
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
      branchFrame?: number;
      background?: boolean;
      promptValue?: string;
      action?: ActiveGeneration["action"];
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
    const promptValue =
      options.promptValue ??
      (mode === "append" ? (activePrompt ?? "") : prompt.value);
    const values = validateFormForGeneration(mode, promptValue);
    if (!values) return;
    if (mode !== "append") {
      activePrompt = values.prompt;
    }
    const background = Boolean(options.background);
    const id = requestId(mode);
    const playback = viewer?.getPlaybackState();
    activeGeneration = {
      id,
      mode,
      action:
        options.action ??
        (mode === "append"
          ? "extend"
          : mode === "branch"
            ? "update"
            : currentMotion
              ? "new-motion"
              : "start"),
      receivedChunk: false,
      resumePlayback:
        mode === "replace"
          ? shouldAutoplayMotion(reducedMotion.matches)
          : playbackIntent,
    };
    setGenerationBusy();
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
      // The protocol keeps this field required, but append and branch retain
      // the continuation RNG rather than reading the replace-only seed input.
      seed:
        mode === "replace"
          ? values.seed
          : (currentContinuation?.random.seed ?? currentProvenance.seed ?? 0),
      durationFrames: options.durationFrames ?? targetFrameCount(),
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
      !playbackIntent ||
      !modelReady ||
      activeGeneration ||
      !currentMotion ||
      !currentContinuation ||
      !activePrompt ||
      state.frameCount === 0
    ) {
      return;
    }
    const remaining = state.frameCount - state.frame - 1;
    const threshold = Math.min(
      DEFAULT_REPLAN_THRESHOLD_FRAMES,
      Math.max(1, targetBufferControl.getSnapshot().value),
    );
    if (remaining > threshold) return;
    startGeneration("append", {
      durationFrames: targetFrameCount(),
      background: true,
      promptValue: activePrompt,
      action: "extend",
    });
  }

  function setPlaybackIntent(playing: boolean): void {
    playbackIntent = playing;
    if (activeGeneration) {
      activeGeneration.resumePlayback = playing;
    }
    viewer?.setPlaying(playing);
    const state = viewer?.getPlaybackState();
    if (playing && state) maybeAutoReplan(state);
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
        case "webGpuCapabilities": {
          if (
            event.requestId !== pendingCapabilitiesRequest?.requestId
          ) {
            break;
          }
          const request = pendingCapabilitiesRequest;
          pendingCapabilitiesRequest = null;
          request.resolve(preferredModelVariant(event.shaderF16));
          break;
        }
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
              reportInternalError(
                "Could not display generation chunk",
                error,
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
          if (event.targetRequestId === activeLoadRequest) {
            activeLoadRequest = null;
            modelLoading = false;
            modelReady = false;
            modelUiControl.dispatch({ type: "runtime-idle" });
            void reconcileCancelledModelLoad();
            updateGenerateAvailability();
          } else if (event.targetRequestId === activeGeneration?.id) {
            activeGeneration = null;
            markCurrentMotionPlaybackOnly();
            setGenerationBusy();
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
            modelUiControl.dispatch({ type: "runtime-idle" });
          } else if (
            event.status.state === "ready" ||
            event.status.state === "generating"
          ) {
            modelReady = true;
            modelUiControl.dispatch({ type: "runtime-ready" });
          }
          updateGenerateAvailability();
          break;
        case "error": {
          const wasLoading = event.requestId === activeLoadRequest;
          const wasGenerating = event.requestId === activeGeneration?.id;
          const wasRestoring = event.requestId === activeRestoreRequest;
          if (
            event.requestId === pendingCapabilitiesRequest?.requestId
          ) {
            const request = pendingCapabilitiesRequest;
            pendingCapabilitiesRequest = null;
            request.reject(new Error(event.error.message));
            break;
          }
          if (wasLoading) {
            const loadErrorOperation =
              modelUiControl.getSnapshot().cache === "downloading"
                ? "download"
                : "initialization";
            activeLoadRequest = null;
            modelLoading = false;
            modelReady = false;
            modelUiControl.dispatch({ type: "runtime-error" });
            modelUiControl.dispatch({
              type: "cache-error",
              operation: loadErrorOperation,
            });
          }
          if (wasGenerating) {
            activeGeneration = null;
            markCurrentMotionPlaybackOnly();
            setGenerationBusy();
          }
          if (wasRestoring) {
            activeRestoreRequest = null;
            announceContinuationRestore = true;
            markCurrentMotionPlaybackOnly();
            resetWorkerSession();
            announce(
              "The saved continuation is incompatible with this model revision. The motion remains available for playback only.",
            );
          }
          if (wasLoading) {
            reportInternalError("Model loading failed", event.error);
            announce("The model could not be prepared. Try again.");
          } else if (wasRestoring) {
            reportInternalError(
              "Continuation unavailable",
              `${event.error.message} The imported motion is still available for playback.`,
            );
          } else {
            reportInternalError("Inference failed", event.error);
          }
          updateGenerateAvailability();
          break;
        }
        case "disposed":
          modelReady = false;
          modelUiControl.dispatch({ type: "runtime-idle" });
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
      workerFailure = new Error(
        event.message || "The inference worker stopped unexpectedly.",
      );
      if (pendingCapabilitiesRequest !== null) {
        const request = pendingCapabilitiesRequest;
        pendingCapabilitiesRequest = null;
        request.reject(workerFailure);
      }
      const wasLoading = activeLoadRequest !== null;
      activeLoadRequest = null;
      modelLoading = false;
      modelReady = false;
      activeGeneration = null;
      setGenerationBusy();
      modelUiControl.dispatch({ type: "runtime-error" });
      const cacheState = modelUiControl.getSnapshot().cache;
      if (
        wasLoading ||
        cacheState === "downloading" ||
        cacheState === "cancelling" ||
        cacheState === "verifying" ||
        cacheState === "initializing"
      ) {
        modelUiControl.dispatch({
          type: "cache-error",
          operation:
            cacheState === "downloading"
              ? "download"
              : "initialization",
        });
      }
      reportInternalError(
        "Inference worker stopped",
        event.message || "An unexpected worker error occurred.",
      );
      unsupportedDeviceControl.commit({
        open: true,
        title: "Inference runtime unavailable",
        description:
          "The local inference worker stopped unexpectedly. Reload the page to try again.",
      });
      announce("The inference worker stopped. Reload the page to try again.");
    },
    { signal: lifecycle.signal },
  );

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!currentMotion) {
      startGeneration("replace", { action: "start" });
      return;
    }
    if (!currentContinuation) {
      updateGenerateAvailability();
      return;
    }
    const playback = viewer?.getPlaybackState();
    startGeneration("branch", {
      branchFrame: livePromptBranchFrame(
        playback?.frame ?? 0,
        currentMotion.frameCount,
      ),
      durationFrames: targetFrameCount(),
      background: playbackIntent,
      action: "update",
    });
  });

  prompt.addEventListener("input", updatePrompt);
  seed.addEventListener("input", updateSeed);
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

  modelDownloadAction.onTrigger(async () => {
    if (modelLoading || activeLoadRequest) return;
    try {
      modelUiControl.dispatch({ type: "cache-check-started" });
      const cache = await discoverModelSource();
      if (cache.complete && modelReady) return;
      if (!cache.complete) {
        const remainingBytes = Math.max(
          0,
          cache.transportSizeBytes - cache.cachedTransportSizeBytes,
        );
        const estimate = await navigator.storage?.estimate?.();
        const available =
          estimate?.quota === undefined
            ? undefined
            : estimate.quota - (estimate.usage ?? 0);
        if (
          available !== undefined &&
          available < remainingBytes * 1.05
        ) {
          throw new Error(
            `The remaining model files need ${formatBytes(remainingBytes)}, but only about ${formatBytes(Math.max(0, available))} is available.`,
          );
        }
        await navigator.storage?.persist?.();
        modelUiControl.dispatch({ type: "download-started" });
      }
      loadModel();
    } catch (error) {
      modelUiControl.dispatch({
        type: "cache-error",
        operation: "download",
      });
      modelUiControl.dispatch({ type: "runtime-error" });
      reportInternalError("Could not download model files", error);
      announce("The model download could not start. Try again.");
      updateGenerateAvailability();
    }
  }, lifecycle.signal);

  clearModelCacheAction.onTrigger(async () => {
    try {
      await clearModelCache();
      modelUiControl.dispatch({ type: "cache-cleared" });
      updateGenerateAvailability();
      announce(
        modelReady
          ? "Cached model files removed. The loaded model remains available in this tab."
          : "Cached model files removed.",
      );
    } catch (error) {
      modelUiControl.dispatch({
        type: "cache-error",
        operation: "clear",
      });
      reportInternalError("Could not clear model cache", error);
      announce("The model cache could not be cleared. Try again.");
    }
  }, lifecycle.signal);

  modelDownloadCancelAction.onTrigger(() => {
    if (
      !modelLoading ||
      !activeLoadRequest ||
      modelUiControl.getSnapshot().cache !== "downloading"
    ) {
      return;
    }
    modelUiControl.dispatch({ type: "download-cancel-requested" });
    postCommand({
      type: "cancel",
      requestId: requestId("cancel"),
      targetRequestId: activeLoadRequest,
    });
    announce("Stopping the model download.");
  }, lifecycle.signal);

  startNewMotionAction.onTrigger(() => {
    startGeneration("replace", { action: "new-motion" });
  }, lifecycle.signal);
  regenerateMotionAction.onTrigger(() => {
    const frame = viewer?.getPlaybackState().frame ?? 0;
    startGeneration("branch", {
      branchFrame: frame,
      durationFrames: targetFrameCount(),
      action: "regenerate",
    });
  }, lifecycle.signal);

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
  window.addEventListener(
    "blur",
    () => {
      resetVrmDropTarget();
      clearCameraMovement();
    },
    {
      signal: lifecycle.signal,
    },
  );
  canvas.addEventListener("blur", clearCameraMovement, {
    signal: lifecycle.signal,
  });
  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.hidden) clearCameraMovement();
    },
    {
      signal: lifecycle.signal,
    },
  );
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

  dismissVrmError.addEventListener("click", () => clearVrmError(true));
  playPause.addEventListener("click", () => {
    setPlaybackIntent(!playbackIntent);
  });
  timelineControl.onCommit((value) => viewer?.seek(value), lifecycle.signal);
  playbackSpeedControl.onCommit(
    (value) => viewer?.setSpeed(Number(value)),
    lifecycle.signal,
  );
  resetCamera.addEventListener("click", (event) =>
    viewer?.resetCamera({ animated: event.detail !== 0 }),
  );

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    viewer?.setReducedMotion(event.matches);
    if (event.matches) {
      setPlaybackIntent(false);
      viewer?.setLoop(false);
      announce("Reduced motion enabled. Playback is paused.");
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
      if (
        pressedCameraKeys.size > 0 &&
        (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey)
      ) {
        clearCameraMovement();
      }
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
        if (!pressedCameraKeys.has(event.code)) {
          pressedCameraKeys.add(event.code);
          syncCameraMovement();
        }
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
        setPlaybackIntent(!playbackIntent);
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
        viewer?.resetCamera({ animated: false });
      }
    },
    { signal: lifecycle.signal },
  );
  document.addEventListener(
    "keyup",
    (event) => {
      if (!cameraMoveForCode(event.code)) return;
      if (!pressedCameraKeys.delete(event.code)) return;
      event.preventDefault();
      syncCameraMovement();
    },
    { signal: lifecycle.signal },
  );

  const cleanup = (): void => {
    if (disposed) return;
    disposed = true;
    if (pendingCapabilitiesRequest !== null) {
      const request = pendingCapabilitiesRequest;
      pendingCapabilitiesRequest = null;
      request.reject(new DOMException("Application disposed", "AbortError"));
    }
    activeVrmLoad += 1;
    resetVrmDropTarget();
    clearCameraMovement();
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

  async function initializeModel(): Promise<void> {
    const preflight = await inspectWebGpuSupport();
    if (disposed) return;
    if (preflight.unavailableReason !== null) {
      webGpuState = "unavailable";
      modelUiControl.dispatch({ type: "runtime-error" });
      updateGenerateAvailability();
      unsupportedDeviceControl.commit({
        open: true,
        title: "WebGPU is required",
        description: preflight.unavailableReason,
      });
      return;
    }

    try {
      if (!(await initializeViewer())) return;
    } catch (error) {
      if (disposed) return;
      const message = error instanceof Error ? error.message : String(error);
      webGpuState = "unavailable";
      modelUiControl.dispatch({ type: "runtime-error" });
      updateGenerateAvailability();
      unsupportedDeviceControl.commit({
        open: true,
        title: "WebGPU preview unavailable",
        description: message,
      });
      return;
    }

    try {
      modelVariant = await requestWorkerModelVariant();
    } catch (error) {
      if (disposed) return;
      const message = error instanceof Error ? error.message : String(error);
      webGpuState = "unavailable";
      modelUiControl.dispatch({ type: "runtime-error" });
      updateGenerateAvailability();
      unsupportedDeviceControl.commit({
        open: true,
        title: "WebGPU inference unavailable",
        description: message,
      });
      return;
    }

    webGpuState = "ready";
    modelUiControl.dispatch({ type: "cache-check-started" });
    updateGenerateAvailability();
    try {
      const cache = await discoverModelSource();
      if (disposed) return;
      if (cache.complete) {
        loadModel();
      }
    } catch (error) {
      if (disposed) return;
      modelUiControl.dispatch({
        type: "cache-error",
        operation: "download",
      });
      modelUiControl.dispatch({ type: "runtime-error" });
      reportInternalError("Could not initialize model files", error);
      announce("The model files are unavailable. Check the model source and retry.");
      updateGenerateAvailability();
    }
  }

  async function discoverModelSource(): Promise<ModelCacheStatus> {
    const familyBaseUrl = configuredModelFamilyBaseUrl();
    if (!familyBaseUrl) {
      throw new Error(
        "Set VITE_MODEL_BASE_URL to an immutable hosted model-family directory for production.",
      );
    }
    if (!modelVariant) {
      throw new Error("WebGPU model selection has not completed.");
    }
    const baseUrl = modelVariantBaseUrl(familyBaseUrl, modelVariant);
    modelSource = await fetchModelManifest(baseUrl, {
      signal: lifecycle.signal,
    });
    const cache = await inspectModelCache(modelSource);
    modelUiControl.dispatch(
      cache.complete
        ? {
            type: "cache-ready",
            totalFiles: cache.fileCount,
            totalBytes: cache.transportSizeBytes,
          }
        : {
            type: "cache-missing",
            cachedFiles: cache.cachedFileCount,
            totalFiles: cache.fileCount,
            cachedBytes: cache.cachedTransportSizeBytes,
            totalBytes: cache.transportSizeBytes,
          },
    );
    updateGenerateAvailability();
    return cache;
  }

  updatePrompt();
  updateSeed();
  updateTargetBuffer();
  syncOutputVisibility();
  setVrmStatus(null);
  setVrmLoading(false);
  updateGenerateAvailability();
  void initializeModel();
  return cleanup;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import "./style.css";

import type {
  GenerateCommand,
  GenerationResultEvent,
  LoadModelPackCommand,
  ModelLoadedEvent,
  ProgressEvent,
  RuntimeBackendPreference,
  WorkerCommand,
  WorkerEvent,
} from "./runtime/protocol";
import {
  CORE27_JOINT_COUNT,
  normalizeMotionClip,
  SkeletonViewer,
  type PlaybackState,
} from "./viewer";

const UINT32_MAX = 0xffff_ffff;
const ALLOWED_DURATIONS = new Set([2, 4, 6, 8, 10]);
const CACHE_ROOT = "ardy-mini-model-cache";
const CACHE_PACK = "active-pack";
const CACHE_INDEX = "index.json";
const PACK_MANIFEST = "manifest.json";
const ORT_WASM_PATH = new URL("ort/", document.baseURI).href;

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

export function validateGenerationForm(promptValue: string, durationValue: string, seedValue: string): FormValidation {
  const prompt = promptValue.trim();
  const durationSeconds = Number(durationValue);
  const seed = Number(seedValue);
  const validation: FormValidation = {};

  if (!prompt) {
    validation.promptError = "Describe the motion you want to generate.";
  } else if (prompt.length > 280) {
    validation.promptError = "Keep the prompt to 280 characters or fewer.";
  } else if (!/^[\x09\x0a\x0d\x20-\x7e]+$/.test(prompt)) {
    validation.promptError = "This model supports typo-free English prompts using standard Latin characters.";
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
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

function canonicalPackPath(file: FileWithRelativePath): string {
  return (file.webkitRelativePath || file.name).replaceAll("\\", "/").replace(/^\.\/+/, "");
}

/**
 * Strip the folder selected by `<input webkitdirectory>` while retaining all
 * paths below the pack's manifest. `File.name` carries the canonical path
 * through structured clone for files returned by showDirectoryPicker().
 */
export function canonicalizePackFiles(input: readonly File[]): File[] {
  if (input.length === 0) throw new Error("The selected model-pack folder is empty.");
  const entries = input.map((file) => ({ file, path: canonicalPackPath(file) }));
  const manifestPaths = entries.filter(({ path }) => path === PACK_MANIFEST || path.endsWith(`/${PACK_MANIFEST}`));
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
      throw new Error("Every model-pack file must be inside the folder containing manifest.json.");
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

function requiredElement<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required UI element #${id}.`);
  return element as unknown as T;
}

function requestId(prefix: string): string {
  const suffix = typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  return `${prefix}-${suffix}`;
}

function humanizeStage(stage: ProgressEvent["stage"]): string {
  const labels: Record<ProgressEvent["stage"], string> = {
    "reading-pack": "Reading model pack",
    "hashing-pack": "Verifying model files",
    "loading-tokenizer": "Loading tokenizer",
    "loading-sessions": "Preparing inference sessions",
    "encoding-text": "Encoding motion prompt",
    denoising: "Generating motion",
    decoding: "Decoding skeleton",
  };
  return labels[stage];
}

function progressFraction(event: ProgressEvent): number {
  if (!(event.total > 0)) return 0;
  return Math.max(0, Math.min(1, event.completed / event.total));
}

function generationProgress(event: ProgressEvent): number {
  const local = progressFraction(event);
  if (event.stage === "encoding-text") return local * 0.08;
  if (event.stage === "denoising") return 0.08 + local * 0.84;
  if (event.stage === "decoding") return 0.92 + local * 0.08;
  return local;
}

function modelLoadProgress(event: ProgressEvent): number {
  const local = progressFraction(event);
  const ranges: Partial<Record<ProgressEvent["stage"], readonly [number, number]>> = {
    "reading-pack": [0, 0.18],
    "hashing-pack": [0.18, 0.38],
    "loading-tokenizer": [0.38, 0.48],
    "loading-sessions": [0.48, 1],
  };
  const range = ranges[event.stage];
  return range ? range[0] + local * (range[1] - range[0]) : local;
}

export function shouldAutoplayMotion(reducedMotion: boolean): boolean {
  return !reducedMotion;
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

async function collectDirectoryFiles(
  directory: FileSystemDirectoryHandle,
  prefix = "",
): Promise<File[]> {
  const files: File[] = [];
  for await (const [name, handle] of directory.entries()) {
    const path = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === "directory") {
      files.push(...(await collectDirectoryFiles(handle as FileSystemDirectoryHandle, path)));
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

async function getCacheRoot(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  if (!supportsPersistentPackCache()) return null;
  const opfs = await navigator.storage.getDirectory();
  try {
    return await opfs.getDirectoryHandle(CACHE_ROOT, { create });
  } catch (error) {
    if (!create && error instanceof DOMException && error.name === "NotFoundError") return null;
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

async function writeTextFile(directory: FileSystemDirectoryHandle, name: string, text: string): Promise<void> {
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
    if (!(error instanceof DOMException && error.name === "NotFoundError")) throw error;
  }
  const packDirectory = await root.getDirectoryHandle(CACHE_PACK, { create: true });
  let completedBytes = 0;
  const index: CacheIndex = {
    schemaVersion: 1,
    files: [],
  };

  for (const file of files) {
    const path = file.name;
    const parts = path.split("/");
    const filename = parts.pop();
    if (!filename) throw new Error(`Invalid model-pack path ${path}.`);
    const directory = await ensureDirectory(packDirectory, parts);
    const handle = await directory.getFileHandle(filename, { create: true });
    const writable = await handle.createWritable();
    await writable.write(file);
    await writable.close();
    completedBytes += file.size;
    index.files.push({
      path,
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
    const index = JSON.parse(await (await indexHandle.getFile()).text()) as CacheIndex;
    if (index.schemaVersion !== 1 || !Array.isArray(index.files) || index.files.length === 0) return null;
    const packDirectory = await root.getDirectoryHandle(CACHE_PACK);
    const files: File[] = [];
    for (const entry of index.files) {
      if (!entry || typeof entry.path !== "string") return null;
      const parts = entry.path.split("/");
      const filename = parts.pop();
      if (!filename || parts.some((part) => !part || part === "." || part === "..")) return null;
      let directory = packDirectory;
      for (const part of parts) directory = await directory.getDirectoryHandle(part);
      const stored = await (await directory.getFileHandle(filename)).getFile();
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
  const opfs = supportsPersistentPackCache() ? await navigator.storage.getDirectory() : null;
  if (!opfs) return;
  try {
    await opfs.removeEntry(CACHE_ROOT, { recursive: true });
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "NotFoundError")) throw error;
  }
}

function bootstrap(): void {
  if (!document.getElementById("app")) return;

  const form = requiredElement<HTMLFormElement>("generation-form");
  const prompt = requiredElement<HTMLTextAreaElement>("prompt");
  const promptCount = requiredElement<HTMLElement>("prompt-count");
  const promptError = requiredElement<HTMLElement>("prompt-error");
  const duration = requiredElement<HTMLInputElement>("duration");
  const durationOutput = requiredElement<HTMLOutputElement>("duration-output");
  const seed = requiredElement<HTMLInputElement>("seed");
  const seedError = requiredElement<HTMLElement>("seed-error");
  const randomizeSeed = requiredElement<HTMLButtonElement>("randomize-seed");
  const runtimeSettings = requiredElement<HTMLDetailsElement>("runtime-settings");
  const backend = requiredElement<HTMLSelectElement>("backend");
  const backendHelp = requiredElement<HTMLElement>("backend-help");
  const importModel = requiredElement<HTMLButtonElement>("import-model");
  const importModelLabel = requiredElement<HTMLElement>("import-model-label");
  const removeModel = requiredElement<HTMLButtonElement>("remove-model");
  const fileInput = requiredElement<HTMLInputElement>("model-file-input");
  const modelCard = requiredElement<HTMLElement>("model-card");
  const modelTitle = requiredElement<HTMLElement>("model-title");
  const modelDetail = requiredElement<HTMLElement>("model-detail");
  const modelState = requiredElement<HTMLElement>("model-state");
  const modelProgress = requiredElement<HTMLElement>("model-progress");
  const modelProgressbar = requiredElement<HTMLElement>("model-progressbar");
  const modelProgressFill = requiredElement<HTMLElement>("model-progress-fill");
  const modelProgressLabel = requiredElement<HTMLElement>("model-progress-label");
  const modelErrorBanner = requiredElement<HTMLElement>("model-error-banner");
  const modelErrorTitle = requiredElement<HTMLElement>("model-error-title");
  const modelErrorMessage = requiredElement<HTMLElement>("model-error-message");
  const dismissModelError = requiredElement<HTMLButtonElement>("dismiss-model-error");
  const generate = requiredElement<HTMLButtonElement>("generate");
  const generateLabel = requiredElement<HTMLElement>("generate-label");
  const generateHelp = requiredElement<HTMLElement>("generate-help");
  const buttonShortcut = requiredElement<HTMLElement>("button-shortcut");
  const cancelGeneration = requiredElement<HTMLButtonElement>("cancel-generation");
  const generationProgressElement = requiredElement<HTMLElement>("generation-progress");
  const generationStage = requiredElement<HTMLElement>("generation-stage");
  const generationPercent = requiredElement<HTMLElement>("generation-percent");
  const generationProgressbar = requiredElement<HTMLElement>("generation-progressbar");
  const generationProgressFill = requiredElement<HTMLElement>("generation-progress-fill");
  const errorBanner = requiredElement<HTMLElement>("error-banner");
  const errorTitle = requiredElement<HTMLElement>("error-title");
  const errorMessage = requiredElement<HTMLElement>("error-message");
  const dismissError = requiredElement<HTMLButtonElement>("dismiss-error");
  const canvas = requiredElement<HTMLCanvasElement>("motion-canvas");
  const emptyState = requiredElement<HTMLElement>("empty-state");
  const loadingOverlay = requiredElement<HTMLElement>("loading-overlay");
  const loadingTitle = requiredElement<HTMLElement>("loading-title");
  const loadingDetail = requiredElement<HTMLElement>("loading-detail");
  const motionBadge = requiredElement<HTMLElement>("motion-badge");
  const runtimeMetric = requiredElement<HTMLElement>("runtime-metric");
  const runtimeValue = requiredElement<HTMLElement>("runtime-value");
  const playPause = requiredElement<HTMLButtonElement>("play-pause");
  const timeline = requiredElement<HTMLInputElement>("timeline");
  const currentTime = requiredElement<HTMLElement>("current-time");
  const totalTime = requiredElement<HTMLElement>("total-time");
  const playbackSpeed = requiredElement<HTMLSelectElement>("playback-speed");
  const loopToggle = requiredElement<HTMLButtonElement>("loop-toggle");
  const resetCamera = requiredElement<HTMLButtonElement>("reset-camera");
  const gpuDot = requiredElement<HTMLElement>("gpu-dot");
  const gpuLabel = requiredElement<HTMLElement>("gpu-label");
  const isolationDot = requiredElement<HTMLElement>("isolation-dot");
  const isolationLabel = requiredElement<HTMLElement>("isolation-label");
  const appStatus = requiredElement<HTMLElement>("app-status");
  const viewportPanel = requiredElement<HTMLElement>("viewport-panel");
  const viewport = requiredElement<HTMLElement>("viewport");
  const playbackBar = requiredElement<HTMLElement>("playback-bar");

  fileInput.setAttribute("webkitdirectory", "");
  fileInput.setAttribute("directory", "");

  const hasWebGpu = "gpu" in navigator;
  const webGpuOption = backend.querySelector<HTMLOptionElement>('option[value="webgpu"]');
  if (webGpuOption) webGpuOption.disabled = !hasWebGpu;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  buttonShortcut.textContent = /Mac|iPhone|iPad|iPod/.test(navigator.platform) ? "⌘ ↵" : "Ctrl ↵";
  gpuDot.dataset.state = hasWebGpu ? "available" : "unavailable";
  gpuLabel.textContent = hasWebGpu ? "WebGPU available" : "WebGPU unavailable";
  isolationDot.dataset.state = crossOriginIsolated ? "available" : "unavailable";
  isolationLabel.textContent = crossOriginIsolated ? "WASM threads ready" : "WASM single-thread";

  let errorReturnFocus: HTMLElement | null = null;
  let modelErrorReturnFocus: HTMLElement | null = null;
  let viewer: SkeletonViewer | null = null;
  try {
    viewer = new SkeletonViewer(canvas);
    viewer.setReducedMotion(reducedMotion.matches);
  } catch (error) {
    showError("3D preview unavailable", error instanceof Error ? error.message : String(error));
  }

  const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {
    type: "module",
    name: "ardy-inference",
  });

  let workerReady = false;
  let modelReady = false;
  let modelLoading = false;
  let modelCaching = false;
  let generating = false;
  let activeLoadRequest: string | null = null;
  let activeGenerationRequest: string | null = null;
  let lastPackFiles: File[] | null = null;
  let packPendingCache = false;
  let cachedPack = false;
  let modelLabel = "";
  let modelBackend = "";
  let generationProgressValue = 0;
  let modelProgressValue = 0;
  let loadingOverlayTimer = 0;

  const postCommand = (command: WorkerCommand): void => worker.postMessage(command);

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
    fill.style.transform = `scaleX(${safeFraction})`;
    progressbar.setAttribute("aria-valuenow", String(percent));
    return percent;
  }

  function updateGenerateAvailability(): void {
    const hasPrompt = prompt.value.trim().length > 0;
    generate.disabled = !canAttemptGeneration(
      prompt.value,
      workerReady,
      modelReady,
      modelLoading,
      generating,
    );
    backend.disabled = modelLoading || modelCaching || generating;
    importModel.disabled = modelLoading || modelCaching || generating;
    removeModel.disabled = modelLoading || modelCaching || generating;
    if (generating) {
      generateHelp.textContent = "Generation is running locally. You can cancel after the current inference step.";
    } else if (modelLoading) {
      generateHelp.textContent = "Preparing the model. Generation will be available when loading finishes.";
    } else if (modelCaching) {
      generateHelp.textContent = "Ready to generate. A local browser copy is still being saved.";
    } else if (!modelReady) {
      generateHelp.textContent = "Load a model pack to enable generation.";
    } else if (!hasPrompt) {
      generateHelp.textContent = "Enter an English motion prompt to continue.";
    } else {
      generateHelp.textContent = "Ready to generate locally in this tab.";
    }
  }

  function setModelStatus(
    state: "missing" | "loading" | "ready",
    title: string,
    detail: string,
    badge: string,
  ): void {
    modelCard.dataset.state = state;
    modelState.dataset.state = state;
    modelState.textContent = badge;
    modelTitle.textContent = title;
    modelDetail.textContent = detail;
    form.dataset.modelState = state;
    importModelLabel.textContent = state === "ready" ? "Replace model-pack folder" : "Choose model-pack folder";
  }

  function showError(title: string, message: string, focus = true): void {
    const activeElement = document.activeElement;
    if (activeElement instanceof HTMLElement && !errorBanner.contains(activeElement)) {
      errorReturnFocus = activeElement;
    }
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorBanner.hidden = false;
    if (focus) window.requestAnimationFrame(() => errorBanner.focus());
  }

  function clearError(restoreFocus = false): void {
    const focusWasInside = errorBanner.contains(document.activeElement);
    errorBanner.hidden = true;
    errorMessage.textContent = "";
    if ((restoreFocus || focusWasInside) && errorReturnFocus?.isConnected) errorReturnFocus.focus();
    errorReturnFocus = null;
  }

  function showModelError(title: string, message: string, focus = true): void {
    const activeElement = document.activeElement;
    if (focus && activeElement instanceof HTMLElement && !modelErrorBanner.contains(activeElement)) {
      modelErrorReturnFocus = activeElement;
    }
    modelErrorTitle.textContent = title;
    modelErrorMessage.textContent = message;
    modelErrorBanner.hidden = false;
    if (focus) window.requestAnimationFrame(() => modelErrorBanner.focus());
  }

  function clearModelError(restoreFocus = false): void {
    const focusWasInside = modelErrorBanner.contains(document.activeElement);
    modelErrorBanner.hidden = true;
    modelErrorMessage.textContent = "";
    if ((restoreFocus || focusWasInside) && modelErrorReturnFocus?.isConnected) {
      modelErrorReturnFocus.focus();
    }
    modelErrorReturnFocus = null;
  }

  function resetRuntimeBadge(): void {
    gpuDot.dataset.state = hasWebGpu ? "available" : "unavailable";
    gpuLabel.textContent = hasWebGpu ? "WebGPU available" : "WebGPU unavailable";
  }

  function updateRangeProgress(input: HTMLInputElement): void {
    const min = Number(input.min);
    const max = Number(input.max);
    const progress = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--range-progress", `${progress}%`);
  }

  function updatePrompt(): void {
    promptCount.textContent = `${prompt.value.length} / 280`;
    prompt.removeAttribute("aria-invalid");
    promptError.textContent = "";
    updateGenerateAvailability();
  }

  function updateDuration(): void {
    durationOutput.textContent = `${duration.value} seconds`;
    duration.setAttribute("aria-valuetext", `${duration.value} seconds`);
    updateRangeProgress(duration);
    updateGenerateAvailability();
  }

  function updateSeed(): void {
    seed.removeAttribute("aria-invalid");
    seedError.textContent = "";
    updateGenerateAvailability();
  }

  function setGenerationBusy(busy: boolean): void {
    generating = busy;
    generationProgressElement.hidden = !busy;
    cancelGeneration.hidden = !busy;
    prompt.disabled = busy;
    duration.disabled = busy;
    seed.disabled = busy;
    randomizeSeed.disabled = busy;
    window.clearTimeout(loadingOverlayTimer);
    if (!busy) {
      loadingOverlay.hidden = true;
      cancelGeneration.disabled = false;
      cancelGeneration.textContent = "Cancel generation";
      generateLabel.textContent = "Generate motion";
      if (!viewer?.getPlaybackState().frameCount) emptyState.hidden = false;
    }
    if (busy) {
      viewer?.setPlaying(false);
      emptyState.hidden = true;
      generationStage.textContent = "Preparing generation";
      generationPercent.textContent = "0%";
      generationProgressValue = 0;
      setProgress(generationProgressFill, generationProgressbar, 0);
      generateLabel.textContent = "Generating · 0%";
      loadingTitle.textContent = "Generating motion";
      loadingDetail.textContent = "Preparing inference…";
      loadingOverlay.hidden = true;
      loadingOverlayTimer = window.setTimeout(() => {
        if (generating) loadingOverlay.hidden = false;
      }, 250);
    }
    updateGenerateAvailability();
  }

  function updatePlayback(state: PlaybackState): void {
    const hasMotion = state.frameCount > 0;
    playPause.disabled = !hasMotion;
    timeline.disabled = !hasMotion;
    playbackSpeed.disabled = !hasMotion;
    loopToggle.disabled = !hasMotion;
    playPause.dataset.playing = String(state.playing);
    playPause.setAttribute("aria-label", state.playing ? "Pause motion" : "Play motion");
    timeline.max = String(Math.max(1, state.frameCount - 1));
    timeline.value = String(state.frame);
    const progress = state.frameCount > 1 ? (state.frame / (state.frameCount - 1)) * 100 : 0;
    timeline.style.setProperty("--range-progress", `${progress}%`);
    const elapsed = formatTime(state.frame / state.fps);
    const total = formatTime(state.frameCount / state.fps);
    currentTime.textContent = elapsed;
    totalTime.textContent = total;
    timeline.setAttribute("aria-valuetext", `${elapsed} of ${total}`);
  }

  if (viewer) viewer.onPlaybackChange = updatePlayback;

  async function chooseModelPack(): Promise<File[] | null> {
    const picker = (window as unknown as DirectoryPickerWindow).showDirectoryPicker;
    if (!picker) {
      fileInput.click();
      return null;
    }
    try {
      const directory = await picker({ mode: "read" });
      return canonicalizePackFiles(await collectDirectoryFiles(directory));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return null;
      throw error;
    }
  }

  function loadModelPack(files: File[], shouldCache: boolean): void {
    clearError();
    clearModelError();
    const canonicalFiles = canonicalizePackFiles(files);
    const totalBytes = canonicalFiles.reduce((total, file) => total + file.size, 0);
    activeLoadRequest = requestId("load");
    lastPackFiles = canonicalFiles;
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
    setModelStatus("loading", "Loading Core40 model", `${canonicalFiles.length} files · ${formatBytes(totalBytes)}`, "Loading");
    updateGenerateAvailability();
    announce(`Loading model pack, ${formatBytes(totalBytes)}. Files stay on this device.`);
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
        const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
        modelProgressValue = Math.max(modelProgressValue, percent / 100);
        setProgress(modelProgressFill, modelProgressbar, modelProgressValue);
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
        error instanceof Error ? error.message : "The browser could not persist this model pack.",
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
      modelProgressValue = Math.max(modelProgressValue, modelLoadProgress(event));
      const percent = setProgress(modelProgressFill, modelProgressbar, modelProgressValue);
      modelProgress.hidden = false;
      modelProgressLabel.textContent = `${percent}%`;
      modelTitle.textContent = event.message || humanizeStage(event.stage);
      return;
    }
    if (event.requestId === activeGenerationRequest) {
      generationProgressValue = Math.max(generationProgressValue, generationProgress(event));
      const percent = setProgress(generationProgressFill, generationProgressbar, generationProgressValue);
      generationStage.textContent = event.message || humanizeStage(event.stage);
      generationPercent.textContent = `${percent}%`;
      generateLabel.textContent = `Generating · ${percent}%`;
      loadingDetail.textContent = event.message || humanizeStage(event.stage);
    }
  }

  function handleModelLoaded(event: ModelLoadedEvent): void {
    if (event.requestId !== activeLoadRequest) return;
    modelLoading = false;
    modelReady = true;
    activeLoadRequest = null;
    modelLabel = event.model.variant || event.model.id;
    modelBackend = event.model.backend === "webgpu" ? "WebGPU" : "WebAssembly";
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
    removeModel.hidden = !cachedPack && !packPendingCache;
    updateGenerateAvailability();
    announce(`${event.model.variant || "ARDY Mini Core40"} is ready. Inference will run on ${modelBackend}.`);
    if (packPendingCache && lastPackFiles) {
      packPendingCache = false;
      void persistLoadedPack(lastPackFiles);
    }
  }

  function handleGenerationResult(event: GenerationResultEvent): void {
    if (event.requestId !== activeGenerationRequest) return;
    try {
      const [, frames, joints, components] = event.result.jointsShape;
      if (frames !== event.result.frameCount || joints !== CORE27_JOINT_COUNT || components !== 3) {
        throw new Error(
          `Expected Core27 joints [1,T,${CORE27_JOINT_COUNT},3], received [${event.result.jointsShape.join(",")}].`,
        );
      }
      const clip = normalizeMotionClip({
        positions: event.result.joints,
        frameCount: event.result.frameCount,
        fps: event.result.fps,
      });
      const autoplay = shouldAutoplayMotion(reducedMotion.matches);
      viewer?.setLoop(autoplay);
      viewer?.setMotion(clip, autoplay);
      loopToggle.setAttribute("aria-pressed", String(autoplay));
      loopToggle.classList.toggle("is-active", autoplay);
      emptyState.hidden = true;
      motionBadge.dataset.state = "ready";
      motionBadge.textContent = `${clip.frameCount} frames · ${clip.fps} FPS · seed ${event.result.seed}`;
      runtimeMetric.hidden = false;
      runtimeValue.textContent =
        event.result.timingsMs.total >= 1000
          ? `${(event.result.timingsMs.total / 1000).toFixed(2)} s`
          : `${Math.round(event.result.timingsMs.total)} ms`;
      generationPercent.textContent = "100%";
      setProgress(generationProgressFill, generationProgressbar, 1);
      setGenerationBusy(false);
      activeGenerationRequest = null;
      announce(
        `Generation complete: ${clip.frameCount} frames at ${clip.fps} frames per second.${
          autoplay ? " Playback started." : " Playback is paused because reduced motion is enabled."
        }`,
      );
      if (window.matchMedia("(max-width: 900px)").matches) {
        const viewportBounds = viewport.getBoundingClientRect();
        const playbackBounds = playbackBar.getBoundingClientRect();
        const visibleViewportHeight = Math.max(
          0,
          Math.min(window.innerHeight, viewportBounds.bottom) - Math.max(0, viewportBounds.top),
        );
        const viewportVisibility =
          viewportBounds.height > 0 ? visibleViewportHeight / viewportBounds.height : 0;
        const controlsVisible =
          playbackBounds.top >= 0 && playbackBounds.bottom <= window.innerHeight;
        if (viewportVisibility < 0.5 || !controlsVisible) {
          viewportPanel.scrollIntoView({
            behavior: reducedMotion.matches ? "auto" : "smooth",
            block: "start",
          });
        }
      }
    } catch (error) {
      setGenerationBusy(false);
      activeGenerationRequest = null;
      showError("Could not display generated motion", error instanceof Error ? error.message : String(error));
    }
  }

  worker.addEventListener("message", (message: MessageEvent<WorkerEvent>) => {
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
      case "generationResult":
        handleGenerationResult(event);
        break;
      case "cancelled":
        if (event.targetRequestId === activeGenerationRequest) {
          activeGenerationRequest = null;
          setGenerationBusy(false);
          announce("Generation cancelled.");
        }
        break;
      case "status":
        if (event.status.state === "empty" && !modelLoading) modelReady = false;
        updateGenerateAvailability();
        break;
      case "error": {
        const wasLoading = event.requestId === activeLoadRequest;
        const wasGenerating = event.requestId === activeGenerationRequest;
        if (wasLoading) {
          activeLoadRequest = null;
          modelLoading = false;
          modelReady = false;
          modelProgress.hidden = true;
          modelCard.removeAttribute("aria-busy");
          resetRuntimeBadge();
          setModelStatus("missing", "Model pack could not be loaded", "Check the pack and try importing it again.", "Error");
        }
        if (wasGenerating) {
          activeGenerationRequest = null;
          setGenerationBusy(false);
        }
        if (wasLoading) {
          showModelError("Model import failed", event.error.message);
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
  });

  worker.addEventListener("error", (event) => {
    modelLoading = false;
    modelReady = false;
    setGenerationBusy(false);
    modelCard.removeAttribute("aria-busy");
    resetRuntimeBadge();
    setModelStatus("missing", "Inference worker unavailable", "Reload the page to try again.", "Error");
    showModelError("Inference worker stopped", event.message || "An unexpected worker error occurred.");
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!modelReady || generating) return;
    clearError();
    const validation = validateGenerationForm(prompt.value, duration.value, seed.value);
    promptError.textContent = validation.promptError ?? "";
    seedError.textContent = validation.seedError ?? "";
    prompt.toggleAttribute("aria-invalid", Boolean(validation.promptError));
    seed.toggleAttribute("aria-invalid", Boolean(validation.seedError));
    if (!validation.values) {
      if (validation.promptError) {
        prompt.focus();
      } else if (validation.seedError) {
        runtimeSettings.open = true;
        window.requestAnimationFrame(() => seed.focus());
      }
      return;
    }

    activeGenerationRequest = requestId("generate");
    setGenerationBusy(true);
    window.requestAnimationFrame(() => {
      if (generating && !cancelGeneration.hidden) cancelGeneration.focus();
    });
    announce("Generation started. Inference is running locally.");
    const command: GenerateCommand = {
      type: "generate",
      requestId: activeGenerationRequest,
      prompt: validation.values.prompt,
      seed: validation.values.seed,
      durationSeconds: validation.values.durationSeconds,
    };
    postCommand(command);
  });

  prompt.addEventListener("input", updatePrompt);
  duration.addEventListener("input", updateDuration);
  seed.addEventListener("input", updateSeed);

  document.querySelectorAll<HTMLButtonElement>(".preset-chip").forEach((button) => {
    button.addEventListener("click", () => {
      prompt.value = button.dataset.prompt ?? "";
      updatePrompt();
      prompt.focus();
    });
  });

  randomizeSeed.addEventListener("click", () => {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    seed.value = String(random[0]);
    updateSeed();
  });

  backend.addEventListener("change", () => {
    const selected = backend.value as RuntimeBackendPreference;
    if (selected === "webgpu" && !hasWebGpu) {
      backendHelp.textContent = "WebGPU is unavailable in this browser. Select Auto or WebAssembly.";
    } else if (selected === "webgpu") {
      backendHelp.textContent = "Changing the backend reloads the model. WebGPU is preferred, with WebAssembly fallback.";
    } else if (selected === "wasm") {
      backendHelp.textContent = crossOriginIsolated
        ? "Changing the backend reloads the model. WebAssembly uses local SIMD and multi-threaded CPU inference."
        : "Changing the backend reloads the model. WebAssembly is single-threaded without COOP/COEP headers.";
    } else {
      backendHelp.textContent = "Changing the backend reloads the model. Auto prefers WebGPU and falls back to WebAssembly.";
    }
    if (lastPackFiles && modelReady) loadModelPack(lastPackFiles, false);
  });

  importModel.addEventListener("click", async () => {
    try {
      const files = await chooseModelPack();
      if (files) loadModelPack(files, true);
    } catch (error) {
      showModelError("Could not open model pack", error instanceof Error ? error.message : String(error));
    }
  });

  fileInput.addEventListener("change", () => {
    try {
      if (fileInput.files?.length) loadModelPack(Array.from(fileInput.files), true);
    } catch (error) {
      showModelError("Could not open model pack", error instanceof Error ? error.message : String(error));
    } finally {
      fileInput.value = "";
    }
  });

  removeModel.addEventListener("click", async () => {
    if (!window.confirm("Unload the active model and remove its saved browser copy? Generated motion will remain visible.")) {
      return;
    }
    try {
      await removeCachedModelPack();
      cachedPack = false;
      lastPackFiles = null;
      modelReady = false;
      removeModel.hidden = true;
      postCommand({ type: "dispose", requestId: requestId("dispose") });
      resetRuntimeBadge();
      setModelStatus("missing", "Model pack required", "Choose the exported Core40 browser-pack folder.", "Not loaded");
      updateGenerateAvailability();
      announce("The model was unloaded and its saved browser copy was removed.");
    } catch (error) {
      showModelError("Could not remove cached pack", error instanceof Error ? error.message : String(error));
    }
  });

  cancelGeneration.addEventListener("click", () => {
    if (!activeGenerationRequest) return;
    cancelGeneration.disabled = true;
    cancelGeneration.textContent = "Cancelling…";
    generateLabel.textContent = "Cancelling…";
    postCommand({
      type: "cancel",
      requestId: requestId("cancel"),
      targetRequestId: activeGenerationRequest,
    });
    announce("Cancelling after the current inference step.");
  });

  dismissError.addEventListener("click", () => clearError(true));
  dismissModelError.addEventListener("click", () => clearModelError(true));
  playPause.addEventListener("click", () => viewer?.togglePlaying());
  timeline.addEventListener("input", () => viewer?.seek(Number(timeline.value)));
  playbackSpeed.addEventListener("change", () => viewer?.setSpeed(Number(playbackSpeed.value)));
  loopToggle.addEventListener("click", () => {
    const loop = loopToggle.getAttribute("aria-pressed") !== "true";
    loopToggle.setAttribute("aria-pressed", String(loop));
    loopToggle.classList.toggle("is-active", loop);
    viewer?.setLoop(loop);
  });
  resetCamera.addEventListener("click", () => viewer?.resetCamera());

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    viewer?.setReducedMotion(event.matches);
    if (event.matches) {
      viewer?.setPlaying(false);
      viewer?.setLoop(false);
      loopToggle.setAttribute("aria-pressed", "false");
      loopToggle.classList.remove("is-active");
      announce("Reduced motion enabled. Motion playback is paused and looping is off.");
    }
  };
  reducedMotion.addEventListener("change", handleReducedMotionChange);

  document.addEventListener("keydown", (event) => {
    const target = event.target as HTMLElement | null;
    if (event.isComposing || event.repeat) return;
    if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key === "Enter") {
      event.preventDefault();
      if (!generate.disabled) form.requestSubmit();
      return;
    }
    if (event.key === "Escape" && activeGenerationRequest) {
      cancelGeneration.click();
      return;
    }
    if (target !== canvas || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.shiftKey && event.key.startsWith("Arrow")) {
      event.preventDefault();
      const horizontal = event.key === "ArrowLeft" ? 1 : event.key === "ArrowRight" ? -1 : 0;
      const vertical = event.key === "ArrowUp" ? 1 : event.key === "ArrowDown" ? -1 : 0;
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
  });

  window.addEventListener("beforeunload", () => {
    postCommand({ type: "dispose", requestId: requestId("dispose") });
    worker.terminate();
    reducedMotion.removeEventListener("change", handleReducedMotionChange);
    viewer?.dispose();
  });

  updatePrompt();
  updateDuration();
  updateSeed();
  modelCard.setAttribute("aria-busy", "true");
  setModelStatus("loading", "Checking local model cache", "No network request is made.", "Checking");
  void readCachedModelPack()
    .then((files) => {
      if (files) {
        cachedPack = true;
        removeModel.hidden = false;
        loadModelPack(files, false);
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
      modelCard.removeAttribute("aria-busy");
      setModelStatus("missing", "Model pack required", "Persistent cache could not be read.", "Not loaded");
      showModelError("Could not read model cache", error instanceof Error ? error.message : String(error));
    });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
} else {
  bootstrap();
}

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  createCapturedConstraint,
  createRootConstraint,
  interpolateRootWaypoints,
  postprocessMotion,
  validateModelPackManifest,
  waypointsFromTargetVelocity,
  type BrowserModelPackManifest,
  type GenerationCompleteEvent,
  type GenerationMode,
  type LoadModelPackCommand,
  type ModelLoadedEvent,
  type MotionConstraint,
  type MotionConstraintKind as RuntimeConstraintKind,
  type ProgressEvent,
  type RuntimeBackendPreference,
  type RuntimeCapabilities,
  type RuntimeContinuationState,
  type RuntimeGenerationChunk,
  type RuntimeGenerationResult,
  type WorkerCommand,
  type WorkerEvent,
} from "./runtime";
import {
  normalizeStructuredMotion,
  type RotationTrack,
  type StructuredMotionResult,
} from "./motion-data";
import {
  decodeMotionJson,
  decodeSessionFile,
  downloadMotionCsv,
  downloadMotionJson,
  downloadSessionBinary,
  isContinuationModelCompatible,
  type BrowserMotionSession,
  type MotionSessionProvenance,
} from "./session-format";
import {
  DEFAULT_EDITOR_STATE,
  type MotionConstraintMarker,
  type MotionEditorState,
  type MotionWaypoint,
  type QuaternionTuple,
  type Vector3Tuple,
  type ViewerOutputVisibility,
} from "./editor-state";
import {
  BODY_PROXY_DESCRIPTION,
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
type ConstraintValueKind = "position" | "rotation" | "pose";

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

interface TrackDefinition {
  buttonId: string;
  runtimeKind: RuntimeConstraintKind;
}

const TRACK_DEFINITIONS: readonly TrackDefinition[] = [
  { buttonId: "constraint-track-full-body", runtimeKind: "full-body" },
  { buttonId: "constraint-track-root", runtimeKind: "root" },
  { buttonId: "constraint-track-left-hand", runtimeKind: "left-hand" },
  { buttonId: "constraint-track-right-hand", runtimeKind: "right-hand" },
  { buttonId: "constraint-track-left-foot", runtimeKind: "left-foot" },
  { buttonId: "constraint-track-right-foot", runtimeKind: "right-foot" },
];

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
  } else if (!/^[\x09\x0a\x0d\x20-\x7e]+$/.test(prompt)) {
    validation.promptError =
      "This model supports typo-free English prompts using standard Latin characters.";
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

function rootPositionAt(
  motion: StructuredMotionResult | null,
  frame: number,
): Vector3Tuple {
  if (!motion || motion.frameCount === 0) return [0, 0, 0];
  const safeFrame = Math.max(0, Math.min(motion.frameCount - 1, frame));
  const offset =
    (safeFrame * motion.skeleton.jointNames.length +
      motion.skeleton.rootJointIndex) *
    3;
  return [
    motion.positions[offset],
    motion.positions[offset + 1],
    motion.positions[offset + 2],
  ];
}

function jointPositionAt(
  motion: StructuredMotionResult,
  frame: number,
  joint: number,
): Vector3Tuple {
  const safeFrame = Math.max(0, Math.min(motion.frameCount - 1, frame));
  const offset = (safeFrame * motion.skeleton.jointNames.length + joint) * 3;
  return [
    motion.positions[offset],
    motion.positions[offset + 1],
    motion.positions[offset + 2],
  ];
}

function matrixQuaternion(
  values: Float32Array,
  offset: number,
): QuaternionTuple {
  const m00 = values[offset];
  const m01 = values[offset + 1];
  const m02 = values[offset + 2];
  const m10 = values[offset + 3];
  const m11 = values[offset + 4];
  const m12 = values[offset + 5];
  const m20 = values[offset + 6];
  const m21 = values[offset + 7];
  const m22 = values[offset + 8];
  const trace = m00 + m11 + m22;
  let x: number;
  let y: number;
  let z: number;
  let w: number;
  if (trace > 0) {
    const scale = Math.sqrt(trace + 1) * 2;
    w = 0.25 * scale;
    x = (m21 - m12) / scale;
    y = (m02 - m20) / scale;
    z = (m10 - m01) / scale;
  } else if (m00 > m11 && m00 > m22) {
    const scale = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / scale;
    x = 0.25 * scale;
    y = (m01 + m10) / scale;
    z = (m02 + m20) / scale;
  } else if (m11 > m22) {
    const scale = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / scale;
    x = (m01 + m10) / scale;
    y = 0.25 * scale;
    z = (m12 + m21) / scale;
  } else {
    const scale = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / scale;
    x = (m02 + m20) / scale;
    y = (m12 + m21) / scale;
    z = 0.25 * scale;
  }
  const magnitude = Math.hypot(x, y, z, w) || 1;
  return [x / magnitude, y / magnitude, z / magnitude, w / magnitude];
}

function jointOrientationAt(
  motion: StructuredMotionResult,
  frame: number,
  joint: number,
): QuaternionTuple | undefined {
  const track = motion.globalRotations ?? motion.localRotations;
  if (!track) return undefined;
  const safeFrame = Math.max(0, Math.min(motion.frameCount - 1, frame));
  const offset = (safeFrame * track.shape[1] + joint) * track.shape[2];
  if (track.shape[2] === 4) {
    return [
      track.values[offset],
      track.values[offset + 1],
      track.values[offset + 2],
      track.values[offset + 3],
    ];
  }
  return matrixQuaternion(track.values, offset);
}

function trackJointIndex(
  motion: StructuredMotionResult,
  kind: RuntimeConstraintKind,
): number {
  if (kind === "root" || kind === "full-body") {
    return motion.skeleton.rootJointIndex;
  }
  const side = kind.startsWith("left") ? "left" : "right";
  const part = kind.endsWith("hand") ? "hand" : "foot";
  const candidates = motion.skeleton.jointNames
    .map((name, index) => ({ name: name.toLowerCase(), index }))
    .filter(({ name }) => name.includes(side) && name.includes(part));
  if (candidates.length === 0) {
    throw new Error(`The loaded skeleton has no ${kind} joint.`);
  }
  return candidates[0].index;
}

function markerLabel(
  kind: RuntimeConstraintKind,
  valueKind: ConstraintValueKind,
): string {
  return `${kind}:${valueKind}`;
}

function markerRuntimeKind(
  marker: MotionConstraintMarker,
): RuntimeConstraintKind | null {
  const kind = marker.label?.split(":", 1)[0];
  return TRACK_DEFINITIONS.some((track) => track.runtimeKind === kind)
    ? (kind as RuntimeConstraintKind)
    : null;
}

function markerValueKind(marker: MotionConstraintMarker): ConstraintValueKind {
  const value = marker.label?.split(":")[1];
  return value === "position" || value === "rotation" || value === "pose"
    ? value
    : "pose";
}

function headingFromQuaternion(
  orientation: QuaternionTuple | undefined,
): number | undefined {
  if (!orientation) return undefined;
  const [x, y, z, w] = orientation;
  return Math.atan2(
    2 * (w * y + x * z),
    1 - 2 * (y * y + z * z),
  );
}

function rootHeadingAt(
  motion: StructuredMotionResult,
  manifest: BrowserModelPackManifest,
  frame: number,
): number | undefined {
  const heading = manifest.motion_layout?.global_root_heading;
  const stats = manifest.stats?.motion;
  const stride = motion.normalizedMotionShape?.[1];
  if (
    !motion.normalizedMotion ||
    !stride ||
    !heading ||
    heading[1] - heading[0] < 2 ||
    !stats ||
    stats.mean.length <= heading[0] + 1 ||
    stats.normalization_denominator.length <= heading[0] + 1
  ) {
    return undefined;
  }
  const safeFrame = Math.max(0, Math.min(motion.frameCount - 1, frame));
  const offset = safeFrame * stride + heading[0];
  const cosine =
    motion.normalizedMotion[offset] *
      stats.normalization_denominator[heading[0]] +
    stats.mean[heading[0]];
  const sine =
    motion.normalizedMotion[offset + 1] *
      stats.normalization_denominator[heading[0] + 1] +
    stats.mean[heading[0] + 1];
  return Number.isFinite(cosine) && Number.isFinite(sine)
    ? Math.atan2(sine, cosine)
    : undefined;
}

function filterConstraintValue(
  constraint: MotionConstraint,
  kind: RuntimeConstraintKind,
  valueKind: ConstraintValueKind,
  manifest: BrowserModelPackManifest,
): MotionConstraint {
  if (valueKind === "pose" || kind === "full-body") return constraint;
  const mask = constraint.mask.slice();
  const layout = manifest.motion_layout;
  if (kind === "root") {
    if (valueKind === "position" && layout?.global_root_heading) {
      mask.fill(
        0,
        layout.global_root_heading[0],
        layout.global_root_heading[1],
      );
    } else if (valueKind === "rotation" && layout?.root_pos) {
      mask.fill(0, layout.root_pos[0], layout.root_pos[1]);
    }
  } else if (valueKind === "position" && layout?.global_rot_data) {
    mask.fill(0, layout.global_rot_data[0], layout.global_rot_data[1]);
  } else if (valueKind === "rotation" && layout?.local_joints_positions) {
    mask.fill(
      0,
      layout.local_joints_positions[0],
      layout.local_joints_positions[1],
    );
  }
  return { ...constraint, mask };
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

function cloneRuntimeConstraint(constraint: MotionConstraint): MotionConstraint {
  return {
    ...constraint,
    values: new Float32Array(constraint.values),
    mask: new Float32Array(constraint.mask),
  };
}

export function bootstrap(): void {
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
  const dismissModelError =
    requiredElement<HTMLButtonElement>("dismiss-model-error");
  const generate = requiredElement<HTMLButtonElement>("generate");
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
    requiredElement<HTMLElement>("generation-progress-fill");
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
  const correctionMetric = requiredElement<HTMLElement>("correction-metric");
  const rootErrorValue = requiredElement<HTMLElement>("root-error-value");
  const footSlideValue = requiredElement<HTMLElement>("foot-slide-value");
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
  const playbackBar = requiredElement<HTMLElement>("playback-bar");
  const modelRuntimeStatus =
    requiredElement<HTMLElement>("model-runtime-status");
  const modelRuntimeDetail =
    requiredElement<HTMLElement>("model-runtime-detail");

  const newSession = requiredElement<HTMLButtonElement>("new-session");
  const importSession = requiredElement<HTMLButtonElement>("import-session");
  const sessionFileInput =
    requiredElement<HTMLInputElement>("session-file-input");
  const exportSession = requiredElement<HTMLButtonElement>("export-session");
  const exportMotion = requiredElement<HTMLButtonElement>("export-motion");

  const initialX = requiredElement<HTMLInputElement>("initial-x");
  const initialZ = requiredElement<HTMLInputElement>("initial-z");
  const initialHeading =
    requiredElement<HTMLInputElement>("initial-heading");
  const textCfg = requiredElement<HTMLInputElement>("text-cfg");
  const constraintCfg = requiredElement<HTMLInputElement>("constraint-cfg");
  const historyFrames = requiredElement<HTMLInputElement>("history-frames");
  const futureCrop = requiredElement<HTMLInputElement>("future-crop");
  const replanBuffer = requiredElement<HTMLInputElement>("replan-buffer");
  const replanThreshold =
    requiredElement<HTMLInputElement>("replan-threshold");

  const constraintType =
    requiredElement<HTMLSelectElement>("constraint-type");
  const constraintFrame =
    requiredElement<HTMLInputElement>("constraint-frame");
  const constraintEndFrame =
    requiredElement<HTMLInputElement>("constraint-end-frame");
  const addConstraint = requiredElement<HTMLButtonElement>("add-constraint");
  const deleteConstraint =
    requiredElement<HTMLButtonElement>("delete-constraint");
  const clearConstraints =
    requiredElement<HTMLButtonElement>("clear-constraints");
  const trackButtons = new Map<RuntimeConstraintKind, HTMLButtonElement>(
    TRACK_DEFINITIONS.map(({ buttonId, runtimeKind }) => [
      runtimeKind,
      requiredElement<HTMLButtonElement>(buttonId),
    ]),
  );

  const waypointMode = requiredElement<HTMLInputElement>("waypoint-mode");
  const waypointInterval =
    requiredElement<HTMLInputElement>("waypoint-interval");
  const waypointDense = requiredElement<HTMLInputElement>("waypoint-dense");
  const addWaypoint = requiredElement<HTMLButtonElement>("add-waypoint");
  const targetVelocity =
    requiredElement<HTMLInputElement>("target-velocity");
  const targetHeading =
    requiredElement<HTMLInputElement>("target-heading");
  const applyTargetVelocity =
    requiredElement<HTMLButtonElement>("apply-target-velocity");

  const postprocessEnabled =
    requiredElement<HTMLInputElement>("postprocess-enabled");
  const rootHeightMargin =
    requiredElement<HTMLInputElement>("root-height-margin");
  const contactThreshold =
    requiredElement<HTMLInputElement>("contact-threshold");

  const showSkeleton = requiredElement<HTMLInputElement>("show-skeleton");
  const showContacts = requiredElement<HTMLInputElement>("show-contacts");
  const showOrientations =
    requiredElement<HTMLInputElement>("show-orientations");
  const showTrajectory =
    requiredElement<HTMLInputElement>("show-trajectory");
  const showMesh = requiredElement<HTMLInputElement>("show-mesh");
  const showReference = requiredElement<HTMLInputElement>("show-reference");
  const importReference =
    requiredElement<HTMLButtonElement>("import-reference");
  const referenceFileInput =
    requiredElement<HTMLInputElement>("reference-file-input");

  fileInput.setAttribute("webkitdirectory", "");
  fileInput.setAttribute("directory", "");
  showMesh.title = BODY_PROXY_DESCRIPTION;
  showReference.disabled = true;
  exportSession.disabled = true;
  exportMotion.disabled = true;
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
  let capabilities: RuntimeCapabilities | null = null;
  let generationProgressValue = 0;
  let modelProgressValue = 0;
  let loadingOverlayTimer = 0;
  let currentMotion: StructuredMotionResult | null = null;
  let referenceMotion: StructuredMotionResult | null = null;
  let currentContinuation: RuntimeContinuationState | null = null;
  let currentProvenance: MotionSessionProvenance = {};
  let editorState = cloneEditorState(DEFAULT_EDITOR_STATE);
  let runtimeConstraints = new Map<string, MotionConstraint>();
  let rebuildConstraintsAfterModelLoad = false;
  let selectedTrack: RuntimeConstraintKind = "root";
  let pendingNewSession = false;

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
    fill.style.transform = `scaleX(${safeFraction})`;
    progressbar.setAttribute("aria-valuenow", String(percent));
    return percent;
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

  function updatePrompt(): void {
    promptCount.textContent = `${prompt.value.length} / 280`;
    if (prompt.getAttribute("aria-invalid") === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        duration.value,
        seed.value,
      );
      promptError.textContent = validation.promptError ?? "";
      prompt.toggleAttribute("aria-invalid", Boolean(validation.promptError));
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
    if (seed.getAttribute("aria-invalid") === "true") {
      const validation = validateGenerationForm(
        prompt.value,
        duration.value,
        seed.value,
      );
      seedError.textContent = validation.seedError ?? "";
      seed.toggleAttribute("aria-invalid", Boolean(validation.seedError));
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
    addConstraint.disabled =
      !modelReady ||
      !activeManifest ||
      !currentMotion?.normalizedMotion ||
      generating;
    addWaypoint.disabled = !modelReady || generating;
    applyTargetVelocity.disabled = !modelReady || generating;
    exportSession.disabled = currentMotion === null;
    exportMotion.disabled = currentMotion === null;
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
    generationProgressElement.hidden = !active;
    cancelGeneration.hidden = !active;
    cancelGeneration.disabled = false;
    cancelGeneration.textContent = "Cancel";
    generateLabel.textContent = active
      ? background
        ? "Extending motion"
        : "Generating motion"
      : "Generate motion";
    generate.setAttribute("aria-busy", String(active));
    if (loadingOverlayTimer) {
      window.clearTimeout(loadingOverlayTimer);
      loadingOverlayTimer = 0;
    }
    if (active && !background && !currentMotion) {
      loadingOverlayTimer = window.setTimeout(() => {
        if (activeGeneration && !activeGeneration.background) {
          loadingOverlay.hidden = false;
        }
      }, 300);
    } else {
      loadingOverlay.hidden = true;
    }
    updateGenerateAvailability();
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
    updateConstraintTimeline();
  }

  function updateInitialTransform(): void {
    const x = finiteInput(initialX, 0);
    const z = finiteInput(initialZ, 0);
    const headingRadians =
      (finiteInput(initialHeading, 0, -180, 180) * Math.PI) / 180;
    setEditor({
      ...editorState,
      initialTransform: {
        position: [x, 0, z],
        headingRadians,
      },
    });
  }

  function updateConstraintTimeline(): void {
    const extent = Math.max(
      1,
      currentMotion?.frameCount ?? 0,
      ...editorState.constraints.map((constraint) => constraint.endFrame + 1),
      ...editorState.waypoints.map((waypoint) => waypoint.frame + 1),
    );
    for (const { runtimeKind } of TRACK_DEFINITIONS) {
      const button = trackButtons.get(runtimeKind)!;
      const matching = editorState.constraints.filter(
        (marker) => markerRuntimeKind(marker) === runtimeKind,
      );
      const latest = matching.at(-1);
      button.setAttribute(
        "aria-pressed",
        String(runtimeKind === selectedTrack),
      );
      button.dataset.hasConstraint = String(matching.length > 0);
      if (latest) {
        const position = Math.max(
          0,
          Math.min(100, (latest.startFrame / Math.max(1, extent - 1)) * 100),
        );
        button.style.setProperty("--track-position", `${position}%`);
        button.title = `${matching.length} constraint${matching.length === 1 ? "" : "s"}; latest at frame ${latest.startFrame}`;
      } else {
        button.style.removeProperty("--track-position");
        button.title = "No constraints on this track";
      }
    }
  }

  function outputVisibilityFromControls(): ViewerOutputVisibility {
    return {
      ...editorState.outputVisibility,
      skeleton: showSkeleton.checked,
      mesh: showMesh.checked,
      reference: showReference.checked && referenceMotion !== null,
      contacts: showContacts.checked,
      orientationAxes: showOrientations.checked,
      trajectory: showTrajectory.checked,
    };
  }

  function syncOutputVisibility(): void {
    if (
      referenceMotion &&
      currentMotion &&
      !sameSkeleton(referenceMotion.skeleton, currentMotion.skeleton)
    ) {
      referenceMotion = null;
      showReference.checked = false;
      showReference.disabled = true;
    }
    const visibility = outputVisibilityFromControls();
    editorState = { ...editorState, outputVisibility: visibility };
    viewer?.setReferenceMotion(referenceMotion);
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
    if (
      document.activeElement !== constraintFrame &&
      document.activeElement !== constraintEndFrame &&
      state.frameCount > 0
    ) {
      constraintFrame.value = String(state.frame);
      constraintEndFrame.value = String(state.frame);
    }
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
      generateLabel.textContent = activeGeneration.background
        ? `Extending · ${percent}%`
        : `Generating · ${percent}%`;
      loadingDetail.textContent =
        event.message || humanizeStage(event.stage);
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
        editorState.initialTransform.position,
      ),
      initialHeading: editorState.initialTransform.headingRadians,
    });
  }

  function applyCapabilities(next: RuntimeCapabilities): void {
    capabilities = next;
    const constraintAvailable = next.constraints;
    for (const button of trackButtons.values()) {
      button.disabled = !constraintAvailable;
    }
    constraintType.disabled = !constraintAvailable;
    constraintFrame.disabled = !constraintAvailable;
    constraintEndFrame.disabled = !constraintAvailable;
    clearConstraints.disabled = !constraintAvailable;
    waypointMode.disabled = !constraintAvailable;
    waypointInterval.disabled = !constraintAvailable;
    waypointDense.disabled = !constraintAvailable;
    targetVelocity.disabled = !constraintAvailable;
    targetHeading.disabled = !constraintAvailable;
    postprocessEnabled.disabled = !next.richMotionOutputs;
    if (!constraintAvailable && waypointMode.checked) {
      waypointMode.checked = false;
      if (viewer) viewer.onGroundClick = null;
    }
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
    applyCapabilities(event.model.capabilities);
    const incompatibleContinuation =
      currentContinuation !== null &&
      !isContinuationModelCompatible(currentProvenance, event.model);
    if (incompatibleContinuation) {
      markCurrentMotionPlaybackOnly();
    }
    if (rebuildConstraintsAfterModelLoad) {
      rebuildRuntimeConstraintsFromEditor();
      rebuildConstraintsAfterModelLoad = false;
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
    resetLoop = false,
    resetCamera = false,
  ): void {
    if (!currentMotion || !viewer) return;
    viewer.setMotion(currentMotion, false, resetCamera);
    setEditor(editorState);
    syncOutputVisibility();
    viewer.seek(Math.min(preserveFrame, currentMotion.frameCount - 1));
    if (resetLoop) {
      const loop = !streamGeneration.checked && !reducedMotion.matches;
      viewer.setLoop(loop);
      loopToggle.setAttribute("aria-pressed", String(loop));
      loopToggle.classList.toggle("is-active", loop);
    }
    if (preservePlaying && !reducedMotion.matches) viewer.setPlaying(true);
    emptyState.hidden = true;
    motionBadge.dataset.state = "ready";
    motionBadge.textContent = `${currentMotion.frameCount} frames · ${currentMotion.fps} FPS`;
    exportSession.disabled = false;
    exportMotion.disabled = false;
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
    const resetPresentation =
      active.mode === "replace" && chunk.startFrame === 0;
    refreshViewer(
      preserveFrame,
      preservePlaying,
      resetPresentation,
      resetPresentation,
    );
    generationStage.textContent = `Received ${chunk.frameCount} frames`;
    announce(
      `Generated frames ${chunk.startFrame} through ${chunk.startFrame + chunk.frameCount - 1}.`,
    );
  }

  function allRuntimeConstraints(): MotionConstraint[] {
    const result = [...runtimeConstraints.values()];
    if (activeManifest) {
      const rootWaypoints = editorState.waypoints
        .filter((waypoint) => waypoint.enabled)
        .map((waypoint) => ({
          id: waypoint.id,
          frame: waypoint.frame,
          x: waypoint.position[0],
          z: waypoint.position[2],
          heading: waypoint.headingRadians,
        }));
      result.push(
        ...interpolateRootWaypoints(
          activeManifest,
          rootWaypoints,
          waypointDense.checked,
        ),
      );
    }
    return result;
  }

  function runPostprocess(): boolean {
    correctionMetric.hidden = true;
    correctionMetric.removeAttribute("data-root-error-after");
    correctionMetric.removeAttribute("data-foot-slide-after");
    runtimeValue.removeAttribute("title");
    if (
      !postprocessEnabled.checked ||
      !currentMotion ||
      !activeManifest
    ) {
      return false;
    }
    const contacts =
      currentMotion.contacts && currentMotion.contactsShape
        ? {
            values: currentMotion.contacts,
            channels: currentMotion.contactsShape[1],
            jointIndices: currentMotion.skeleton.contactJointIndices,
          }
        : undefined;
    const roots =
      currentMotion.roots && currentMotion.rootsShape
        ? {
            values: currentMotion.roots,
            components: currentMotion.rootsShape[1],
          }
        : undefined;
    const corrected = postprocessMotion(
      {
        positions: currentMotion.positions,
        frameCount: currentMotion.frameCount,
        jointCount: currentMotion.skeleton.jointNames.length,
        rootJointIndex: currentMotion.skeleton.rootJointIndex,
        roots,
        contacts,
        constraints: allRuntimeConstraints(),
        constraintManifest: activeManifest,
        frameOffset: 0,
      },
      {
        rootMargin: finiteInput(rootHeightMargin, 0.04, 0, 1),
        contactThreshold: finiteInput(contactThreshold, 0.5, 0, 1),
      },
    );
    let normalizedMotion = currentMotion.normalizedMotion;
    if (
      corrected.roots &&
      currentMotion.rootsShape &&
      normalizedMotion &&
      currentMotion.normalizedMotionShape &&
      currentContinuation
    ) {
      const rootSlice = activeManifest.motion_layout?.root_pos;
      const stats = activeManifest.stats?.motion;
      if (
        rootSlice &&
        rootSlice[1] - rootSlice[0] >= 3 &&
        stats &&
        stats.mean.length >= rootSlice[0] + 3 &&
        stats.normalization_denominator.length >= rootSlice[0] + 3
      ) {
        normalizedMotion = normalizedMotion.slice();
        const continuationTokens =
          currentContinuation.hybridTokens.slice();
        const motionStride = currentMotion.normalizedMotionShape[1];
        const rootStride = currentMotion.rootsShape[1];
        const dimensions = activeManifest.dimensions;
        const continuationFrames = Math.min(
          currentMotion.frameCount,
          currentContinuation.frameCount,
          (continuationTokens.length / dimensions.hybrid_dim) *
            dimensions.num_frames_per_token,
        );
        for (let frame = 0; frame < currentMotion.frameCount; frame += 1) {
          for (let axis = 0; axis < 3; axis += 1) {
            const feature = rootSlice[0] + axis;
            const normalized = Math.fround(
              (corrected.roots[frame * rootStride + axis] -
                stats.mean[feature]) /
                stats.normalization_denominator[feature],
            );
            normalizedMotion[frame * motionStride + feature] = normalized;
            if (frame < continuationFrames) {
              const token = Math.floor(
                frame / dimensions.num_frames_per_token,
              );
              const frameInToken =
                frame % dimensions.num_frames_per_token;
              const rootFeature = feature - rootSlice[0];
              continuationTokens[
                token * dimensions.hybrid_dim +
                  frameInToken * dimensions.root_features_per_frame +
                  rootFeature
              ] = normalized;
            }
          }
        }
        currentContinuation = {
          ...currentContinuation,
          hybridTokens: continuationTokens,
        };
      }
    }
    currentMotion = {
      ...currentMotion,
      positions: corrected.positions,
      normalizedMotion,
      roots: corrected.roots ?? currentMotion.roots,
    };
    runtimeValue.title =
      `Foot slide ${corrected.metrics.footSlidingDistanceBefore.toFixed(3)} → ` +
      `${corrected.metrics.footSlidingDistanceAfter.toFixed(3)} m`;
    const rootBefore = corrected.metrics.rootConstraintMeanErrorBefore;
    const rootAfter = corrected.metrics.rootConstraintMeanErrorAfter;
    const slideBefore = corrected.metrics.footSlidingDistanceBefore;
    const slideAfter = corrected.metrics.footSlidingDistanceAfter;
    rootErrorValue.textContent =
      `${rootBefore.toFixed(3)}→${rootAfter.toFixed(3)} m`;
    footSlideValue.textContent =
      `${slideBefore.toFixed(3)}→${slideAfter.toFixed(3)} m`;
    correctionMetric.dataset.rootErrorAfter = String(rootAfter);
    correctionMetric.dataset.footSlideAfter = String(slideAfter);
    correctionMetric.hidden = false;
    return true;
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
      const continuationCorrected = runPostprocess();
      const playback = viewer?.getPlaybackState();
      const resetPresentation =
        active.mode === "replace" && !active.receivedChunk;
      refreshViewer(
        playback?.frame ?? 0,
        Boolean(playback?.playing || active.resumePlayback),
        resetPresentation,
        resetPresentation,
      );
      generationPercent.textContent = "100%";
      setProgress(generationProgressFill, generationProgressbar, 1);
      runtimeMetric.hidden = false;
      runtimeValue.textContent =
        event.result.timingsMs.total >= 1000
          ? `${(event.result.timingsMs.total / 1000).toFixed(2)} s`
          : `${Math.round(event.result.timingsMs.total)} ms`;
      activeGeneration = null;
      setGenerationBusy(false);
      if (continuationCorrected) {
        restoreWorkerContinuation(false);
      }
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
    prompt.toggleAttribute("aria-invalid", Boolean(validation.promptError));
    seed.toggleAttribute("aria-invalid", Boolean(validation.seedError));
    if (!validation.values) {
      if (validation.promptError) {
        prompt.focus();
      } else if (validation.seedError) {
        runtimeSettings.open = true;
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
    loadingTitle.textContent =
      mode === "append"
        ? "Extending motion"
        : mode === "branch"
          ? "Replanning motion"
          : "Generating motion";
    loadingDetail.textContent = "Encoding prompt…";
    setGenerationBusy(true, background);
    if (!background) {
      window.requestAnimationFrame(() => {
        if (activeGeneration && !cancelGeneration.hidden) {
          cancelGeneration.focus();
        }
      });
    }
    const requestedFuture = integerInput(
      futureCrop,
      0,
      0,
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
      textCfgWeight: finiteInput(textCfg, 3.5, 0, 100),
      constraintCfgWeight: finiteInput(constraintCfg, 1, 0, 100),
      historyFrames: integerInput(
        historyFrames,
        40,
        0,
        activeManifest?.dimensions.history_frames ?? 40,
      ),
      futureFrames: requestedFuture,
      constraints: capabilities?.constraints
        ? allRuntimeConstraints()
        : undefined,
      ...(mode === "replace"
        ? {
            initialTranslation: new Float32Array(
              editorState.initialTransform.position,
            ),
            initialHeading: editorState.initialTransform.headingRadians,
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
    const threshold = integerInput(
      replanThreshold,
      10,
      1,
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

  function clearMotionState(): void {
    currentMotion = null;
    referenceMotion = null;
    currentContinuation = null;
    currentProvenance = {};
    runtimeConstraints.clear();
    rebuildConstraintsAfterModelLoad = false;
    editorState = cloneEditorState(DEFAULT_EDITOR_STATE);
    viewer?.clearMotion();
    viewer?.applyEditorState(editorState);
    emptyState.hidden = false;
    motionBadge.removeAttribute("data-state");
    motionBadge.textContent = "No motion";
    runtimeMetric.hidden = true;
    correctionMetric.hidden = true;
    showReference.checked = false;
    showReference.disabled = true;
    viewer?.setReferenceMotion(null);
    referenceFileInput.value = "";
    exportSession.disabled = true;
    exportMotion.disabled = true;
    restartFromNow.disabled = true;
    updateConstraintTimeline();
    updateGenerateAvailability();
  }

  function addWaypointAt(
    frame: number,
    position: Vector3Tuple,
    headingRadians?: number,
    prefix = "waypoint",
  ): void {
    const waypoint: MotionWaypoint = {
      id: `${prefix}-${requestId("point")}`,
      frame: Math.max(0, Math.round(frame)),
      position,
      headingRadians,
      enabled: true,
    };
    setEditor({
      ...editorState,
      waypoints: [...editorState.waypoints, waypoint],
    });
  }

  function rebuildRuntimeConstraintsFromEditor(): void {
    runtimeConstraints.clear();
    if (!activeManifest || !currentMotion?.normalizedMotion) return;
    for (const marker of editorState.constraints) {
      const kind = markerRuntimeKind(marker);
      if (!kind) continue;
      try {
        if (kind === "root" && marker.position) {
          const valueKind = markerValueKind(marker);
          const base = createRootConstraint(activeManifest, {
              id: marker.id,
              frame: marker.startFrame,
              x: marker.position[0],
              z: marker.position[2],
              heading:
                valueKind === "position"
                  ? undefined
                  : headingFromQuaternion(marker.orientation),
            });
          runtimeConstraints.set(
            marker.id,
            filterConstraintValue(
              { ...base, endFrame: marker.endFrame },
              kind,
              valueKind,
              activeManifest,
            ),
          );
        } else if (kind !== "root") {
          const valueKind = markerValueKind(marker);
          runtimeConstraints.set(
            marker.id,
            filterConstraintValue(
              createCapturedConstraint(
              activeManifest,
              marker.id,
              kind,
              marker.startFrame,
              currentMotion.normalizedMotion,
              Math.min(marker.startFrame, currentMotion.frameCount - 1),
              marker.endFrame,
            ),
              kind,
              valueKind,
              activeManifest,
            ),
          );
        }
      } catch {
        // Imported playback remains usable even when its normalized feature
        // layout is not compatible with the active generation pack.
      }
    }
  }

  function addCapturedConstraint(): void {
    if (!activeManifest || !currentMotion?.normalizedMotion) {
      showError(
        "Constraint unavailable",
        "Generate or import motion with normalized features first.",
      );
      return;
    }
    try {
      const playbackFrame = viewer?.getPlaybackState().frame ?? 0;
      const targetFrame = integerInput(
        constraintFrame,
        playbackFrame,
        0,
        1_000_000,
      );
      const targetEndFrame = integerInput(
        constraintEndFrame,
        targetFrame,
        targetFrame,
        1_000_000,
      );
      const valueKind =
        selectedTrack === "full-body"
          ? "pose"
          : (constraintType.value as ConstraintValueKind);
      const id = requestId(`constraint-${selectedTrack}`);
      const joint = trackJointIndex(currentMotion, selectedTrack);
      const position = jointPositionAt(currentMotion, playbackFrame, joint);
      const orientation = jointOrientationAt(
        currentMotion,
        playbackFrame,
        joint,
      );
      let runtimeConstraint: MotionConstraint;
      if (selectedTrack === "root") {
        const heading =
          rootHeadingAt(currentMotion, activeManifest, playbackFrame) ??
          headingFromQuaternion(orientation);
        runtimeConstraint = filterConstraintValue(
          {
            ...createRootConstraint(activeManifest, {
          id,
          frame: targetFrame,
          x: position[0],
          z: position[2],
              heading: valueKind === "position" ? undefined : heading,
            }),
            endFrame: targetEndFrame,
          },
          selectedTrack,
          valueKind,
          activeManifest,
        );
      } else {
        runtimeConstraint = filterConstraintValue(
          createCapturedConstraint(
            activeManifest,
            id,
            selectedTrack,
            targetFrame,
            currentMotion.normalizedMotion,
            playbackFrame,
            targetEndFrame,
          ),
          selectedTrack,
          valueKind,
          activeManifest,
        );
      }
      runtimeConstraints.set(id, runtimeConstraint);
      const marker: MotionConstraintMarker = {
        id,
        kind:
          selectedTrack === "root"
            ? "root"
            : valueKind === "position"
              ? "position"
              : valueKind === "rotation"
                ? "orientation"
                : "transform",
        startFrame: targetFrame,
        endFrame: targetEndFrame,
        jointIndex: joint,
        position:
          selectedTrack === "root"
            ? [position[0], 0, position[2]]
            : position,
        orientation: valueKind === "position" ? undefined : orientation,
        label: markerLabel(selectedTrack, valueKind),
        enabled: true,
      };
      setEditor({
        ...editorState,
        constraints: [...editorState.constraints, marker],
      });
      announce(
        `Added a ${selectedTrack} ${valueKind} constraint at frames ${targetFrame}–${targetEndFrame}.`,
      );
    } catch (error) {
      showError(
        "Could not add constraint",
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  async function restoreSession(session: BrowserMotionSession): Promise<void> {
    currentMotion = session.motion;
    referenceMotion = null;
    viewer?.setReferenceMotion(null);
    currentContinuation = session.continuation ?? null;
    currentProvenance = session.provenance ?? {};
    const incompatibleContinuation =
      currentContinuation !== null &&
      modelReady &&
      modelInfo !== null &&
      !isContinuationModelCompatible(currentProvenance, modelInfo);
    if (incompatibleContinuation) {
      currentContinuation = null;
    }
    editorState = cloneEditorState(session.editor);
    if (session.provenance?.prompt) prompt.value = session.provenance.prompt;
    if (session.provenance?.seed !== undefined) {
      seed.value = String(session.provenance.seed);
    }
    initialX.value = String(editorState.initialTransform.position[0]);
    initialZ.value = String(editorState.initialTransform.position[2]);
    initialHeading.value = String(
      (editorState.initialTransform.headingRadians * 180) / Math.PI,
    );
    showSkeleton.checked = editorState.outputVisibility.skeleton;
    showMesh.checked = editorState.outputVisibility.mesh;
    showReference.checked =
      editorState.outputVisibility.reference && referenceMotion !== null;
    showContacts.checked = editorState.outputVisibility.contacts;
    showOrientations.checked = editorState.outputVisibility.orientationAxes;
    showTrajectory.checked = editorState.outputVisibility.trajectory;
    updatePrompt();
    updateSeed();
    if (session.generationConstraints !== undefined) {
      runtimeConstraints = new Map(
        session.generationConstraints.map((constraint) => [
          constraint.id,
          cloneRuntimeConstraint(constraint),
        ]),
      );
      rebuildConstraintsAfterModelLoad = false;
    } else {
      rebuildRuntimeConstraintsFromEditor();
      rebuildConstraintsAfterModelLoad = activeManifest === null;
    }
    refreshViewer(0, false, true, true);
    if (currentContinuation === null) {
      markCurrentMotionPlaybackOnly();
      if (modelReady) {
        resetWorkerSession();
      }
    } else if (modelReady) {
      restoreWorkerContinuation();
    }
    announce(
      incompatibleContinuation
        ? `Restored ${currentMotion.frameCount} frames for playback only because the saved continuation belongs to a different model pack.`
        : currentContinuation
          ? `Restored a continuable ${currentMotion.frameCount}-frame session.`
          : `Restored a playback-only ${currentMotion.frameCount}-frame session.`,
    );
  }

  async function importReferenceFile(file: File): Promise<void> {
    let motion: StructuredMotionResult;
    try {
      motion = decodeMotionJson(await file.text());
    } catch {
      try {
        motion = (await decodeSessionFile(file)).motion;
      } catch (error) {
        throw new Error(
          `Reference must be an ARDY motion JSON or session file. ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
    }
    referenceMotion = motion;
    showReference.disabled = false;
    showReference.checked = true;
    syncOutputVisibility();
    announce(`Loaded ${motion.frameCount} reference frames.`);
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
            if (pendingNewSession) {
              pendingNewSession = false;
              clearMotionState();
              resetWorkerSession();
              announce("Started a new motion session.");
            }
          }
          break;
        case "status":
          if (event.status.state === "empty" && !modelLoading) {
            modelReady = false;
          } else if (
            event.status.state === "ready" ||
            event.status.state === "generating"
          ) {
            applyCapabilities(event.status.capabilities);
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
          capabilities = null;
          resetRuntimeBadge();
          updateGenerateAvailability();
          break;
        default:
          break;
      }
    },
  );

  worker.addEventListener("error", (event) => {
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
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startGeneration("replace");
  });

  prompt.addEventListener("input", updatePrompt);
  duration.addEventListener("input", updateDuration);
  seed.addEventListener("input", updateSeed);
  targetBuffer.addEventListener("input", updateTargetBuffer);

  document
    .querySelectorAll<HTMLButtonElement>(".preset-chip")
    .forEach((button) => {
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
    const buffer = integerInput(replanBuffer, 20, 0, 1000);
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

  newSession.addEventListener("click", () => {
    if (
      currentMotion &&
      !window.confirm(
        "Start a new session? Export the current session first if you want to keep it.",
      )
    ) {
      return;
    }
    if (activeGeneration) {
      pendingNewSession = true;
      cancelGeneration.click();
      announce("Cancelling generation before starting a new session.");
      return;
    }
    clearMotionState();
    resetWorkerSession();
    announce("Started a new motion session.");
  });

  importSession.addEventListener("click", () => sessionFileInput.click());
  sessionFileInput.addEventListener("change", () => {
    const file = sessionFileInput.files?.[0];
    sessionFileInput.value = "";
    if (!file) return;
    if (activeGeneration) {
      showError(
        "Session import unavailable",
        "Cancel the active generation before importing a session.",
      );
      return;
    }
    void decodeSessionFile(file)
      .then(restoreSession)
      .catch((error) =>
        showError(
          "Could not import session",
          error instanceof Error ? error.message : String(error),
        ),
      );
  });

  exportSession.addEventListener("click", () => {
    if (!currentMotion) return;
    try {
      downloadSessionBinary(
        {
          motion: currentMotion,
          editor: editorState,
          generationConstraints: [...runtimeConstraints.values()],
          provenance: currentProvenance,
          continuation: currentContinuation ?? undefined,
        },
        "ardy-motion.ardysession",
      );
      announce("Session export started.");
    } catch (error) {
      showError(
        "Could not export session",
        error instanceof Error ? error.message : String(error),
      );
    }
  });

  exportMotion.addEventListener("click", () => {
    if (!currentMotion) return;
    try {
      downloadMotionJson(currentMotion, "ardy-motion.json");
      downloadMotionCsv(currentMotion, "ardy-motion.csv");
      announce("Motion JSON and CSV exports started.");
    } catch (error) {
      showError(
        "Could not export motion",
        error instanceof Error ? error.message : String(error),
      );
    }
  });

  for (const [kind, button] of trackButtons) {
    button.addEventListener("click", () => {
      selectedTrack = kind;
      for (const option of constraintType.options) {
        option.disabled =
          kind === "full-body"
            ? option.value !== "pose"
            : false;
      }
      if (kind === "full-body") constraintType.value = "pose";
      updateConstraintTimeline();
      announce(`Selected the ${button.textContent?.trim() ?? kind} constraint track.`);
    });
  }
  addConstraint.addEventListener("click", addCapturedConstraint);
  constraintFrame.addEventListener("input", () => {
    if (document.activeElement !== constraintEndFrame) {
      constraintEndFrame.value = constraintFrame.value;
    }
  });
  deleteConstraint.addEventListener("click", () => {
    const targetFrame = integerInput(
      constraintFrame,
      viewer?.getPlaybackState().frame ?? 0,
      0,
      1_000_000,
    );
    const matching = editorState.constraints
      .filter((marker) => markerRuntimeKind(marker) === selectedTrack)
      .sort(
        (left, right) =>
          Math.abs(left.startFrame - targetFrame) -
          Math.abs(right.startFrame - targetFrame),
      );
    const selected = matching[0];
    if (!selected) {
      announce(`No ${selectedTrack} constraint is available to delete.`);
      return;
    }
    runtimeConstraints.delete(selected.id);
    setEditor({
      ...editorState,
      constraints: editorState.constraints.filter(
        (marker) => marker.id !== selected.id,
      ),
    });
    announce(`Deleted the ${selectedTrack} constraint at frame ${selected.startFrame}.`);
  });
  clearConstraints.addEventListener("click", () => {
    if (
      (editorState.constraints.length > 0 ||
        editorState.waypoints.length > 0) &&
      !window.confirm("Clear every kinematic constraint?")
    ) {
      return;
    }
    runtimeConstraints.clear();
    setEditor({ ...editorState, constraints: [], waypoints: [] });
    announce("Cleared all kinematic constraints.");
  });

  [initialX, initialZ, initialHeading].forEach((input) =>
    input.addEventListener("change", updateInitialTransform),
  );

  addWaypoint.addEventListener("click", () => {
    const state = viewer?.getPlaybackState();
    const frame =
      (state?.frame ?? 0) +
      integerInput(waypointInterval, 20, 1, 10_000);
    const root = rootPositionAt(currentMotion, state?.frame ?? 0);
    addWaypointAt(
      frame,
      root,
      (finiteInput(targetHeading, 0, -180, 180) * Math.PI) / 180,
    );
    announce(`Added a root waypoint at frame ${frame}.`);
  });

  const handleGroundWaypoint = (point: Vector3Tuple): void => {
    if (!waypointMode.checked || !modelReady) return;
    const state = viewer?.getPlaybackState();
    const frame =
      (state?.frame ?? 0) +
      integerInput(waypointInterval, 20, 1, 10_000);
    addWaypointAt(
      frame,
      point,
      (finiteInput(targetHeading, 0, -180, 180) * Math.PI) / 180,
      "canvas-waypoint",
    );
    announce(`Added a ground-plane waypoint at frame ${frame}.`);
  };
  waypointMode.addEventListener("change", () => {
    if (viewer) {
      viewer.onGroundClick = waypointMode.checked
        ? handleGroundWaypoint
        : null;
    }
    announce(
      waypointMode.checked
        ? "Waypoint placement enabled. Click the ground plane to add a target."
        : "Waypoint placement disabled.",
    );
  });
  if (viewer) viewer.onGroundClick = null;

  applyTargetVelocity.addEventListener("click", () => {
    const state = viewer?.getPlaybackState();
    const frame = state?.frame ?? 0;
    const root = rootPositionAt(currentMotion, frame);
    const fps = currentMotion?.fps ?? modelInfo?.fps ?? 20;
    const previousRoot = rootPositionAt(
      currentMotion,
      Math.max(0, frame - 1),
    );
    const speed = finiteInput(targetVelocity, 0, 0, 10);
    const heading =
      (finiteInput(targetHeading, 0, -180, 180) * Math.PI) / 180;
    const waypoints = waypointsFromTargetVelocity({
      startFrame: frame,
      startX: root[0],
      startZ: root[2],
      startVelocityX: frame > 0 ? (root[0] - previousRoot[0]) * fps : 0,
      startVelocityZ: frame > 0 ? (root[2] - previousRoot[2]) * fps : 0,
      velocityX: Math.sin(heading) * speed,
      velocityZ: Math.cos(heading) * speed,
      fps,
      durationSeconds:
        integerInput(targetBuffer, 80, 1, 1000) / fps,
      transitionSeconds: 2,
      intervalFrames: integerInput(waypointInterval, 20, 1, 10_000),
      includeHeading: true,
    }).map<MotionWaypoint>((waypoint) => ({
      id: waypoint.id,
      frame: waypoint.frame,
      position: [waypoint.x, 0, waypoint.z],
      headingRadians: waypoint.heading,
      enabled: true,
    }));
    setEditor({
      ...editorState,
      waypoints: [
        ...editorState.waypoints.filter(
          (waypoint) => !waypoint.id.startsWith(`velocity:${frame}:`),
        ),
        ...waypoints,
      ],
    });
    announce(`Added ${waypoints.length} velocity waypoints.`);
  });

  const outputControls: Array<
    [HTMLInputElement, keyof ViewerOutputVisibility]
  > = [
    [showSkeleton, "skeleton"],
    [showMesh, "mesh"],
    [showContacts, "contacts"],
    [showOrientations, "orientationAxes"],
    [showTrajectory, "trajectory"],
    [showReference, "reference"],
  ];
  for (const [control] of outputControls) {
    control.addEventListener("change", syncOutputVisibility);
  }
  importReference.addEventListener("click", () =>
    referenceFileInput.click(),
  );
  referenceFileInput.addEventListener("change", () => {
    const file = referenceFileInput.files?.[0];
    referenceFileInput.value = "";
    if (!file) return;
    void importReferenceFile(file).catch((error) =>
      showError(
        "Could not import reference",
        error instanceof Error ? error.message : String(error),
      ),
    );
  });

  dismissError.addEventListener("click", () => clearError(true));
  dismissModelError.addEventListener("click", () => clearModelError(true));
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
  reducedMotion.addEventListener("change", handleReducedMotionChange);

  document.addEventListener("keydown", (event) => {
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
  });

  window.addEventListener("beforeunload", () => {
    postCommand({ type: "dispose", requestId: requestId("dispose") });
    worker.terminate();
    reducedMotion.removeEventListener(
      "change",
      handleReducedMotionChange,
    );
    viewer?.dispose();
  });

  updatePrompt();
  updateDuration();
  updateSeed();
  updateTargetBuffer();
  updateInitialTransform();
  updateConstraintTimeline();
  syncOutputVisibility();
  modelCard.setAttribute("aria-busy", "true");
  setModelStatus(
    "loading",
    "Checking local model cache",
    "No network request is made.",
    "Checking",
  );
  void readCachedModelPack()
    .then(async (files) => {
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
}

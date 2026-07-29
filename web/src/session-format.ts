// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  DEFAULT_EDITOR_STATE,
  normalizeEditorState,
  type MotionEditorState,
} from "./editor-state";
import {
  MAX_TRACK_VALUES,
  normalizeStructuredMotion,
  toContactArray,
  toFiniteFloat32Array,
  type StructuredMotionResult,
} from "./motion-data";
import type {
  MotionConstraint,
  MotionConstraintKind as RuntimeConstraintKind,
} from "./runtime/motion-constraint";
import type { PortableRandomState } from "./runtime/random";

export const MOTION_FILE_FORMAT = "ardy-browser-motion";
export const MOTION_FILE_VERSION = 1;
export const SESSION_FILE_FORMAT = "ardy-browser-session";
export const SESSION_FILE_VERSION = 1;

const SESSION_BINARY_MAGIC = "ARDYSES1";
const SESSION_BINARY_HEADER_BYTES = 16;
const MAX_JSON_CHARACTERS = 256 * 1024 * 1024;
const MAX_BINARY_BYTES = 512 * 1024 * 1024;
const MAX_BINARY_HEADER_BYTES = 8 * 1024 * 1024;
const MAX_CSV_CELLS = 20_000_000;
const MAX_SESSION_CONSTRAINTS = 10_000;
const MAX_CONSTRAINT_DIM = 65_536;
const MAX_SESSION_FRAME = 1_000_000;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

type UnknownRecord = Record<string, unknown>;

export interface MotionSessionProvenance {
  prompt?: string;
  seed?: number;
  modelId?: string;
  modelVariant?: string;
  createdAt?: string;
}

export interface MotionModelIdentity {
  id: string;
  variant: string;
}

export interface MotionContinuationState {
  hybridTokens: Float32Array;
  hybridDim: number;
  frameCount: number;
  random: PortableRandomState;
  initialTranslation: [number, number, number];
  initialHeading: number;
}

export interface BrowserMotionSession {
  format: typeof SESSION_FILE_FORMAT;
  version: typeof SESSION_FILE_VERSION;
  motion: StructuredMotionResult;
  editor: MotionEditorState;
  generationConstraints?: readonly MotionConstraint[];
  provenance?: MotionSessionProvenance;
  continuation?: MotionContinuationState;
}

export interface MotionSessionInput {
  motion: StructuredMotionResult;
  editor?: MotionEditorState;
  generationConstraints?: readonly MotionConstraint[];
  provenance?: MotionSessionProvenance;
  continuation?: MotionContinuationState;
}

export type SessionRestoreMode = "continuable" | "playback-only";

interface BinaryArrayReference {
  $array: "float32" | "uint8";
  offset: number;
  length: number;
}

type ArrayWire = number[] | BinaryArrayReference;

interface RotationWire {
  values: ArrayWire;
  shape: readonly number[];
  format: string;
}

interface MotionWire {
  skeleton: unknown;
  positions: ArrayWire;
  positionsShape: readonly number[];
  frameCount: number;
  fps: number;
  normalizedMotion?: ArrayWire;
  normalizedMotionShape?: readonly number[];
  localRotations?: RotationWire;
  globalRotations?: RotationWire;
  roots?: ArrayWire;
  rootsShape?: readonly number[];
  contacts?: ArrayWire;
  contactsShape?: readonly number[];
}

interface MotionFileWire {
  format: typeof MOTION_FILE_FORMAT;
  version: typeof MOTION_FILE_VERSION;
  motion: MotionWire;
}

interface SessionWire {
  format: typeof SESSION_FILE_FORMAT;
  version: typeof SESSION_FILE_VERSION;
  motion: MotionWire;
  editor: unknown;
  generationConstraints?: Array<{
    id: string;
    kind: RuntimeConstraintKind;
    frame: number;
    endFrame?: number;
    values: ArrayWire;
    mask: ArrayWire;
  }>;
  provenance?: unknown;
  continuation?: {
    hybridTokens: ArrayWire;
    hybridDim: number;
    frameCount: number;
    random: PortableRandomState;
    initialTranslation: [number, number, number];
    initialHeading: number;
  };
  binaryEncoding?: "little-endian";
}

type ArrayWriter = (values: Float32Array | Uint8Array) => ArrayWire;
type FloatResolver = (value: unknown, label: string) => Float32Array;
type ContactResolver = (value: unknown, label: string) => Uint8Array;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function boundedString(
  value: unknown,
  label: string,
  maxLength: number,
): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length > maxLength) {
    throw new TypeError(`${label} must be a string of at most ${maxLength} characters.`);
  }
  return value;
}

function normalizeProvenance(value: unknown): MotionSessionProvenance | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value)) throw new TypeError("Session provenance must be an object.");
  let seed: number | undefined;
  if (value.seed !== undefined) {
    if (typeof value.seed !== "number" || !Number.isSafeInteger(value.seed)) {
      throw new RangeError("Session seed must be a safe integer.");
    }
    seed = value.seed;
  }
  const createdAt = boundedString(value.createdAt ?? value.created_at, "Session creation time", 128);
  if (createdAt !== undefined && Number.isNaN(Date.parse(createdAt))) {
    throw new RangeError("Session creation time must be an ISO-compatible date string.");
  }
  return Object.freeze({
    prompt: boundedString(value.prompt, "Session prompt", 16_384),
    seed,
    modelId: boundedString(value.modelId ?? value.model_id, "Session model id", 512),
    modelVariant: boundedString(
      value.modelVariant ?? value.model_variant,
      "Session model variant",
      512,
    ),
    createdAt,
  });
}

const RUNTIME_CONSTRAINT_KINDS = new Set<RuntimeConstraintKind>([
  "root",
  "full-body",
  "left-hand",
  "right-hand",
  "left-foot",
  "right-foot",
]);

function normalizeGenerationConstraints(
  constraints: readonly MotionConstraint[] | undefined,
): readonly MotionConstraint[] | undefined {
  if (constraints === undefined) return undefined;
  if (constraints.length > MAX_SESSION_CONSTRAINTS) {
    throw new RangeError(
      `Session contains more than ${MAX_SESSION_CONSTRAINTS.toLocaleString()} generation constraints.`,
    );
  }
  const ids = new Set<string>();
  return Object.freeze(constraints.map((constraint, index) => {
    const label = `Generation constraint ${index}`;
    const id =
      typeof constraint.id === "string" ? constraint.id.trim() : "";
    if (
      id.length === 0 ||
      id.length > 256
    ) {
      throw new TypeError(`${label} id must be a non-empty string of at most 256 characters.`);
    }
    if (ids.has(id)) {
      throw new RangeError(`Generation constraint id "${id}" is duplicated.`);
    }
    ids.add(id);
    if (!RUNTIME_CONSTRAINT_KINDS.has(constraint.kind)) {
      throw new TypeError(`${label} has an unsupported kind "${String(constraint.kind)}".`);
    }
    const endFrame = constraint.endFrame ?? constraint.frame;
    if (
      !Number.isSafeInteger(constraint.frame) ||
      !Number.isSafeInteger(endFrame) ||
      constraint.frame < 0 ||
      constraint.frame > MAX_SESSION_FRAME ||
      endFrame > MAX_SESSION_FRAME ||
      endFrame < constraint.frame
    ) {
      throw new RangeError(`${label} has invalid frame bounds.`);
    }
    if (
      !(constraint.values instanceof Float32Array) ||
      !(constraint.mask instanceof Float32Array) ||
      constraint.values.length < 1 ||
      constraint.values.length > MAX_CONSTRAINT_DIM ||
      constraint.values.length !== constraint.mask.length
    ) {
      throw new RangeError(
        `${label} values and mask must contain the same 1–${MAX_CONSTRAINT_DIM.toLocaleString()} float values.`,
      );
    }
    let hasObservedFeature = false;
    for (let feature = 0; feature < constraint.values.length; feature += 1) {
      const value = constraint.values[feature];
      const mask = constraint.mask[feature];
      if (!Number.isFinite(value) || !Number.isFinite(mask)) {
        throw new RangeError(`${label} contains a non-finite value.`);
      }
      if (mask < 0 || mask > 1) {
        throw new RangeError(`${label} mask values must be between 0 and 1.`);
      }
      hasObservedFeature ||= mask > 0.5;
    }
    if (!hasObservedFeature) {
      throw new RangeError(`${label} must observe at least one motion feature.`);
    }
    return Object.freeze({
      id,
      kind: constraint.kind,
      frame: constraint.frame,
      ...(constraint.endFrame === undefined ? {} : { endFrame: constraint.endFrame }),
      values: new Float32Array(constraint.values),
      mask: new Float32Array(constraint.mask),
    });
  }));
}

function generationConstraintsToWire(
  constraints: readonly MotionConstraint[] | undefined,
  writeArray: ArrayWriter,
): SessionWire["generationConstraints"] {
  return normalizeGenerationConstraints(constraints)?.map((constraint) => ({
    id: constraint.id,
    kind: constraint.kind,
    frame: constraint.frame,
    ...(constraint.endFrame === undefined ? {} : { endFrame: constraint.endFrame }),
    values: writeArray(constraint.values),
    mask: writeArray(constraint.mask),
  }));
}

function generationConstraintsFromWire(
  value: unknown,
  resolveFloat: FloatResolver,
): readonly MotionConstraint[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) {
    throw new TypeError("Session generation constraints must be an array.");
  }
  if (value.length > MAX_SESSION_CONSTRAINTS) {
    throw new RangeError(
      `Session contains more than ${MAX_SESSION_CONSTRAINTS.toLocaleString()} generation constraints.`,
    );
  }
  const raw: MotionConstraint[] = value.map((item, index) => {
    if (!isRecord(item)) {
      throw new TypeError(`Generation constraint ${index} must be an object.`);
    }
    const constraint: MotionConstraint = {
      id: item.id as string,
      kind: item.kind as RuntimeConstraintKind,
      frame: item.frame as number,
      ...(item.endFrame === undefined && item.end_frame === undefined
        ? {}
        : { endFrame: (item.endFrame ?? item.end_frame) as number }),
      values: resolveFloat(item.values, `Generation constraint ${index} values`),
      mask: resolveFloat(item.mask, `Generation constraint ${index} mask`),
    };
    return constraint;
  });
  return normalizeGenerationConstraints(raw) ?? Object.freeze([]);
}

function shape(value: unknown, expected: readonly number[], label: string): void {
  if (!Array.isArray(value) || value.length !== expected.length) {
    throw new RangeError(`${label} must be [${expected.join(", ")}].`);
  }
  value.forEach((component, index) => {
    if (typeof component !== "number" || component !== expected[index]) {
      throw new RangeError(`${label} must be [${expected.join(", ")}].`);
    }
  });
}

function skeletonToWire(motion: StructuredMotionResult): unknown {
  return {
    id: motion.skeleton.id,
    name: motion.skeleton.name,
    jointNames: [...motion.skeleton.jointNames],
    parents: [...motion.skeleton.parents],
    rootJointIndex: motion.skeleton.rootJointIndex,
    contactJointIndices: [...motion.skeleton.contactJointIndices],
    contactNames: [...motion.skeleton.contactNames],
  };
}

function motionToWire(motion: StructuredMotionResult, writeArray: ArrayWriter): MotionWire {
  return {
    skeleton: skeletonToWire(motion),
    positions: writeArray(motion.positions),
    positionsShape: [...motion.positionsShape],
    frameCount: motion.frameCount,
    fps: motion.fps,
    normalizedMotion: motion.normalizedMotion
      ? writeArray(motion.normalizedMotion)
      : undefined,
    normalizedMotionShape: motion.normalizedMotionShape
      ? [...motion.normalizedMotionShape]
      : undefined,
    localRotations: motion.localRotations
      ? {
          values: writeArray(motion.localRotations.values),
          shape: [...motion.localRotations.shape],
          format: motion.localRotations.format,
        }
      : undefined,
    globalRotations: motion.globalRotations
      ? {
          values: writeArray(motion.globalRotations.values),
          shape: [...motion.globalRotations.shape],
          format: motion.globalRotations.format,
        }
      : undefined,
    roots: motion.roots ? writeArray(motion.roots) : undefined,
    rootsShape: motion.rootsShape ? [...motion.rootsShape] : undefined,
    contacts: motion.contacts ? writeArray(motion.contacts) : undefined,
    contactsShape: motion.contactsShape ? [...motion.contactsShape] : undefined,
  };
}

function motionFromWire(
  value: unknown,
  resolveFloat: FloatResolver,
  resolveContacts: ContactResolver,
): StructuredMotionResult {
  if (!isRecord(value)) throw new TypeError("Session motion must be an object.");
  const local = value.localRotations ?? value.local_rotations;
  const global = value.globalRotations ?? value.global_rotations;
  if (local !== undefined && !isRecord(local)) {
    throw new TypeError("Session local rotations must be an object.");
  }
  if (global !== undefined && !isRecord(global)) {
    throw new TypeError("Session global rotations must be an object.");
  }
  const frameCount = value.frameCount ?? value.frame_count;
  if (typeof frameCount !== "number" || !Number.isInteger(frameCount)) {
    throw new TypeError("Session motion frame count must be an integer.");
  }
  if (typeof value.fps !== "number" || !Number.isFinite(value.fps)) {
    throw new TypeError("Session motion FPS must be a finite number.");
  }
  const raw = {
    skeleton: value.skeleton,
    positions: resolveFloat(value.positions, "Motion joint positions"),
    frameCount,
    fps: value.fps,
    normalizedMotion:
      value.normalizedMotion === undefined && value.normalized_motion === undefined
        ? undefined
        : resolveFloat(
            value.normalizedMotion ?? value.normalized_motion,
            "Normalized motion",
          ),
    localRotations:
      local === undefined
        ? undefined
        : resolveFloat(local.values, "Local rotations"),
    localRotationFormat: local === undefined ? undefined : local.format,
    globalRotations:
      global === undefined
        ? undefined
        : resolveFloat(global.values, "Global rotations"),
    globalRotationFormat: global === undefined ? undefined : global.format,
    roots:
      value.roots === undefined ? undefined : resolveFloat(value.roots, "Root track"),
    contacts:
      value.contacts === undefined
        ? undefined
        : resolveContacts(value.contacts, "Foot contacts"),
  };
  const motion = normalizeStructuredMotion(raw);
  shape(value.positionsShape ?? value.positions_shape, motion.positionsShape, "Motion position shape");
  if (motion.normalizedMotionShape) {
    shape(
      value.normalizedMotionShape ?? value.normalized_motion_shape,
      motion.normalizedMotionShape,
      "Normalized motion shape",
    );
  }
  if (motion.localRotations && isRecord(local)) {
    shape(local.shape, motion.localRotations.shape, "Local rotation shape");
  }
  if (motion.globalRotations && isRecord(global)) {
    shape(global.shape, motion.globalRotations.shape, "Global rotation shape");
  }
  if (motion.rootsShape) {
    shape(value.rootsShape ?? value.roots_shape, motion.rootsShape, "Root track shape");
  }
  if (motion.contactsShape) {
    shape(value.contactsShape ?? value.contacts_shape, motion.contactsShape, "Contact shape");
  }
  return motion;
}

function normalizeContinuation(
  value: unknown,
  frameCount: number,
  resolveFloat: FloatResolver,
): MotionContinuationState | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value)) throw new TypeError("Session continuation state must be an object.");
  const hybridDim = value.hybridDim ?? value.hybrid_dim;
  if (
    typeof hybridDim !== "number" ||
    !Number.isInteger(hybridDim) ||
    hybridDim < 1 ||
    hybridDim > 65_536
  ) {
    throw new RangeError("Continuation hybrid dimension must be an integer between 1 and 65536.");
  }
  const hybridTokens = resolveFloat(
    value.hybridTokens ?? value.hybrid_tokens,
    "Continuation hybrid tokens",
  );
  if (
    hybridTokens.length % hybridDim !== 0 ||
    (frameCount === 0 && hybridTokens.length !== 0) ||
    (frameCount > 0 &&
      (hybridTokens.length === 0 ||
        hybridTokens.length / hybridDim > frameCount))
  ) {
    throw new RangeError(
      `Continuation hybrid tokens must contain complete ${hybridDim}-value tokens compatible with ${frameCount} frames.`,
    );
  }
  const continuationFrameCount = value.frameCount ?? value.frame_count;
  if (
    typeof continuationFrameCount !== "number" ||
    !Number.isSafeInteger(continuationFrameCount) ||
    continuationFrameCount !== frameCount
  ) {
    throw new RangeError("Continuation frame count must match the motion clip.");
  }
  const randomValue = value.random ?? value.random_state;
  if (!isRecord(randomValue)) {
    throw new TypeError("Continuation random state must be an object.");
  }
  const randomSeed = randomValue.seed;
  const randomState = randomValue.state;
  const spareNormal = randomValue.spareNormal ?? randomValue.spare_normal;
  if (
    typeof randomSeed !== "number" ||
    !Number.isSafeInteger(randomSeed) ||
    randomSeed < 0 ||
    randomSeed > 0xffff_ffff ||
    typeof randomState !== "number" ||
    !Number.isSafeInteger(randomState) ||
    randomState < 0 ||
    randomState > 0xffff_ffff ||
    (spareNormal !== undefined &&
      (typeof spareNormal !== "number" || !Number.isFinite(spareNormal)))
  ) {
    throw new RangeError("Continuation random state is invalid.");
  }
  const translationValue =
    value.initialTranslation ?? value.initial_translation;
  if (!Array.isArray(translationValue) || translationValue.length !== 3) {
    throw new TypeError("Continuation initial translation must contain three numbers.");
  }
  const translation = translationValue.map((component) => {
    if (typeof component !== "number" || !Number.isFinite(component)) {
      throw new TypeError("Continuation initial translation must be finite.");
    }
    return component;
  }) as [number, number, number];
  const initialHeading = value.initialHeading ?? value.initial_heading;
  if (typeof initialHeading !== "number" || !Number.isFinite(initialHeading)) {
    throw new TypeError("Continuation initial heading must be finite.");
  }
  return Object.freeze({
    hybridTokens,
    hybridDim,
    frameCount: continuationFrameCount,
    random: Object.freeze({
      seed: randomSeed,
      state: randomState,
      ...(spareNormal === undefined ? {} : { spareNormal }),
    }),
    initialTranslation: translation,
    initialHeading,
  });
}

function jsonFloatResolver(value: unknown, label: string): Float32Array {
  if (isRecord(value) && "$array" in value) {
    throw new TypeError(`${label} contains a binary reference in a JSON-only file.`);
  }
  return toFiniteFloat32Array(value, label);
}

function jsonContactResolver(value: unknown, label: string): Uint8Array {
  if (isRecord(value) && "$array" in value) {
    throw new TypeError(`${label} contains a binary reference in a JSON-only file.`);
  }
  return toContactArray(value, label);
}

function parseJson(text: string, label: string): unknown {
  if (text.length > MAX_JSON_CHARACTERS) {
    throw new RangeError(`${label} exceeds the ${MAX_JSON_CHARACTERS.toLocaleString()} character safety limit.`);
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new SyntaxError(`${label} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function sessionFromWire(
  value: unknown,
  resolveFloat: FloatResolver,
  resolveContacts: ContactResolver,
): BrowserMotionSession {
  if (!isRecord(value)) throw new TypeError("Session file must contain an object.");
  if (value.format !== SESSION_FILE_FORMAT) {
    throw new RangeError(`Unsupported session format "${String(value.format)}".`);
  }
  if (value.version !== SESSION_FILE_VERSION) {
    throw new RangeError(`Unsupported session version "${String(value.version)}".`);
  }
  const motion = motionFromWire(value.motion, resolveFloat, resolveContacts);
  return Object.freeze({
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    motion,
    editor: normalizeEditorState(value.editor ?? DEFAULT_EDITOR_STATE, {
      jointCount: motion.skeleton.jointNames.length,
    }),
    generationConstraints: generationConstraintsFromWire(
      value.generationConstraints ?? value.generation_constraints,
      resolveFloat,
    ),
    provenance: normalizeProvenance(value.provenance),
    continuation: normalizeContinuation(value.continuation, motion.frameCount, resolveFloat),
  });
}

export function createMotionSession(input: MotionSessionInput): BrowserMotionSession {
  const wire: SessionWire = {
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    motion: motionToWire(input.motion, (values) => Array.from(values)),
    editor: input.editor ?? DEFAULT_EDITOR_STATE,
    generationConstraints: generationConstraintsToWire(
      input.generationConstraints,
      (values) => Array.from(values),
    ),
    provenance: input.provenance,
    continuation: input.continuation
      ? {
          hybridTokens: Array.from(input.continuation.hybridTokens),
          hybridDim: input.continuation.hybridDim,
          frameCount: input.continuation.frameCount,
          random: input.continuation.random,
          initialTranslation: input.continuation.initialTranslation,
          initialHeading: input.continuation.initialHeading,
        }
      : undefined,
  };
  return sessionFromWire(wire, jsonFloatResolver, jsonContactResolver);
}

export function encodeSessionJson(session: MotionSessionInput | BrowserMotionSession, pretty = false): string {
  const normalized = createMotionSession(session);
  const wire: SessionWire = {
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    motion: motionToWire(normalized.motion, (values) => Array.from(values)),
    editor: normalized.editor,
    generationConstraints: generationConstraintsToWire(
      normalized.generationConstraints,
      (values) => Array.from(values),
    ),
    provenance: normalized.provenance,
    continuation: normalized.continuation
      ? {
          hybridTokens: Array.from(normalized.continuation.hybridTokens),
          hybridDim: normalized.continuation.hybridDim,
          frameCount: normalized.continuation.frameCount,
          random: normalized.continuation.random,
          initialTranslation: normalized.continuation.initialTranslation,
          initialHeading: normalized.continuation.initialHeading,
        }
      : undefined,
  };
  return JSON.stringify(wire, null, pretty ? 2 : undefined);
}

export function decodeSessionJson(text: string): BrowserMotionSession {
  return sessionFromWire(parseJson(text, "Session file"), jsonFloatResolver, jsonContactResolver);
}

function byteLengthFor(values: Float32Array | Uint8Array): number {
  return values instanceof Float32Array
    ? values.length * Float32Array.BYTES_PER_ELEMENT
    : values.length;
}

function writeArrayLittleEndian(
  target: Uint8Array,
  targetOffset: number,
  values: Float32Array | Uint8Array,
): void {
  if (values instanceof Uint8Array) {
    target.set(values, targetOffset);
    return;
  }
  const view = new DataView(target.buffer, target.byteOffset + targetOffset, values.length * 4);
  for (let index = 0; index < values.length; index += 1) {
    view.setFloat32(index * 4, values[index], true);
  }
}

export function encodeSessionBinary(
  session: MotionSessionInput | BrowserMotionSession,
): Uint8Array {
  const normalized = createMotionSession(session);
  const chunks: Array<Float32Array | Uint8Array> = [];
  let payloadLength = 0;
  const writeArray: ArrayWriter = (values) => {
    const reference: BinaryArrayReference = {
      $array: values instanceof Float32Array ? "float32" : "uint8",
      offset: payloadLength,
      length: values.length,
    };
    payloadLength += byteLengthFor(values);
    chunks.push(values);
    return reference;
  };
  const wire: SessionWire = {
    format: SESSION_FILE_FORMAT,
    version: SESSION_FILE_VERSION,
    binaryEncoding: "little-endian",
    motion: motionToWire(normalized.motion, writeArray),
    editor: normalized.editor,
    generationConstraints: generationConstraintsToWire(
      normalized.generationConstraints,
      writeArray,
    ),
    provenance: normalized.provenance,
    continuation: normalized.continuation
      ? {
          hybridTokens: writeArray(normalized.continuation.hybridTokens),
          hybridDim: normalized.continuation.hybridDim,
          frameCount: normalized.continuation.frameCount,
          random: normalized.continuation.random,
          initialTranslation: normalized.continuation.initialTranslation,
          initialHeading: normalized.continuation.initialHeading,
        }
      : undefined,
  };
  const header = encoder.encode(JSON.stringify(wire));
  if (header.length > MAX_BINARY_HEADER_BYTES) {
    throw new RangeError("Binary session metadata is too large.");
  }
  const totalLength = SESSION_BINARY_HEADER_BYTES + header.length + payloadLength;
  if (totalLength > MAX_BINARY_BYTES) throw new RangeError("Binary session exceeds the safety size limit.");
  const result = new Uint8Array(totalLength);
  result.set(encoder.encode(SESSION_BINARY_MAGIC), 0);
  const headerView = new DataView(result.buffer);
  headerView.setUint32(8, header.length, true);
  headerView.setUint32(12, payloadLength, true);
  result.set(header, SESSION_BINARY_HEADER_BYTES);
  let offset = SESSION_BINARY_HEADER_BYTES + header.length;
  for (const chunk of chunks) {
    writeArrayLittleEndian(result, offset, chunk);
    offset += byteLengthFor(chunk);
  }
  return result;
}

function binaryBytes(input: ArrayBuffer | Uint8Array): Uint8Array {
  return input instanceof Uint8Array
    ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
    : new Uint8Array(input);
}

export function decodeSessionBinary(input: ArrayBuffer | Uint8Array): BrowserMotionSession {
  const bytes = binaryBytes(input);
  if (bytes.length < SESSION_BINARY_HEADER_BYTES || bytes.length > MAX_BINARY_BYTES) {
    throw new RangeError("Binary session has an invalid size.");
  }
  const magic = decoder.decode(bytes.subarray(0, 8));
  if (magic !== SESSION_BINARY_MAGIC) throw new RangeError("Binary session magic is invalid.");
  const headerView = new DataView(bytes.buffer, bytes.byteOffset, SESSION_BINARY_HEADER_BYTES);
  const headerLength = headerView.getUint32(8, true);
  const payloadLength = headerView.getUint32(12, true);
  if (
    headerLength === 0 ||
    headerLength > MAX_BINARY_HEADER_BYTES ||
    SESSION_BINARY_HEADER_BYTES + headerLength + payloadLength !== bytes.length
  ) {
    throw new RangeError("Binary session header or payload length is invalid.");
  }
  const wire = parseJson(
    decoder.decode(bytes.subarray(SESSION_BINARY_HEADER_BYTES, SESSION_BINARY_HEADER_BYTES + headerLength)),
    "Binary session metadata",
  );
  if (!isRecord(wire) || wire.binaryEncoding !== "little-endian") {
    throw new RangeError("Binary session encoding is unsupported.");
  }
  const payloadOffset = SESSION_BINARY_HEADER_BYTES + headerLength;
  const resolveReference = (
    value: unknown,
    expectedType: "float32" | "uint8",
    label: string,
  ): Float32Array | Uint8Array => {
    if (!isRecord(value) || value.$array !== expectedType) {
      throw new TypeError(`${label} does not contain a valid ${expectedType} binary reference.`);
    }
    const offset = value.offset;
    const length = value.length;
    if (
      typeof offset !== "number" ||
      typeof length !== "number" ||
      !Number.isInteger(offset) ||
      !Number.isInteger(length) ||
      offset < 0 ||
      length < 0 ||
      length > MAX_TRACK_VALUES
    ) {
      throw new RangeError(`${label} binary reference is outside the allowed range.`);
    }
    const byteLength = expectedType === "float32" ? length * 4 : length;
    if (offset + byteLength > payloadLength) {
      throw new RangeError(`${label} binary reference exceeds the session payload.`);
    }
    if (expectedType === "uint8") {
      return bytes.slice(payloadOffset + offset, payloadOffset + offset + byteLength);
    }
    const values = new Float32Array(length);
    const source = new DataView(bytes.buffer, bytes.byteOffset + payloadOffset + offset, byteLength);
    for (let index = 0; index < length; index += 1) {
      values[index] = source.getFloat32(index * 4, true);
      if (!Number.isFinite(values[index])) {
        throw new RangeError(`${label} contains a non-finite value at index ${index}.`);
      }
    }
    return values;
  };
  const floatResolver: FloatResolver = (value, label) =>
    resolveReference(value, "float32", label) as Float32Array;
  const contactResolver: ContactResolver = (value, label) =>
    resolveReference(value, "uint8", label) as Uint8Array;
  return sessionFromWire(wire, floatResolver, contactResolver);
}

export function isContinuationModelCompatible(
  provenance: MotionSessionProvenance | undefined,
  model: MotionModelIdentity,
): boolean {
  return (
    (provenance?.modelId === undefined ||
      provenance.modelId === model.id) &&
    (provenance?.modelVariant === undefined ||
      provenance.modelVariant === model.variant)
  );
}

export function getSessionRestoreMode(
  session: BrowserMotionSession,
  model?: MotionModelIdentity,
): SessionRestoreMode {
  return session.continuation &&
    (model === undefined ||
      isContinuationModelCompatible(session.provenance, model))
    ? "continuable"
    : "playback-only";
}

/**
 * Import either portable JSON or the ARDY binary container from a File/Blob.
 * The content signature, not the filename extension, selects the decoder.
 */
export async function decodeSessionFile(file: Blob): Promise<BrowserMotionSession> {
  if (file.size <= 0 || file.size > MAX_BINARY_BYTES) {
    throw new RangeError("Session file has an invalid size.");
  }
  const prefix = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  const isBinary =
    prefix.length === 8 && decoder.decode(prefix) === SESSION_BINARY_MAGIC;
  if (isBinary) return decodeSessionBinary(await file.arrayBuffer());
  return decodeSessionJson(await file.text());
}

export function encodeMotionJson(motion: StructuredMotionResult, pretty = false): string {
  const canonical = motionFromWire(
    motionToWire(motion, (values) => Array.from(values)),
    jsonFloatResolver,
    jsonContactResolver,
  );
  const wire: MotionFileWire = {
    format: MOTION_FILE_FORMAT,
    version: MOTION_FILE_VERSION,
    motion: motionToWire(canonical, (values) => Array.from(values)),
  };
  return JSON.stringify(wire, null, pretty ? 2 : undefined);
}

export function decodeMotionJson(text: string): StructuredMotionResult {
  const value = parseJson(text, "Motion file");
  if (!isRecord(value)) throw new TypeError("Motion file must contain an object.");
  if (value.format !== MOTION_FILE_FORMAT || value.version !== MOTION_FILE_VERSION) {
    throw new RangeError("Unsupported motion file format or version.");
  }
  return motionFromWire(value.motion, jsonFloatResolver, jsonContactResolver);
}

function csvCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

export function encodeMotionCsv(motion: StructuredMotionResult): string {
  const jointCount = motion.skeleton.jointNames.length;
  const contactCount = motion.skeleton.contactJointIndices.length;
  const rootComponents = motion.rootsShape?.[1] ?? 0;
  const columnsPerFrame = 2 + jointCount * 3 + rootComponents + contactCount;
  if (motion.frameCount * columnsPerFrame > MAX_CSV_CELLS) {
    throw new RangeError(
      `CSV export exceeds the ${MAX_CSV_CELLS.toLocaleString()} cell safety limit; use JSON or binary session export.`,
    );
  }
  const header = ["frame", "time_seconds"];
  for (const jointName of motion.skeleton.jointNames) {
    header.push(
      `position:${jointName}:x`,
      `position:${jointName}:y`,
      `position:${jointName}:z`,
    );
  }
  for (let component = 0; component < rootComponents; component += 1) {
    header.push(`root:${component}`);
  }
  for (const contactName of motion.skeleton.contactNames) {
    header.push(`contact:${contactName}`);
  }
  const rows = [header.map(csvCell).join(",")];
  const positionComponents = jointCount * 3;
  for (let frame = 0; frame < motion.frameCount; frame += 1) {
    const row: string[] = [String(frame), String(frame / motion.fps)];
    const positionOffset = frame * positionComponents;
    for (let index = 0; index < positionComponents; index += 1) {
      row.push(String(motion.positions[positionOffset + index]));
    }
    const rootOffset = frame * rootComponents;
    for (let index = 0; index < rootComponents; index += 1) {
      row.push(String(motion.roots?.[rootOffset + index] ?? 0));
    }
    const contactOffset = frame * contactCount;
    for (let index = 0; index < contactCount; index += 1) {
      row.push(String(motion.contacts?.[contactOffset + index] ?? 0));
    }
    rows.push(row.join(","));
  }
  return rows.join("\n");
}

export function safeDownloadFilename(name: string, fallback: string): string {
  const sanitized = name
    .normalize("NFKC")
    .replaceAll(/[\u0000-\u001f<>:"/\\|?*]+/g, "-")
    .replaceAll(/\s+/g, "-")
    .replaceAll(/-+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 120);
  return sanitized || fallback;
}

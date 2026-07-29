// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const DEFAULT_MOTION_FPS = 20;
export const MAX_MOTION_FRAMES = 1_000_000;
export const MAX_SKELETON_JOINTS = 1_024;
export const MAX_TRACK_VALUES = 100_000_000;

export const CORE27_JOINT_NAMES = Object.freeze([
  "Hips",
  "Spine",
  "Spine1",
  "Spine2",
  "Spine3",
  "Neck",
  "Head",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "RightHandEnd",
  "RightHandThumb1",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "LeftHandEnd",
  "LeftHandThumb1",
  "RightUpLeg",
  "RightLeg",
  "RightFoot",
  "RightToeBase",
  "LeftUpLeg",
  "LeftLeg",
  "LeftFoot",
  "LeftToeBase",
] as const);

export const CORE27_JOINT_COUNT = CORE27_JOINT_NAMES.length;
export const CORE27_PARENTS = Object.freeze([
  -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25,
]);
export const CORE27_FOOT_CONTACT_JOINTS = Object.freeze([25, 26, 21, 22]);
export const CORE27_FOOT_CONTACT_NAMES = Object.freeze([
  "LeftFoot",
  "LeftToeBase",
  "RightFoot",
  "RightToeBase",
]);

export interface SkeletonMetadata {
  id: string;
  name: string;
  jointNames: readonly string[];
  parents: readonly number[];
  rootJointIndex: number;
  contactJointIndices: readonly number[];
  contactNames: readonly string[];
}

export const CORE27_SKELETON: SkeletonMetadata = Object.freeze({
  id: "cskel27",
  name: "Core27",
  jointNames: CORE27_JOINT_NAMES,
  parents: CORE27_PARENTS,
  rootJointIndex: 0,
  contactJointIndices: CORE27_FOOT_CONTACT_JOINTS,
  contactNames: CORE27_FOOT_CONTACT_NAMES,
});

export type RotationFormat = "quaternion-xyzw" | "matrix3x3-row-major";

export interface RotationTrack {
  values: Float32Array;
  shape: readonly [frames: number, joints: number, components: 4 | 9];
  format: RotationFormat;
}

/**
 * Canonical, renderer- and persistence-friendly result of motion generation.
 *
 * Arrays omit the model's singleton batch dimension. Shapes make every
 * optional track self-describing and prevent callers from guessing layouts.
 */
export interface StructuredMotionResult {
  skeleton: SkeletonMetadata;
  positions: Float32Array;
  positionsShape: readonly [frames: number, joints: number, xyz: 3];
  frameCount: number;
  fps: number;
  normalizedMotion?: Float32Array;
  normalizedMotionShape?: readonly [frames: number, features: number];
  localRotations?: RotationTrack;
  globalRotations?: RotationTrack;
  roots?: Float32Array;
  rootsShape?: readonly [frames: number, components: number];
  contacts?: Uint8Array;
  contactsShape?: readonly [frames: number, channels: number];
}

type UnknownRecord = Record<string, unknown>;

export interface MotionNormalizationOptions {
  skeleton?: unknown;
  defaultFps?: number;
}

export function isJointInContact(
  motion: StructuredMotionResult,
  frame: number,
  jointIndex: number,
): boolean {
  if (
    !motion.contacts ||
    !Number.isInteger(frame) ||
    frame < 0 ||
    frame >= motion.frameCount ||
    !Number.isInteger(jointIndex)
  ) {
    return false;
  }
  const channel = motion.skeleton.contactJointIndices.indexOf(jointIndex);
  if (channel === -1) return false;
  return Boolean(motion.contacts[frame * motion.skeleton.contactJointIndices.length + channel]);
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function valueAt(record: UnknownRecord, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined) return record[key];
  }
  return undefined;
}

function flattenNumericArray(
  value: unknown,
  label: string,
  allowBooleans: boolean,
): number[] {
  const flattened: number[] = [];
  const pending: unknown[] = [value];
  while (pending.length > 0) {
    const item = pending.pop();
    if (Array.isArray(item)) {
      for (let index = item.length - 1; index >= 0; index -= 1) {
        pending.push(item[index]);
      }
      continue;
    }
    if (typeof item === "number" || (allowBooleans && typeof item === "boolean")) {
      flattened.push(Number(item));
      if (flattened.length > MAX_TRACK_VALUES) {
        throw new RangeError(`${label} exceeds the ${MAX_TRACK_VALUES.toLocaleString()} value safety limit.`);
      }
      continue;
    }
    throw new TypeError(`${label} contains a non-numeric value.`);
  }
  return flattened;
}

function isArrayBufferView(value: unknown): value is ArrayBufferView {
  return ArrayBuffer.isView(value);
}

export function toFiniteFloat32Array(value: unknown, label: string): Float32Array {
  let result: Float32Array;
  if (value instanceof Float32Array) {
    result = value;
  } else if (value instanceof ArrayBuffer) {
    if (value.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new RangeError(`${label} has a byte length that is not divisible by four.`);
    }
    result = new Float32Array(value);
  } else if (isArrayBufferView(value)) {
    if (value instanceof DataView) {
      throw new TypeError(`${label} must use a numeric typed array rather than DataView.`);
    }
    if (value.byteLength > MAX_TRACK_VALUES * 8) {
      throw new RangeError(`${label} exceeds the ${MAX_TRACK_VALUES.toLocaleString()} value safety limit.`);
    }
    result = Float32Array.from(value as unknown as ArrayLike<number>, Number);
  } else if (Array.isArray(value)) {
    result = Float32Array.from(flattenNumericArray(value, label, false));
  } else {
    throw new TypeError(`${label} must be a typed array, ArrayBuffer, or numeric array.`);
  }
  if (result.length > MAX_TRACK_VALUES) {
    throw new RangeError(`${label} exceeds the ${MAX_TRACK_VALUES.toLocaleString()} value safety limit.`);
  }
  for (let index = 0; index < result.length; index += 1) {
    if (!Number.isFinite(result[index])) {
      throw new RangeError(`${label} contains a non-finite value at index ${index}.`);
    }
  }
  return result;
}

export function toContactArray(value: unknown, label = "Foot contacts"): Uint8Array {
  let values: ArrayLike<number>;
  if (value instanceof Uint8Array) {
    values = value;
  } else if (value instanceof ArrayBuffer) {
    values = new Uint8Array(value);
  } else if (isArrayBufferView(value)) {
    values = value as unknown as ArrayLike<number>;
  } else if (Array.isArray(value)) {
    values = flattenNumericArray(value, label, true);
  } else {
    throw new TypeError(`${label} must be a typed array, ArrayBuffer, or numeric array.`);
  }
  if (values.length > MAX_TRACK_VALUES) {
    throw new RangeError(`${label} exceeds the ${MAX_TRACK_VALUES.toLocaleString()} value safety limit.`);
  }
  const result = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const valueAtIndex = Number(values[index]);
    if (!Number.isFinite(valueAtIndex)) {
      throw new RangeError(`${label} contains a non-finite value at index ${index}.`);
    }
    result[index] = valueAtIndex > 0.5 ? 1 : 0;
  }
  return result;
}

function requiredString(value: unknown, label: string, fallback?: string): string {
  const candidate = value === undefined ? fallback : value;
  if (typeof candidate !== "string" || candidate.trim().length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`);
  }
  if (candidate.length > 256) throw new RangeError(`${label} must not exceed 256 characters.`);
  return candidate.trim();
}

function integerArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value) && !isArrayBufferView(value)) {
    throw new TypeError(`${label} must be an array of integers.`);
  }
  const numbers = Array.from(value as ArrayLike<unknown>, Number);
  numbers.forEach((number, index) => {
    if (!Number.isInteger(number)) throw new TypeError(`${label}[${index}] must be an integer.`);
  });
  return numbers;
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array of strings.`);
  return value.map((item, index) => requiredString(item, `${label}[${index}]`));
}

/**
 * Validate both browser-manifest snake_case and application camelCase skeletons.
 * The returned object contains only known fields and frozen cloned arrays.
 */
export function normalizeSkeletonMetadata(input: unknown): SkeletonMetadata {
  if (input === undefined || input === null) return CORE27_SKELETON;
  if (!isRecord(input)) throw new TypeError("Skeleton metadata must be an object.");

  const jointNames = stringArray(valueAt(input, "jointNames", "joint_names"), "Skeleton joint names");
  const parents = integerArray(input.parents, "Skeleton parents");
  if (jointNames.length === 0 || jointNames.length > MAX_SKELETON_JOINTS) {
    throw new RangeError(`Skeleton joint count must be between 1 and ${MAX_SKELETON_JOINTS}.`);
  }
  if (parents.length !== jointNames.length) {
    throw new RangeError(
      `Skeleton parents length ${parents.length} does not match ${jointNames.length} joint names.`,
    );
  }
  if (new Set(jointNames).size !== jointNames.length) {
    throw new RangeError("Skeleton joint names must be unique.");
  }

  const rootValue = valueAt(input, "rootJointIndex", "root_index");
  const rootJointIndex = Number(rootValue);
  if (!Number.isInteger(rootJointIndex) || rootJointIndex < 0 || rootJointIndex >= jointNames.length) {
    throw new RangeError("Skeleton root joint index is outside the joint array.");
  }
  const roots = parents.flatMap((parent, joint) => (parent === -1 ? [joint] : []));
  if (roots.length !== 1 || roots[0] !== rootJointIndex) {
    throw new RangeError("Skeleton parents must contain exactly one -1 entry at the declared root joint.");
  }
  parents.forEach((parent, joint) => {
    if (parent < -1 || parent >= jointNames.length || parent === joint) {
      throw new RangeError(`Skeleton parent ${parent} is invalid for joint ${joint}.`);
    }
  });

  const visitState = new Uint8Array(jointNames.length);
  const visit = (joint: number): void => {
    if (visitState[joint] === 1) throw new RangeError("Skeleton parent links contain a cycle.");
    if (visitState[joint] === 2) return;
    visitState[joint] = 1;
    const parent = parents[joint];
    if (parent !== -1) visit(parent);
    visitState[joint] = 2;
  };
  for (let joint = 0; joint < jointNames.length; joint += 1) visit(joint);

  const rawContactIndices = valueAt(
    input,
    "contactJointIndices",
    "contact_joint_indices",
    "footContactJointIndices",
    "foot_contact_joint_indices",
  );
  const hasCore27Layout =
    jointNames.length === CORE27_JOINT_NAMES.length &&
    parents.length === CORE27_PARENTS.length &&
    jointNames.every((name, index) => name === CORE27_JOINT_NAMES[index]) &&
    parents.every((parent, index) => parent === CORE27_PARENTS[index]);
  const contactJointIndices =
    rawContactIndices === undefined
      ? hasCore27Layout
        ? [...CORE27_FOOT_CONTACT_JOINTS]
        : []
      : integerArray(rawContactIndices, "Skeleton contact joint indices");
  if (new Set(contactJointIndices).size !== contactJointIndices.length) {
    throw new RangeError("Skeleton contact joint indices must be unique.");
  }
  contactJointIndices.forEach((joint, channel) => {
    if (joint < 0 || joint >= jointNames.length) {
      throw new RangeError(`Skeleton contact joint index ${joint} at channel ${channel} is invalid.`);
    }
  });

  const rawContactNames = valueAt(input, "contactNames", "contact_names", "footContactNames", "foot_contact_names");
  const contactNames =
    rawContactNames === undefined
      ? contactJointIndices.map((joint) => jointNames[joint])
      : stringArray(rawContactNames, "Skeleton contact names");
  if (contactNames.length !== contactJointIndices.length) {
    throw new RangeError(
      `Skeleton contact names length ${contactNames.length} does not match ${contactJointIndices.length} contact joints.`,
    );
  }

  const id = requiredString(valueAt(input, "id", "name"), "Skeleton id");
  const name = requiredString(input.name, "Skeleton name", id);
  return Object.freeze({
    id,
    name,
    jointNames: Object.freeze([...jointNames]),
    parents: Object.freeze([...parents]),
    rootJointIndex,
    contactJointIndices: Object.freeze([...contactJointIndices]),
    contactNames: Object.freeze([...contactNames]),
  });
}

function positiveNumber(value: unknown, fallback: number, label: string): number {
  const number = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new RangeError(`${label} must be positive.`);
  return number;
}

function statedFrameCount(payload: UnknownRecord, inferred: number): number {
  const raw = valueAt(payload, "frameCount", "frame_count", "numFrames", "num_frames");
  const count = raw === undefined ? inferred : Number(raw);
  if (!Number.isInteger(count) || count < 1 || count > MAX_MOTION_FRAMES) {
    throw new RangeError(`Motion frame count must be an integer between 1 and ${MAX_MOTION_FRAMES}.`);
  }
  if (count !== inferred) {
    throw new RangeError(`Motion frame count ${String(raw)} does not match ${inferred} frames of data.`);
  }
  return count;
}

function optionalFloatTrack(payload: UnknownRecord, label: string, ...keys: string[]): Float32Array | undefined {
  const value = valueAt(payload, ...keys);
  return value === undefined ? undefined : toFiniteFloat32Array(value, label);
}

function normalizeRotationTrack(
  payload: UnknownRecord,
  frameCount: number,
  jointCount: number,
  kind: "local" | "global",
): RotationTrack | undefined {
  const raw =
    kind === "local"
      ? valueAt(payload, "localRotations", "local_rotations", "jointLocalRotations", "joint_local_rotations")
      : valueAt(payload, "globalRotations", "global_rotations", "jointGlobalRotations", "joint_global_rotations");
  if (raw === undefined) return undefined;
  const values = toFiniteFloat32Array(raw, `${kind === "local" ? "Local" : "Global"} rotations`);
  const perJoint = values.length / (frameCount * jointCount);
  if (perJoint !== 4 && perJoint !== 9) {
    throw new RangeError(
      `${kind === "local" ? "Local" : "Global"} rotations must contain T × J × 4 quaternion or T × J × 9 matrix values.`,
    );
  }
  const formatValue = valueAt(
    payload,
    `${kind}RotationFormat`,
    `${kind}_rotation_format`,
    "rotationFormat",
    "rotation_format",
  );
  const expected: RotationFormat = perJoint === 4 ? "quaternion-xyzw" : "matrix3x3-row-major";
  if (formatValue !== undefined && formatValue !== expected) {
    throw new RangeError(
      `${kind === "local" ? "Local" : "Global"} rotation format must be "${expected}" for ${perJoint}-component data.`,
    );
  }
  return {
    values,
    shape: [frameCount, jointCount, perJoint],
    format: expected,
  };
}

/**
 * Normalize Python, worker, imported JSON, and already-canonical motion data.
 */
export function normalizeStructuredMotion(
  input: unknown,
  options: MotionNormalizationOptions = {},
): StructuredMotionResult {
  if (!isRecord(input)) {
    throw new TypeError("The inference worker returned an invalid motion result.");
  }

  let payload = input;
  const nestedMotion = payload.motion;
  if (isRecord(nestedMotion)) payload = nestedMotion;

  const skeleton = normalizeSkeletonMetadata(options.skeleton ?? payload.skeleton);
  const jointCount = skeleton.jointNames.length;
  const rawPositions = valueAt(payload, "posedJoints", "posed_joints", "positions", "joints");
  const positions = toFiniteFloat32Array(rawPositions, "Motion joint positions");
  const componentsPerFrame = jointCount * 3;
  if (positions.length === 0 || positions.length % componentsPerFrame !== 0) {
    throw new RangeError(
      `Motion joint positions must contain T × ${jointCount} × 3 values; received ${positions.length}.`,
    );
  }
  const inferredFrameCount = positions.length / componentsPerFrame;
  const frameCount = statedFrameCount(payload, inferredFrameCount);
  const fps = positiveNumber(payload.fps, options.defaultFps ?? DEFAULT_MOTION_FPS, "Motion FPS");

  const normalizedMotion =
    !isRecord(nestedMotion) && nestedMotion !== undefined
      ? toFiniteFloat32Array(nestedMotion, "Normalized motion")
      : optionalFloatTrack(payload, "Normalized motion", "normalizedMotion", "normalized_motion", "features");
  let normalizedMotionShape: readonly [number, number] | undefined;
  if (normalizedMotion) {
    if (normalizedMotion.length % frameCount !== 0) {
      throw new RangeError("Normalized motion must contain T × D values.");
    }
    normalizedMotionShape = [frameCount, normalizedMotion.length / frameCount];
  }

  const roots = optionalFloatTrack(
    payload,
    "Root track",
    "roots",
    "root",
    "rootTransforms",
    "root_transforms",
    "globalRoot",
    "global_root",
  );
  let rootsShape: readonly [number, number] | undefined;
  if (roots) {
    if (roots.length % frameCount !== 0 || roots.length === 0) {
      throw new RangeError("Root track must contain T × R values.");
    }
    rootsShape = [frameCount, roots.length / frameCount];
  }

  const rawContacts = valueAt(payload, "contacts", "footContacts", "foot_contacts");
  const contacts = rawContacts === undefined ? undefined : toContactArray(rawContacts);
  let contactsShape: readonly [number, number] | undefined;
  if (contacts) {
    const channels = skeleton.contactJointIndices.length;
    if (channels === 0) {
      throw new RangeError("Motion includes contacts but the skeleton has no contact joint metadata.");
    }
    if (contacts.length !== frameCount * channels) {
      throw new RangeError(
        `Foot contacts must contain T × ${channels} values; received ${contacts.length}.`,
      );
    }
    contactsShape = [frameCount, channels];
  }

  return {
    skeleton,
    positions,
    positionsShape: [frameCount, jointCount, 3],
    frameCount,
    fps,
    normalizedMotion,
    normalizedMotionShape,
    localRotations: normalizeRotationTrack(payload, frameCount, jointCount, "local"),
    globalRotations: normalizeRotationTrack(payload, frameCount, jointCount, "global"),
    roots,
    rootsShape,
    contacts,
    contactsShape,
  };
}

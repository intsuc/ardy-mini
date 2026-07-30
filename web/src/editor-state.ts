// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export type Vector3Tuple = readonly [x: number, y: number, z: number];
export type QuaternionTuple = readonly [x: number, y: number, z: number, w: number];

export interface InitialTransform {
  position: Vector3Tuple;
  headingRadians: number;
}

export interface MotionWaypoint {
  id: string;
  frame: number;
  position: Vector3Tuple;
  headingRadians?: number;
  label?: string;
  enabled: boolean;
}

export type MotionConstraintKind =
  | "position"
  | "orientation"
  | "transform"
  | "trajectory"
  | "root"
  | "end-effector";

/**
 * A visual constraint track. startFrame/endFrame make the same object suitable
 * for a timeline track and for frame-filtered markers in the 3D viewer.
 */
export interface MotionConstraintMarker {
  id: string;
  kind: MotionConstraintKind;
  startFrame: number;
  endFrame: number;
  jointIndex?: number;
  position?: Vector3Tuple;
  orientation?: QuaternionTuple;
  label?: string;
  color?: string;
  enabled: boolean;
}

export interface ViewerOutputVisibility {
  skeleton: boolean;
  /** Joint-driven capsule/sphere body proxy; this is not an SMPL mesh. */
  mesh: boolean;
  /** Synchronously rendered comparison motion. */
  reference: boolean;
  trajectory: boolean;
  contacts: boolean;
  orientationAxes: boolean;
  constraints: boolean;
  initialTransform: boolean;
  waypoints: boolean;
}

export interface MotionEditorState {
  initialTransform: InitialTransform;
  waypoints: readonly MotionWaypoint[];
  constraints: readonly MotionConstraintMarker[];
  outputVisibility: ViewerOutputVisibility;
}

export interface EditorValidationOptions {
  frameCount?: number;
  jointCount?: number;
}

export const DEFAULT_INITIAL_TRANSFORM: InitialTransform = Object.freeze({
  position: Object.freeze([0, 0, 0] as const),
  headingRadians: 0,
});

export const DEFAULT_OUTPUT_VISIBILITY: ViewerOutputVisibility = Object.freeze({
  skeleton: true,
  mesh: false,
  reference: false,
  trajectory: true,
  contacts: true,
  orientationAxes: true,
  constraints: true,
  initialTransform: true,
  waypoints: true,
});

export const DEFAULT_EDITOR_STATE: MotionEditorState = Object.freeze({
  initialTransform: DEFAULT_INITIAL_TRANSFORM,
  waypoints: Object.freeze([]),
  constraints: Object.freeze([]),
  outputVisibility: DEFAULT_OUTPUT_VISIBILITY,
});

type UnknownRecord = Record<string, unknown>;

const MAX_EDITOR_ITEMS = 10_000;
const ALLOWED_CONSTRAINT_KINDS = new Set<MotionConstraintKind>([
  "position",
  "orientation",
  "transform",
  "trajectory",
  "root",
  "end-effector",
]);

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number.`);
  }
  return value;
}

function frameNumber(value: unknown, label: string, frameCount?: number): number {
  const frame = finiteNumber(value, label);
  if (!Number.isInteger(frame) || frame < 0) throw new RangeError(`${label} must be a non-negative integer.`);
  if (frameCount !== undefined && frame >= frameCount) {
    throw new RangeError(`${label} ${frame} is outside the ${frameCount}-frame motion.`);
  }
  return frame;
}

function identifier(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0 || value.length > 128) {
    throw new TypeError(`${label} must be a non-empty string of at most 128 characters.`);
  }
  return value.trim();
}

function optionalLabel(value: unknown, label: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.length > 512) {
    throw new TypeError(`${label} must be a string of at most 512 characters.`);
  }
  return value;
}

function booleanValue(value: unknown, label: string, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean.`);
  return value;
}

function vector3(value: unknown, label: string): Vector3Tuple {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new TypeError(`${label} must contain three numbers.`);
  }
  const components = Array.from(value as ArrayLike<unknown>);
  if (components.length !== 3) throw new RangeError(`${label} must contain exactly three numbers.`);
  return Object.freeze([
    finiteNumber(components[0], `${label}[0]`),
    finiteNumber(components[1], `${label}[1]`),
    finiteNumber(components[2], `${label}[2]`),
  ]);
}

function quaternion(value: unknown, label: string): QuaternionTuple {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new TypeError(`${label} must contain four numbers.`);
  }
  const components = Array.from(value as ArrayLike<unknown>);
  if (components.length !== 4) throw new RangeError(`${label} must contain exactly four numbers.`);
  const x = finiteNumber(components[0], `${label}[0]`);
  const y = finiteNumber(components[1], `${label}[1]`);
  const z = finiteNumber(components[2], `${label}[2]`);
  const w = finiteNumber(components[3], `${label}[3]`);
  const magnitude = Math.hypot(x, y, z, w);
  if (magnitude < 1e-8) throw new RangeError(`${label} must not be a zero quaternion.`);
  return Object.freeze([x / magnitude, y / magnitude, z / magnitude, w / magnitude]);
}

function optionalColor(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || !/^#[0-9a-f]{3}(?:[0-9a-f]{3})?$/i.test(value)) {
    throw new TypeError("Constraint color must be a #RGB or #RRGGBB hexadecimal color.");
  }
  return value.toLowerCase();
}

export function normalizeInitialTransform(value: unknown): InitialTransform {
  if (!isRecord(value)) throw new TypeError("Initial transform must be an object.");
  return Object.freeze({
    position: vector3(value.position, "Initial transform position"),
    headingRadians: finiteNumber(value.headingRadians ?? value.heading_radians ?? 0, "Initial transform heading"),
  });
}

export function normalizeWaypoint(value: unknown, options: EditorValidationOptions = {}): MotionWaypoint {
  if (!isRecord(value)) throw new TypeError("Waypoint must be an object.");
  const headingValue = value.headingRadians ?? value.heading_radians;
  return Object.freeze({
    id: identifier(value.id, "Waypoint id"),
    frame: frameNumber(value.frame, "Waypoint frame", options.frameCount),
    position: vector3(value.position, "Waypoint position"),
    headingRadians:
      headingValue === undefined ? undefined : finiteNumber(headingValue, "Waypoint heading"),
    label: optionalLabel(value.label, "Waypoint label"),
    enabled: booleanValue(value.enabled, "Waypoint enabled state", true),
  });
}

export function normalizeConstraintMarker(
  value: unknown,
  options: EditorValidationOptions = {},
): MotionConstraintMarker {
  if (!isRecord(value)) throw new TypeError("Constraint marker must be an object.");
  if (typeof value.kind !== "string" || !ALLOWED_CONSTRAINT_KINDS.has(value.kind as MotionConstraintKind)) {
    throw new TypeError(`Unsupported constraint kind "${String(value.kind)}".`);
  }
  const startFrame = frameNumber(value.startFrame ?? value.start_frame, "Constraint start frame", options.frameCount);
  const endFrame = frameNumber(
    value.endFrame ?? value.end_frame ?? startFrame,
    "Constraint end frame",
    options.frameCount,
  );
  if (endFrame < startFrame) throw new RangeError("Constraint end frame must not precede its start frame.");

  let jointIndex: number | undefined;
  const rawJoint = value.jointIndex ?? value.joint_index;
  if (rawJoint !== undefined) {
    jointIndex = finiteNumber(rawJoint, "Constraint joint index");
    if (!Number.isInteger(jointIndex) || jointIndex < 0) {
      throw new RangeError("Constraint joint index must be a non-negative integer.");
    }
    if (options.jointCount !== undefined && jointIndex >= options.jointCount) {
      throw new RangeError(`Constraint joint index ${jointIndex} is outside the skeleton.`);
    }
  }

  const position = value.position === undefined ? undefined : vector3(value.position, "Constraint position");
  const orientation =
    value.orientation === undefined ? undefined : quaternion(value.orientation, "Constraint orientation");
  if (position === undefined && orientation === undefined && value.kind !== "trajectory") {
    throw new RangeError("A non-trajectory constraint needs a position, orientation, or both.");
  }
  return Object.freeze({
    id: identifier(value.id, "Constraint id"),
    kind: value.kind as MotionConstraintKind,
    startFrame,
    endFrame,
    jointIndex,
    position,
    orientation,
    label: optionalLabel(value.label, "Constraint label"),
    color: optionalColor(value.color),
    enabled: booleanValue(value.enabled, "Constraint enabled state", true),
  });
}

export function normalizeOutputVisibility(value: unknown): ViewerOutputVisibility {
  if (!isRecord(value)) throw new TypeError("Output visibility must be an object.");
  const aliases: Partial<Record<keyof ViewerOutputVisibility, string>> = {
    orientationAxes: "orientation_axes",
    initialTransform: "initial_transform",
  };
  const get = (key: keyof ViewerOutputVisibility): boolean =>
    booleanValue(
      value[key] ?? (aliases[key] ? value[aliases[key]!] : undefined),
      `Output visibility "${key}"`,
      DEFAULT_OUTPUT_VISIBILITY[key],
    );
  return Object.freeze({
    skeleton: get("skeleton"),
    mesh: get("mesh"),
    reference: get("reference"),
    trajectory: get("trajectory"),
    contacts: get("contacts"),
    orientationAxes: get("orientationAxes"),
    constraints: get("constraints"),
    initialTransform: get("initialTransform"),
    waypoints: get("waypoints"),
  });
}

export function normalizeEditorState(
  value: unknown,
  options: EditorValidationOptions = {},
): MotionEditorState {
  if (value === undefined || value === null) return DEFAULT_EDITOR_STATE;
  if (!isRecord(value)) throw new TypeError("Motion editor state must be an object.");
  const rawWaypoints = value.waypoints ?? [];
  const rawConstraints = value.constraints ?? [];
  if (!Array.isArray(rawWaypoints) || !Array.isArray(rawConstraints)) {
    throw new TypeError("Editor waypoints and constraints must be arrays.");
  }
  if (rawWaypoints.length > MAX_EDITOR_ITEMS || rawConstraints.length > MAX_EDITOR_ITEMS) {
    throw new RangeError(`Editor state must not contain more than ${MAX_EDITOR_ITEMS} items per track.`);
  }
  // Timeline editing intentionally allows constraints beyond the current clip.
  // A future auto-replan target must remain valid before those frames exist.
  const waypoints = rawWaypoints.map((item) => normalizeWaypoint(item));
  const constraints = rawConstraints.map((item) =>
    normalizeConstraintMarker(item, { jointCount: options.jointCount }),
  );
  if (new Set(waypoints.map((item) => item.id)).size !== waypoints.length) {
    throw new RangeError("Waypoint ids must be unique.");
  }
  if (new Set(constraints.map((item) => item.id)).size !== constraints.length) {
    throw new RangeError("Constraint ids must be unique.");
  }
  return Object.freeze({
    initialTransform:
      value.initialTransform === undefined && value.initial_transform === undefined
        ? DEFAULT_INITIAL_TRANSFORM
        : normalizeInitialTransform(value.initialTransform ?? value.initial_transform),
    waypoints: Object.freeze(waypoints),
    constraints: Object.freeze(constraints),
    outputVisibility:
      value.outputVisibility === undefined && value.output_visibility === undefined
        ? DEFAULT_OUTPUT_VISIBILITY
        : normalizeOutputVisibility(value.outputVisibility ?? value.output_visibility),
  });
}

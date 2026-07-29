// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type {
  BrowserModelPackManifest,
  BrowserMotionLayout,
  BrowserStatsPayload,
} from "./manifest";

export type MotionConstraintKind =
  | "root"
  | "full-body"
  | "left-hand"
  | "right-hand"
  | "left-foot"
  | "right-foot";

/**
 * One sparse constraint in absolute session-frame coordinates.
 *
 * `values` are normalized ARDY motion features. An interval repeats the same
 * observation from `frame` through `endFrame`, inclusive.
 */
export interface MotionConstraint {
  id: string;
  kind: MotionConstraintKind;
  frame: number;
  endFrame?: number;
  values: Float32Array;
  mask: Float32Array;
}

export interface ConstraintWindowBuffers {
  observedMotion: Float32Array;
  motionMask: Float32Array;
  futureFrames: number;
  appliedConstraintIds: string[];
}

export interface RootWaypoint {
  id: string;
  frame: number;
  x: number;
  z: number;
  heading?: number;
}

export interface TargetVelocityOptions {
  startFrame: number;
  startX: number;
  startZ: number;
  startVelocityX?: number;
  startVelocityZ?: number;
  velocityX: number;
  velocityZ: number;
  fps: number;
  durationSeconds?: number;
  transitionSeconds?: number;
  intervalFrames?: number;
  includeHeading?: boolean;
}

const ROOT_PATH_SMOOTHING_MARGIN = 0.06;
const ROOT_PATH_SMOOTHING_ITERATIONS = 32;

function requiredLayout(
  manifest: BrowserModelPackManifest,
): BrowserMotionLayout {
  if (manifest.motion_layout === undefined) {
    throw new Error("Model pack does not describe its motion feature layout");
  }
  return manifest.motion_layout;
}

function requiredMotionStats(
  manifest: BrowserModelPackManifest,
): BrowserStatsPayload {
  if (manifest.stats?.motion === undefined) {
    throw new Error("Model pack does not contain motion normalization statistics");
  }
  return manifest.stats.motion;
}

function frameRange(constraint: MotionConstraint): [number, number] {
  const end = constraint.endFrame ?? constraint.frame;
  if (
    !Number.isSafeInteger(constraint.frame) ||
    !Number.isSafeInteger(end) ||
    constraint.frame < 0 ||
    end < constraint.frame
  ) {
    throw new RangeError(`Constraint ${constraint.id} has invalid frame bounds`);
  }
  return [constraint.frame, end];
}

function validateConstraint(
  constraint: MotionConstraint,
  motionDim: number,
): void {
  if (constraint.id.trim().length === 0) {
    throw new TypeError("Constraint id must not be empty");
  }
  frameRange(constraint);
  if (
    constraint.values.length !== motionDim ||
    constraint.mask.length !== motionDim
  ) {
    throw new RangeError(
      `Constraint ${constraint.id} must contain ${motionDim} motion features`,
    );
  }
  let hasObservedFeature = false;
  for (let index = 0; index < motionDim; index += 1) {
    if (
      !Number.isFinite(constraint.values[index]) ||
      !Number.isFinite(constraint.mask[index])
    ) {
      throw new TypeError(`Constraint ${constraint.id} contains non-finite data`);
    }
    if (constraint.mask[index] < 0 || constraint.mask[index] > 1) {
      throw new RangeError(
        `Constraint ${constraint.id} mask values must be between 0 and 1`,
      );
    }
    hasObservedFeature ||= constraint.mask[index] > 0.5;
  }
  if (!hasObservedFeature) {
    throw new RangeError(
      `Constraint ${constraint.id} must observe at least one motion feature`,
    );
  }
}

function normalizedValue(
  raw: number,
  feature: number,
  stats: BrowserStatsPayload,
): number {
  return Math.fround(
    (raw - stats.mean[feature]) /
      stats.normalization_denominator[feature],
  );
}

function rawValue(
  normalized: number,
  feature: number,
  stats: BrowserStatsPayload,
): number {
  return (
    normalized * stats.normalization_denominator[feature] +
    stats.mean[feature]
  );
}

export function createRootConstraint(
  manifest: BrowserModelPackManifest,
  waypoint: RootWaypoint,
): MotionConstraint {
  const layout = requiredLayout(manifest);
  const stats = requiredMotionStats(manifest);
  const root = layout.root_pos;
  const heading = layout.global_root_heading;
  if (root === undefined || heading === undefined) {
    throw new Error("Model pack does not expose root position and heading slices");
  }
  const values = new Float32Array(manifest.dimensions.motion_dim);
  const mask = new Float32Array(manifest.dimensions.motion_dim);
  values[root[0]] = normalizedValue(waypoint.x, root[0], stats);
  values[root[0] + 2] = normalizedValue(waypoint.z, root[0] + 2, stats);
  mask[root[0]] = 1;
  mask[root[0] + 2] = 1;
  if (waypoint.heading !== undefined) {
    values[heading[0]] = normalizedValue(
      Math.cos(waypoint.heading),
      heading[0],
      stats,
    );
    values[heading[0] + 1] = normalizedValue(
      Math.sin(waypoint.heading),
      heading[0] + 1,
      stats,
    );
    mask[heading[0]] = 1;
    mask[heading[0] + 1] = 1;
  }
  return {
    id: waypoint.id,
    kind: "root",
    frame: waypoint.frame,
    values,
    mask,
  };
}

function normalizedMotionFrame(
  motion: Float32Array,
  frame: number,
  motionDim: number,
): Float32Array {
  if (
    !Number.isSafeInteger(frame) ||
    frame < 0 ||
    (frame + 1) * motionDim > motion.length
  ) {
    throw new RangeError("Source motion frame is outside the clip");
  }
  return motion.slice(frame * motionDim, (frame + 1) * motionDim);
}

function findJointGroup(
  manifest: BrowserModelPackManifest,
  kind: Exclude<MotionConstraintKind, "root" | "full-body">,
): { positionJoints: number[]; rotationJoints: number[] } {
  const names = manifest.skeleton?.joint_names;
  if (names === undefined) {
    throw new Error("Model pack does not describe skeleton joint names");
  }
  const side = kind.startsWith("left") ? "left" : "right";
  const part = kind.endsWith("hand") ? "hand" : "foot";
  const normalizedNames = names.map((name) =>
    name.toLowerCase().replaceAll(/[^a-z0-9]/g, ""),
  );
  const baseName = `${side}${part}`;
  const base = normalizedNames.findIndex((name) => name === baseName);
  if (base < 0) {
    throw new Error(`Skeleton does not contain a ${kind} joint chain`);
  }
  const preferredTip =
    part === "hand"
      ? `${side}handend`
      : `${side}toebase`;
  let tip = normalizedNames.findIndex((name) => name === preferredTip);
  if (tip < 0) {
    const parents = manifest.skeleton?.parents;
    tip =
      parents?.findIndex(
        (parent, index) =>
          parent === base &&
          index !== base &&
          normalizedNames[index].startsWith(side),
      ) ?? -1;
  }
  return {
    positionJoints: tip < 0 ? [base] : [base, tip],
    rotationJoints: [base],
  };
}

function markSlice(
  mask: Float32Array,
  slice: [number, number] | undefined,
): void {
  if (slice !== undefined) {
    mask.fill(1, slice[0], slice[1]);
  }
}

/**
 * Capture a Viser-compatible keyframe from an already generated motion frame.
 * Full-body constraints use joint positions plus root pose; end effectors use
 * the selected chain positions, base rotation, and root pose.
 */
export function createCapturedConstraint(
  manifest: BrowserModelPackManifest,
  id: string,
  kind: Exclude<MotionConstraintKind, "root">,
  targetFrame: number,
  sourceMotion: Float32Array,
  sourceFrame: number,
  endFrame?: number,
): MotionConstraint {
  const layout = requiredLayout(manifest);
  const values = normalizedMotionFrame(
    sourceMotion,
    sourceFrame,
    manifest.dimensions.motion_dim,
  );
  const mask = new Float32Array(manifest.dimensions.motion_dim);
  markSlice(mask, layout.root_pos);
  markSlice(mask, layout.global_root_heading);
  if (kind === "full-body") {
    markSlice(mask, layout.local_joints_positions);
  } else {
    const positions = layout.local_joints_positions;
    const rotations = layout.global_rot_data;
    if (positions === undefined || rotations === undefined) {
      throw new Error("Model pack does not expose joint position/rotation slices");
    }
    const group = findJointGroup(manifest, kind);
    const rootIndex = manifest.skeleton?.root_index;
    if (
      rootIndex !== undefined &&
      !group.rotationJoints.includes(rootIndex)
    ) {
      group.rotationJoints.push(rootIndex);
    }
    for (const joint of group.positionJoints) {
      if (joint === manifest.skeleton?.root_index) {
        continue;
      }
      const start = positions[0] + (joint - 1) * 3;
      mask.fill(1, start, start + 3);
    }
    for (const joint of group.rotationJoints) {
      const start = rotations[0] + joint * 6;
      mask.fill(1, start, start + 6);
    }
    // Viser's end-effector sets include the Hips joint in the rotation
    // constraints in addition to the selected hand/foot chain.
    const rootJoint = manifest.skeleton?.root_index;
    if (rootJoint !== undefined) {
      const start = rotations[0] + rootJoint * 6;
      mask.fill(1, start, start + 6);
    }
  }
  for (let index = 0; index < values.length; index += 1) {
    if (mask[index] === 0) {
      values[index] = 0;
    }
  }
  return { id, kind, frame: targetFrame, endFrame, values, mask };
}

export function interpolateRootWaypoints(
  manifest: BrowserModelPackManifest,
  waypoints: readonly RootWaypoint[],
  dense: boolean,
): MotionConstraint[] {
  const sorted = [...waypoints].sort((left, right) => left.frame - right.frame);
  if (!dense || sorted.length < 2) {
    return sorted.map((waypoint) => createRootConstraint(manifest, waypoint));
  }
  const samples: RootWaypoint[] = [];
  for (let segment = 0; segment < sorted.length - 1; segment += 1) {
    const start = sorted[segment];
    const end = sorted[segment + 1];
    if (end.frame <= start.frame) {
      throw new RangeError("Dense waypoints must use unique increasing frames");
    }
    const finalFrame = segment === sorted.length - 2 ? end.frame : end.frame - 1;
    for (let frame = start.frame; frame <= finalFrame; frame += 1) {
      const alpha = (frame - start.frame) / (end.frame - start.frame);
      const heading =
        start.heading === undefined || end.heading === undefined
          ? undefined
          : start.heading +
            Math.atan2(
              Math.sin(end.heading - start.heading),
              Math.cos(end.heading - start.heading),
            ) *
              alpha;
      samples.push({
        id: `${start.id}:${end.id}:${frame}`,
        frame,
        x: start.x + (end.x - start.x) * alpha,
        z: start.z + (end.z - start.z) * alpha,
        heading,
      });
    }
  }

  // Match Viser's smooth dense-root path semantics without pulling a numeric
  // optimization package into the browser. Repeated Laplacian relaxation
  // removes sharp corners while projecting every sample back inside the same
  // 6 cm deviation envelope used by the native smoother.
  if (sorted.length >= 3 && samples.length >= 3) {
    const original = samples.map(({ x, z }) => [x, z] as const);
    let smoothed = original.map(([x, z]) => [x, z] as [number, number]);
    for (
      let iteration = 0;
      iteration < ROOT_PATH_SMOOTHING_ITERATIONS;
      iteration += 1
    ) {
      const next = smoothed.map(([x, z]) => [x, z] as [number, number]);
      for (let index = 1; index < smoothed.length - 1; index += 1) {
        const desiredX =
          (smoothed[index - 1][0] +
            2 * smoothed[index][0] +
            smoothed[index + 1][0]) /
          4;
        const desiredZ =
          (smoothed[index - 1][1] +
            2 * smoothed[index][1] +
            smoothed[index + 1][1]) /
          4;
        const deltaX = desiredX - original[index][0];
        const deltaZ = desiredZ - original[index][1];
        const distance = Math.hypot(deltaX, deltaZ);
        const scale =
          distance > ROOT_PATH_SMOOTHING_MARGIN
            ? ROOT_PATH_SMOOTHING_MARGIN / distance
            : 1;
        next[index] = [
          original[index][0] + deltaX * scale,
          original[index][1] + deltaZ * scale,
        ];
      }
      smoothed = next;
    }
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = {
        ...samples[index],
        x: smoothed[index][0],
        z: smoothed[index][1],
      };
    }
  }

  return samples.map((waypoint) => createRootConstraint(manifest, waypoint));
}

export function waypointsFromTargetVelocity(
  options: TargetVelocityOptions,
): RootWaypoint[] {
  const duration = options.durationSeconds ?? 2;
  const transition = options.transitionSeconds ?? 2;
  const interval = options.intervalFrames ?? 10;
  const totalFrames = Math.max(1, Math.round(duration * options.fps));
  const transitionFrames = Math.max(
    0,
    Math.min(totalFrames, Math.round(transition * options.fps)),
  );
  const startVelocityX = options.startVelocityX ?? options.velocityX;
  const startVelocityZ = options.startVelocityZ ?? options.velocityZ;
  const speed = Math.hypot(options.velocityX, options.velocityZ);
  const heading =
    options.includeHeading && speed > 1e-6
      ? Math.atan2(options.velocityX, options.velocityZ)
      : undefined;
  const result: RootWaypoint[] = [];
  let x = options.startX;
  let z = options.startZ;
  for (let offset = 1; offset <= totalFrames; offset += 1) {
    const alpha =
      transitionFrames === 0 ? 1 : Math.min(1, offset / transitionFrames);
    const velocityX =
      startVelocityX + (options.velocityX - startVelocityX) * alpha;
    const velocityZ =
      startVelocityZ + (options.velocityZ - startVelocityZ) * alpha;
    x += velocityX / options.fps;
    z += velocityZ / options.fps;
    if (offset % interval === 0) {
      result.push({
        id: `velocity:${options.startFrame}:${offset}`,
        frame: options.startFrame + offset,
        x: Math.fround(x),
        z: Math.fround(z),
        heading,
      });
    }
  }
  return result;
}

/**
 * Project absolute sparse constraints into one fixed ONNX window and translate
 * observed root positions into the same recentered coordinates as its history.
 */
export function buildConstraintWindowBuffers(
  manifest: BrowserModelPackManifest,
  constraints: readonly MotionConstraint[],
  windowStartFrame: number,
  historyFrames: number,
  requestedFutureFrames: number,
  globalTranslation: Float32Array,
): ConstraintWindowBuffers {
  const { dimensions } = manifest;
  const maxFrames =
    dimensions.constraint_max_frames ?? dimensions.max_frames;
  const motionDim = dimensions.motion_dim;
  const generationFrames = dimensions.generation_frames;
  const maxFutureFrames = Math.max(
    0,
    maxFrames - historyFrames - generationFrames,
  );
  const futureLimit = Math.min(
    maxFutureFrames,
    Math.max(0, requestedFutureFrames),
  );
  const windowGenerationEnd =
    windowStartFrame + historyFrames + generationFrames;
  let furthestConstraint = windowGenerationEnd - 1;
  for (const constraint of constraints) {
    validateConstraint(constraint, motionDim);
    const [, end] = frameRange(constraint);
    if (
      end >= windowGenerationEnd &&
      constraint.frame < windowGenerationEnd + futureLimit
    ) {
      furthestConstraint = Math.max(
        furthestConstraint,
        Math.min(end, windowGenerationEnd + futureLimit - 1),
      );
    }
  }
  const framesPerToken = dimensions.num_frames_per_token;
  const rawFutureFrames = Math.max(
    0,
    furthestConstraint - windowGenerationEnd + 1,
  );
  const futureFrames = Math.min(
    futureLimit,
    Math.ceil(rawFutureFrames / framesPerToken) * framesPerToken,
  );

  const observedMotion = new Float32Array(maxFrames * motionDim);
  const motionMask = new Float32Array(maxFrames * motionDim);
  const appliedConstraintIds = new Set<string>();
  const rootSlice = requiredLayout(manifest).root_pos;
  const stats = requiredMotionStats(manifest);
  const visibleEnd =
    windowStartFrame + historyFrames + generationFrames + futureFrames;
  for (const constraint of constraints) {
    const [start, end] = frameRange(constraint);
    const first = Math.max(start, windowStartFrame + historyFrames);
    const last = Math.min(end, visibleEnd - 1);
    if (last < first) {
      continue;
    }
    for (let absoluteFrame = first; absoluteFrame <= last; absoluteFrame += 1) {
      const relativeFrame = absoluteFrame - windowStartFrame;
      const outputOffset = relativeFrame * motionDim;
      for (let feature = 0; feature < motionDim; feature += 1) {
        if (constraint.mask[feature] <= 0.5) {
          continue;
        }
        let value = constraint.values[feature];
        if (
          rootSlice !== undefined &&
          feature >= rootSlice[0] &&
          feature < rootSlice[1]
        ) {
          const axis = feature - rootSlice[0];
          const world = rawValue(value, feature, stats);
          value = normalizedValue(
            world - globalTranslation[axis],
            feature,
            stats,
          );
        }
        observedMotion[outputOffset + feature] = Math.fround(value);
        motionMask[outputOffset + feature] = 1;
      }
    }
    appliedConstraintIds.add(constraint.id);
  }
  return {
    observedMotion,
    motionMask,
    futureFrames,
    appliedConstraintIds: [...appliedConstraintIds],
  };
}

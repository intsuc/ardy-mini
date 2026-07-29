// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type { MotionConstraint } from "./constraints";
import type { BrowserModelPackManifest } from "./manifest";

const DEFAULT_CONTACT_THRESHOLD = 0.5;
const DEFAULT_ROOT_MARGIN = 0.04;
const DEFAULT_CONSTRAINT_BLEND_FRAMES = 8;
const DEFAULT_FOOT_LOCK_STRENGTH = 0.8;
const DEFAULT_FOOT_LOCK_RAMP_FRAMES = 2;
const DEFAULT_MAX_FOOT_CORRECTION = 0.25;
const DEFAULT_MAX_ROOT_CORRECTION = 1_000;
const DEFAULT_MAX_POSITION_MAGNITUDE = 1_000_000;
const MAX_FINITE_FLOAT32_MAGNITUDE = 3e38;
const MAX_FRAMES = 1_000_000;
const MAX_JOINTS = 1_024;
const MAX_ROOT_COMPONENTS = 256;

export interface RootPositionTrack {
  /** Flat `frameCount * components` root track. */
  values: Float32Array;
  components: number;
  /** Indices of x, y, and z in one root record. Defaults to `[0, 1, 2]`. */
  positionIndices?: readonly [x: number, y: number, z: number];
}

export interface FootContactTrack {
  /** Flat `frameCount * channels` contact probabilities or binary labels. */
  values: ArrayLike<number>;
  channels: number;
  /** Skeleton joint driven by each contact channel. */
  jointIndices: readonly number[];
}

export interface MotionPostprocessInput {
  /** Flat global joint positions with shape `[frameCount, jointCount, 3]`. */
  positions: Float32Array;
  frameCount: number;
  jointCount: number;
  rootJointIndex?: number;
  roots?: RootPositionTrack;
  contacts?: FootContactTrack;
  /** Constraints use absolute session-frame coordinates. */
  constraints?: readonly MotionConstraint[];
  /** Required when `constraints` is non-empty. */
  constraintManifest?: BrowserModelPackManifest;
  /** Absolute session frame represented by local frame zero. */
  frameOffset?: number;
}

export interface MotionPostprocessOptions {
  contactThreshold?: number;
  /**
   * Maximum permitted horizontal deviation from root and end-effector root
   * targets. Full-body targets remain exact.
   */
  rootMargin?: number;
  /** Smooth correction into and out of the first/last root keyframe. */
  constraintBlendFrames?: number;
  footLockStrength?: number;
  /** Taper a contact lock at contact-run boundaries. */
  footLockRampFrames?: number;
  /** Maximum horizontal whole-body shift contributed by foot locking per frame. */
  maxFootCorrectionPerFrame?: number;
  /** Safety cap for one root-keyframe correction. */
  maxRootCorrection?: number;
  /** Reject coordinates outside this magnitude and cap corrected coordinates. */
  maxPositionMagnitude?: number;
}

export interface MotionPostprocessMetrics {
  rootConstraintFrames: number;
  rootConstraintMeanErrorBefore: number;
  rootConstraintMeanErrorAfter: number;
  contactSamples: number;
  footSlidingDistanceBefore: number;
  footSlidingDistanceAfter: number;
  maxAppliedTranslation: number;
}

export interface MotionPostprocessResult {
  /** Corrected copy; the input is never mutated. */
  positions: Float32Array;
  /** Corrected copy when an explicit root track was supplied. */
  roots?: Float32Array;
  /** Per-frame xyz translation applied to the decoded pose. */
  rootTranslations: Float32Array;
  metrics: MotionPostprocessMetrics;
}

interface ResolvedOptions {
  contactThreshold: number;
  rootMargin: number;
  constraintBlendFrames: number;
  footLockStrength: number;
  footLockRampFrames: number;
  maxFootCorrectionPerFrame: number;
  maxRootCorrection: number;
  maxPositionMagnitude: number;
}

interface RootAccess {
  values?: Float32Array;
  components: number;
  indices: readonly [number, number, number];
  rootJointIndex: number;
}

interface ConstraintTargets {
  values: Float64Array;
  present: Uint8Array;
  exact: Uint8Array;
  constrainedFrames: Uint8Array;
}

function finiteNumber(
  value: number | undefined,
  fallback: number,
  label: string,
  min: number,
  max: number,
): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < min || resolved > max) {
    throw new RangeError(`${label} must be a finite number in [${min}, ${max}]`);
  }
  return resolved;
}

function safeInteger(
  value: number | undefined,
  fallback: number,
  label: string,
  min: number,
  max: number,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < min || resolved > max) {
    throw new RangeError(`${label} must be an integer in [${min}, ${max}]`);
  }
  return resolved;
}

function resolveOptions(options: MotionPostprocessOptions): ResolvedOptions {
  return {
    contactThreshold: finiteNumber(
      options.contactThreshold,
      DEFAULT_CONTACT_THRESHOLD,
      "Contact threshold",
      0,
      1,
    ),
    rootMargin: finiteNumber(
      options.rootMargin,
      DEFAULT_ROOT_MARGIN,
      "Root margin",
      0,
      DEFAULT_MAX_POSITION_MAGNITUDE,
    ),
    constraintBlendFrames: safeInteger(
      options.constraintBlendFrames,
      DEFAULT_CONSTRAINT_BLEND_FRAMES,
      "Constraint blend frames",
      0,
      MAX_FRAMES,
    ),
    footLockStrength: finiteNumber(
      options.footLockStrength,
      DEFAULT_FOOT_LOCK_STRENGTH,
      "Foot-lock strength",
      0,
      1,
    ),
    footLockRampFrames: safeInteger(
      options.footLockRampFrames,
      DEFAULT_FOOT_LOCK_RAMP_FRAMES,
      "Foot-lock ramp frames",
      1,
      MAX_FRAMES,
    ),
    maxFootCorrectionPerFrame: finiteNumber(
      options.maxFootCorrectionPerFrame,
      DEFAULT_MAX_FOOT_CORRECTION,
      "Maximum foot correction",
      0,
      DEFAULT_MAX_POSITION_MAGNITUDE,
    ),
    maxRootCorrection: finiteNumber(
      options.maxRootCorrection,
      DEFAULT_MAX_ROOT_CORRECTION,
      "Maximum root correction",
      0,
      DEFAULT_MAX_POSITION_MAGNITUDE,
    ),
    maxPositionMagnitude: finiteNumber(
      options.maxPositionMagnitude,
      DEFAULT_MAX_POSITION_MAGNITUDE,
      "Maximum position magnitude",
      1,
      MAX_FINITE_FLOAT32_MAGNITUDE,
    ),
  };
}

function validateFiniteTrack(
  values: ArrayLike<number>,
  expectedLength: number,
  label: string,
  maximumMagnitude: number,
): void {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `${label} must contain ${expectedLength} values; received ${values.length}`,
    );
  }
  for (let index = 0; index < values.length; index += 1) {
    const value = Number(values[index]);
    if (!Number.isFinite(value)) {
      throw new RangeError(`${label} contains a non-finite value at index ${index}`);
    }
    if (Math.abs(value) > maximumMagnitude) {
      throw new RangeError(
        `${label} exceeds the configured coordinate bound at index ${index}`,
      );
    }
  }
}

function resolveRootAccess(
  input: MotionPostprocessInput,
  maximumMagnitude: number,
): RootAccess {
  const rootJointIndex = safeInteger(
    input.rootJointIndex,
    0,
    "Root joint index",
    0,
    input.jointCount - 1,
  );
  if (input.roots === undefined) {
    return {
      values: undefined,
      components: 3,
      indices: [0, 1, 2],
      rootJointIndex,
    };
  }
  const components = safeInteger(
    input.roots.components,
    3,
    "Root-track component count",
    3,
    MAX_ROOT_COMPONENTS,
  );
  const indices = input.roots.positionIndices ?? [0, 1, 2];
  const seen = new Set<number>();
  for (const [axis, index] of ["x", "y", "z"].map(
    (axis, index) => [axis, indices[index]] as const,
  )) {
    if (
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= components ||
      seen.has(index)
    ) {
      throw new RangeError(`Root ${axis} index is invalid for ${components} components`);
    }
    seen.add(index);
  }
  validateFiniteTrack(
    input.roots.values,
    input.frameCount * components,
    "Root track",
    maximumMagnitude,
  );
  return {
    values: input.roots.values.slice(),
    components,
    indices,
    rootJointIndex,
  };
}

function positionOffset(
  frame: number,
  joint: number,
  jointCount: number,
): number {
  return (frame * jointCount + joint) * 3;
}

function rootValue(
  positions: Float32Array,
  access: RootAccess,
  frame: number,
  axis: 0 | 1 | 2,
  jointCount: number,
): number {
  if (access.values !== undefined) {
    return access.values[frame * access.components + access.indices[axis]];
  }
  return positions[
    positionOffset(frame, access.rootJointIndex, jointCount) + axis
  ];
}

function clampMagnitude(x: number, z: number, maximum: number): [number, number] {
  const magnitude = Math.hypot(x, z);
  if (magnitude === 0 || magnitude <= maximum) return [x, z];
  const scale = maximum / magnitude;
  return [x * scale, z * scale];
}

function boundedFloat32(value: number, maximumMagnitude: number): number {
  if (!Number.isFinite(value)) {
    throw new RangeError("Motion correction produced a non-finite coordinate");
  }
  const rounded = Math.fround(
    Math.max(-maximumMagnitude, Math.min(maximumMagnitude, value)),
  );
  if (!Number.isFinite(rounded)) {
    throw new RangeError("Motion correction exceeded the finite float32 range");
  }
  return rounded;
}

function applyTranslation(
  positions: Float32Array,
  access: RootAccess,
  frame: number,
  x: number,
  y: number,
  z: number,
  jointCount: number,
  maximumMagnitude: number,
): void {
  for (let joint = 0; joint < jointCount; joint += 1) {
    const offset = positionOffset(frame, joint, jointCount);
    positions[offset] = boundedFloat32(
      positions[offset] + x,
      maximumMagnitude,
    );
    positions[offset + 1] = boundedFloat32(
      positions[offset + 1] + y,
      maximumMagnitude,
    );
    positions[offset + 2] = boundedFloat32(
      positions[offset + 2] + z,
      maximumMagnitude,
    );
  }
  if (access.values !== undefined) {
    const offset = frame * access.components;
    access.values[offset + access.indices[0]] = boundedFloat32(
      access.values[offset + access.indices[0]] + x,
      maximumMagnitude,
    );
    access.values[offset + access.indices[1]] = boundedFloat32(
      access.values[offset + access.indices[1]] + y,
      maximumMagnitude,
    );
    access.values[offset + access.indices[2]] = boundedFloat32(
      access.values[offset + access.indices[2]] + z,
      maximumMagnitude,
    );
  }
}

function constraintTargets(
  input: MotionPostprocessInput,
): ConstraintTargets | undefined {
  const constraints = input.constraints ?? [];
  if (constraints.length === 0) return undefined;
  const manifest = input.constraintManifest;
  if (manifest === undefined) {
    throw new TypeError(
      "A model-pack manifest is required to decode normalized constraints",
    );
  }
  const rootSlice = manifest.motion_layout?.root_pos;
  const stats = manifest.stats?.motion;
  if (rootSlice === undefined || rootSlice[1] - rootSlice[0] < 3) {
    throw new TypeError("The model pack does not expose a 3D root-position slice");
  }
  if (
    stats === undefined ||
    stats.mean.length < rootSlice[0] + 3 ||
    stats.normalization_denominator.length < rootSlice[0] + 3
  ) {
    throw new TypeError("The model pack has incomplete root normalization statistics");
  }

  const values = new Float64Array(input.frameCount * 3);
  const present = new Uint8Array(input.frameCount * 3);
  const exact = new Uint8Array(input.frameCount * 3);
  const priorities = new Uint32Array(input.frameCount * 3);
  const constrainedFrames = new Uint8Array(input.frameCount);
  const frameOffset = safeInteger(
    input.frameOffset,
    0,
    "Frame offset",
    0,
    Number.MAX_SAFE_INTEGER,
  );

  constraints.forEach((constraint, constraintIndex) => {
    if (
      constraint.values.length !== manifest.dimensions.motion_dim ||
      constraint.mask.length !== manifest.dimensions.motion_dim
    ) {
      throw new RangeError(
        `Constraint ${constraint.id} does not match the model motion dimension`,
      );
    }
    const absoluteEnd = constraint.endFrame ?? constraint.frame;
    if (
      !Number.isSafeInteger(constraint.frame) ||
      !Number.isSafeInteger(absoluteEnd) ||
      constraint.frame < 0 ||
      absoluteEnd < constraint.frame
    ) {
      throw new RangeError(`Constraint ${constraint.id} has invalid frame bounds`);
    }
    for (let feature = 0; feature < constraint.values.length; feature += 1) {
      if (
        !Number.isFinite(constraint.values[feature]) ||
        !Number.isFinite(constraint.mask[feature])
      ) {
        throw new RangeError(`Constraint ${constraint.id} contains non-finite data`);
      }
    }
    const first = Math.max(0, constraint.frame - frameOffset);
    const last = Math.min(
      input.frameCount - 1,
      absoluteEnd - frameOffset,
    );
    if (last < first) return;

    // Full-body constraints have the final say, matching the Python pipeline.
    const priority =
      (constraint.kind === "full-body" ? 2 : 1) * (constraints.length + 1) +
      constraintIndex +
      1;
    for (let frame = first; frame <= last; frame += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const feature = rootSlice[0] + axis;
        if (constraint.mask[feature] <= 0.5) continue;
        const targetOffset = frame * 3 + axis;
        if (priority < priorities[targetOffset]) continue;
        const raw =
          constraint.values[feature] *
            stats.normalization_denominator[feature] +
          stats.mean[feature];
        if (!Number.isFinite(raw)) {
          throw new RangeError(
            `Constraint ${constraint.id} decodes to a non-finite root target`,
          );
        }
        values[targetOffset] = raw;
        present[targetOffset] = 1;
        exact[targetOffset] = constraint.kind === "full-body" ? 1 : 0;
        priorities[targetOffset] = priority;
        constrainedFrames[frame] = 1;
      }
    }
  });
  return { values, present, exact, constrainedFrames };
}

function rootCorrections(
  positions: Float32Array,
  access: RootAccess,
  input: MotionPostprocessInput,
  targets: ConstraintTargets | undefined,
  options: ResolvedOptions,
): Float64Array {
  const correction = new Float64Array(input.frameCount * 3);
  if (targets === undefined) return correction;
  const anchors = new Float64Array(input.frameCount * 3);
  const anchorPresent = new Uint8Array(input.frameCount * 3);

  for (let frame = 0; frame < input.frameCount; frame += 1) {
    const offset = frame * 3;
    let deltaX = 0;
    let deltaZ = 0;
    const hasX = targets.present[offset] === 1;
    const hasZ = targets.present[offset + 2] === 1;
    if (hasX) {
      deltaX =
        targets.values[offset] -
        rootValue(positions, access, frame, 0, input.jointCount);
    }
    if (hasZ) {
      deltaZ =
        targets.values[offset + 2] -
        rootValue(positions, access, frame, 2, input.jointCount);
    }
    if (hasX || hasZ) {
      const exact =
        (hasX && targets.exact[offset] === 1) ||
        (hasZ && targets.exact[offset + 2] === 1);
      const margin = exact ? 0 : options.rootMargin;
      const distance = Math.hypot(deltaX, deltaZ);
      const scale =
        distance <= margin || distance === 0
          ? 0
          : (distance - margin) / distance;
      const [boundedX, boundedZ] = clampMagnitude(
        deltaX * scale,
        deltaZ * scale,
        options.maxRootCorrection,
      );
      if (hasX) {
        anchors[offset] = boundedX;
        anchorPresent[offset] = 1;
      }
      if (hasZ) {
        anchors[offset + 2] = boundedZ;
        anchorPresent[offset + 2] = 1;
      }
    }
    if (targets.present[offset + 1] === 1) {
      const delta =
        targets.values[offset + 1] -
        rootValue(positions, access, frame, 1, input.jointCount);
      anchors[offset + 1] = Math.max(
        -options.maxRootCorrection,
        Math.min(options.maxRootCorrection, delta),
      );
      anchorPresent[offset + 1] = 1;
    }
  }

  for (let axis = 0; axis < 3; axis += 1) {
    const frames: number[] = [];
    for (let frame = 0; frame < input.frameCount; frame += 1) {
      if (anchorPresent[frame * 3 + axis] === 1) frames.push(frame);
    }
    if (frames.length === 0) continue;

    const first = frames[0];
    const firstValue = anchors[first * 3 + axis];
    const fadeStart = Math.max(0, first - options.constraintBlendFrames);
    for (let frame = fadeStart; frame <= first; frame += 1) {
      const alpha =
        first === fadeStart ? 1 : (frame - fadeStart) / (first - fadeStart);
      correction[frame * 3 + axis] = firstValue * alpha;
    }
    for (let index = 0; index < frames.length - 1; index += 1) {
      const start = frames[index];
      const end = frames[index + 1];
      const startValue = anchors[start * 3 + axis];
      const endValue = anchors[end * 3 + axis];
      for (let frame = start; frame <= end; frame += 1) {
        const alpha = end === start ? 1 : (frame - start) / (end - start);
        correction[frame * 3 + axis] =
          startValue + (endValue - startValue) * alpha;
      }
    }
    const last = frames[frames.length - 1];
    const lastValue = anchors[last * 3 + axis];
    const fadeEnd = Math.min(
      input.frameCount - 1,
      last + options.constraintBlendFrames,
    );
    for (let frame = last; frame <= fadeEnd; frame += 1) {
      const alpha =
        last === fadeEnd ? 1 : (fadeEnd - frame) / (fadeEnd - last);
      correction[frame * 3 + axis] = lastValue * alpha;
    }
  }
  return correction;
}

function contactActive(
  contacts: FootContactTrack,
  frame: number,
  channel: number,
  threshold: number,
): boolean {
  return Number(contacts.values[frame * contacts.channels + channel]) >= threshold;
}

function validateContacts(
  contacts: FootContactTrack | undefined,
  input: MotionPostprocessInput,
): void {
  if (contacts === undefined) return;
  const channels = safeInteger(
    contacts.channels,
    contacts.jointIndices.length,
    "Contact channel count",
    1,
    input.jointCount,
  );
  if (contacts.jointIndices.length !== channels) {
    throw new RangeError(
      "Contact joint metadata must contain one joint index per channel",
    );
  }
  validateFiniteTrack(
    contacts.values,
    input.frameCount * channels,
    "Foot contacts",
    Number.MAX_VALUE,
  );
  for (const joint of contacts.jointIndices) {
    if (!Number.isSafeInteger(joint) || joint < 0 || joint >= input.jointCount) {
      throw new RangeError(`Contact joint index ${joint} is outside the skeleton`);
    }
  }
}

function footSlidingDistance(
  positions: Float32Array,
  input: MotionPostprocessInput,
  contacts: FootContactTrack | undefined,
  threshold: number,
): { distance: number; samples: number } {
  if (contacts === undefined) return { distance: 0, samples: 0 };
  let distance = 0;
  let samples = 0;
  for (let channel = 0; channel < contacts.channels; channel += 1) {
    const joint = contacts.jointIndices[channel];
    for (let frame = 1; frame < input.frameCount; frame += 1) {
      if (
        !contactActive(contacts, frame - 1, channel, threshold) ||
        !contactActive(contacts, frame, channel, threshold)
      ) {
        continue;
      }
      const previous = positionOffset(frame - 1, joint, input.jointCount);
      const current = positionOffset(frame, joint, input.jointCount);
      distance += Math.hypot(
        positions[current] - positions[previous],
        positions[current + 2] - positions[previous + 2],
      );
      samples += 1;
    }
  }
  return { distance, samples };
}

function projectFootTranslationAtConstraint(
  proposedX: number,
  proposedZ: number,
  positions: Float32Array,
  access: RootAccess,
  input: MotionPostprocessInput,
  targets: ConstraintTargets | undefined,
  options: ResolvedOptions,
  frame: number,
): [number, number] {
  if (targets === undefined) return [proposedX, proposedZ];
  const offset = frame * 3;
  const hasX = targets.present[offset] === 1;
  const hasZ = targets.present[offset + 2] === 1;
  if (!hasX && !hasZ) return [proposedX, proposedZ];

  const rootX = rootValue(positions, access, frame, 0, input.jointCount);
  const rootZ = rootValue(positions, access, frame, 2, input.jointCount);
  const exact =
    (hasX && targets.exact[offset] === 1) ||
    (hasZ && targets.exact[offset + 2] === 1);
  const margin = exact ? 0 : options.rootMargin;
  let candidateX = rootX + proposedX;
  let candidateZ = rootZ + proposedZ;

  if (hasX && hasZ) {
    const targetX = targets.values[offset];
    const targetZ = targets.values[offset + 2];
    const deltaX = candidateX - targetX;
    const deltaZ = candidateZ - targetZ;
    const distance = Math.hypot(deltaX, deltaZ);
    if (distance > margin) {
      const scale = distance === 0 ? 0 : margin / distance;
      candidateX = targetX + deltaX * scale;
      candidateZ = targetZ + deltaZ * scale;
    }
  } else if (hasX) {
    const target = targets.values[offset];
    candidateX = Math.max(target - margin, Math.min(target + margin, candidateX));
  } else {
    const target = targets.values[offset + 2];
    candidateZ = Math.max(target - margin, Math.min(target + margin, candidateZ));
  }
  return [candidateX - rootX, candidateZ - rootZ];
}

function applyFootLocks(
  positions: Float32Array,
  access: RootAccess,
  input: MotionPostprocessInput,
  targets: ConstraintTargets | undefined,
  options: ResolvedOptions,
  translations: Float32Array,
): void {
  const contacts = input.contacts;
  if (
    contacts === undefined ||
    options.footLockStrength === 0 ||
    options.maxFootCorrectionPerFrame === 0
  ) {
    return;
  }
  const proposalX = new Float64Array(input.frameCount);
  const proposalZ = new Float64Array(input.frameCount);
  const proposalCount = new Uint16Array(input.frameCount);

  for (let channel = 0; channel < contacts.channels; channel += 1) {
    const joint = contacts.jointIndices[channel];
    let frame = 0;
    while (frame < input.frameCount) {
      if (!contactActive(contacts, frame, channel, options.contactThreshold)) {
        frame += 1;
        continue;
      }
      const start = frame;
      while (
        frame + 1 < input.frameCount &&
        contactActive(contacts, frame + 1, channel, options.contactThreshold)
      ) {
        frame += 1;
      }
      const end = frame;
      const anchorOffset = positionOffset(start, joint, input.jointCount);
      const anchorX = positions[anchorOffset];
      const anchorZ = positions[anchorOffset + 2];
      for (let runFrame = start; runFrame <= end; runFrame += 1) {
        const offset = positionOffset(runFrame, joint, input.jointCount);
        const ramp = Math.min(
          1,
          (runFrame - start + 1) / options.footLockRampFrames,
          (end - runFrame + 1) / options.footLockRampFrames,
        );
        proposalX[runFrame] +=
          (anchorX - positions[offset]) * options.footLockStrength * ramp;
        proposalZ[runFrame] +=
          (anchorZ - positions[offset + 2]) * options.footLockStrength * ramp;
        proposalCount[runFrame] += 1;
      }
      frame += 1;
    }
  }

  for (let frame = 0; frame < input.frameCount; frame += 1) {
    if (proposalCount[frame] === 0) continue;
    let proposedX = proposalX[frame] / proposalCount[frame];
    let proposedZ = proposalZ[frame] / proposalCount[frame];
    [proposedX, proposedZ] = clampMagnitude(
      proposedX,
      proposedZ,
      options.maxFootCorrectionPerFrame,
    );
    [proposedX, proposedZ] = projectFootTranslationAtConstraint(
      proposedX,
      proposedZ,
      positions,
      access,
      input,
      targets,
      options,
      frame,
    );
    applyTranslation(
      positions,
      access,
      frame,
      proposedX,
      0,
      proposedZ,
      input.jointCount,
      options.maxPositionMagnitude,
    );
    translations[frame * 3] = boundedFloat32(
      translations[frame * 3] + proposedX,
      options.maxPositionMagnitude,
    );
    translations[frame * 3 + 2] = boundedFloat32(
      translations[frame * 3 + 2] + proposedZ,
      options.maxPositionMagnitude,
    );
  }
}

function rootConstraintMeanError(
  positions: Float32Array,
  access: RootAccess,
  input: MotionPostprocessInput,
  targets: ConstraintTargets | undefined,
): number {
  if (targets === undefined) return 0;
  let sum = 0;
  let count = 0;
  for (let frame = 0; frame < input.frameCount; frame += 1) {
    const offset = frame * 3;
    const hasX = targets.present[offset] === 1;
    const hasY = targets.present[offset + 1] === 1;
    const hasZ = targets.present[offset + 2] === 1;
    if (!hasX && !hasY && !hasZ) continue;
    const dx = hasX
      ? rootValue(positions, access, frame, 0, input.jointCount) -
        targets.values[offset]
      : 0;
    const dy = hasY
      ? rootValue(positions, access, frame, 1, input.jointCount) -
        targets.values[offset + 1]
      : 0;
    const dz = hasZ
      ? rootValue(positions, access, frame, 2, input.jointCount) -
        targets.values[offset + 2]
      : 0;
    sum += Math.hypot(dx, dy, dz);
    count += 1;
  }
  return count === 0 ? 0 : sum / count;
}

/**
 * Deterministic, browser-native translation correction.
 *
 * This is deliberately bounded and dependency-free: it smooths the decoded
 * root trajectory through sparse targets, then applies contact-run foot locks
 * while projecting the root back inside each constraint's allowed margin.
 * It does not perform the native pipeline's rotation-space IK.
 */
export function postprocessMotion(
  input: MotionPostprocessInput,
  options: MotionPostprocessOptions = {},
): MotionPostprocessResult {
  const resolved = resolveOptions(options);
  const frameCount = safeInteger(
    input.frameCount,
    0,
    "Frame count",
    1,
    MAX_FRAMES,
  );
  const jointCount = safeInteger(
    input.jointCount,
    0,
    "Joint count",
    1,
    MAX_JOINTS,
  );
  if (frameCount !== input.frameCount || jointCount !== input.jointCount) {
    throw new RangeError("Motion dimensions must be safe integers");
  }
  validateFiniteTrack(
    input.positions,
    frameCount * jointCount * 3,
    "Joint positions",
    resolved.maxPositionMagnitude,
  );
  validateContacts(input.contacts, input);

  const positions = input.positions.slice();
  const access = resolveRootAccess(input, resolved.maxPositionMagnitude);
  const targets = constraintTargets(input);
  const beforeConstraintError = rootConstraintMeanError(
    positions,
    access,
    input,
    targets,
  );
  const correction = rootCorrections(
    positions,
    access,
    input,
    targets,
    resolved,
  );
  const translations = new Float32Array(frameCount * 3);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * 3;
    const x = correction[offset];
    const y = correction[offset + 1];
    const z = correction[offset + 2];
    applyTranslation(
      positions,
      access,
      frame,
      x,
      y,
      z,
      jointCount,
      resolved.maxPositionMagnitude,
    );
    translations[offset] = boundedFloat32(x, resolved.maxPositionMagnitude);
    translations[offset + 1] = boundedFloat32(
      y,
      resolved.maxPositionMagnitude,
    );
    translations[offset + 2] = boundedFloat32(
      z,
      resolved.maxPositionMagnitude,
    );
  }

  const beforeFootLock = footSlidingDistance(
    positions,
    input,
    input.contacts,
    resolved.contactThreshold,
  );
  applyFootLocks(
    positions,
    access,
    input,
    targets,
    resolved,
    translations,
  );
  const afterFootLock = footSlidingDistance(
    positions,
    input,
    input.contacts,
    resolved.contactThreshold,
  );
  let constrainedFrames = 0;
  if (targets !== undefined) {
    for (const value of targets.constrainedFrames) constrainedFrames += value;
  }
  let maxAppliedTranslation = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * 3;
    maxAppliedTranslation = Math.max(
      maxAppliedTranslation,
      Math.hypot(
        translations[offset],
        translations[offset + 1],
        translations[offset + 2],
      ),
    );
  }

  return {
    positions,
    roots: access.values,
    rootTranslations: translations,
    metrics: {
      rootConstraintFrames: constrainedFrames,
      rootConstraintMeanErrorBefore: beforeConstraintError,
      rootConstraintMeanErrorAfter: rootConstraintMeanError(
        positions,
        access,
        input,
        targets,
      ),
      contactSamples: beforeFootLock.samples,
      footSlidingDistanceBefore: beforeFootLock.distance,
      footSlidingDistanceAfter: afterFootLock.distance,
      maxAppliedTranslation,
    },
  };
}

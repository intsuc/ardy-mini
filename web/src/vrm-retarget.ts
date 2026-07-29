// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type { VRMHumanBoneName } from "@pixiv/three-vrm";

import type {
  RotationTrack,
  SkeletonMetadata,
  StructuredMotionResult,
} from "./motion-data";

export type VrmMetaVersion = "0" | "1";
export type QuaternionTuple = readonly [
  x: number,
  y: number,
  z: number,
  w: number,
];
export type Vector3Tuple = readonly [x: number, y: number, z: number];

/**
 * Distance from the Core27 hips origin to the lowest neutral-pose toe.
 * It is the source-rig scale used when translating ARDY root motion to a VRM.
 */
export const CORE27_REST_HIPS_HEIGHT = 0.9544128252334833;

export interface VrmRetargetBindingDefinition {
  readonly sourceJointName: string;
  readonly targetBone: VRMHumanBoneName;
}

/**
 * Core27 has one more spine segment than the VRM humanoid hierarchy. Spine2 is
 * deliberately omitted: its rotation is folded into the Spine3 → upperChest
 * local rotation by retargeting through source-global rotations.
 */
export const CORE27_VRM_BINDINGS: readonly VrmRetargetBindingDefinition[] =
  Object.freeze([
    { sourceJointName: "Hips", targetBone: "hips" },
    { sourceJointName: "Spine", targetBone: "spine" },
    { sourceJointName: "Spine1", targetBone: "chest" },
    { sourceJointName: "Spine3", targetBone: "upperChest" },
    { sourceJointName: "Neck", targetBone: "neck" },
    { sourceJointName: "Head", targetBone: "head" },
    { sourceJointName: "LeftShoulder", targetBone: "leftShoulder" },
    { sourceJointName: "LeftArm", targetBone: "leftUpperArm" },
    { sourceJointName: "LeftForeArm", targetBone: "leftLowerArm" },
    { sourceJointName: "LeftHand", targetBone: "leftHand" },
    {
      sourceJointName: "LeftHandThumb1",
      targetBone: "leftThumbMetacarpal",
    },
    { sourceJointName: "RightShoulder", targetBone: "rightShoulder" },
    { sourceJointName: "RightArm", targetBone: "rightUpperArm" },
    { sourceJointName: "RightForeArm", targetBone: "rightLowerArm" },
    { sourceJointName: "RightHand", targetBone: "rightHand" },
    {
      sourceJointName: "RightHandThumb1",
      targetBone: "rightThumbMetacarpal",
    },
    { sourceJointName: "LeftUpLeg", targetBone: "leftUpperLeg" },
    { sourceJointName: "LeftLeg", targetBone: "leftLowerLeg" },
    { sourceJointName: "LeftFoot", targetBone: "leftFoot" },
    { sourceJointName: "LeftToeBase", targetBone: "leftToes" },
    { sourceJointName: "RightUpLeg", targetBone: "rightUpperLeg" },
    { sourceJointName: "RightLeg", targetBone: "rightLowerLeg" },
    { sourceJointName: "RightFoot", targetBone: "rightFoot" },
    { sourceJointName: "RightToeBase", targetBone: "rightToes" },
  ] satisfies readonly VrmRetargetBindingDefinition[]);

/**
 * Runtime copy of the relevant VRM humanoid hierarchy. Keeping this local
 * avoids a runtime import from @pixiv/three-vrm in the initial application
 * bundle; the VRM loader itself can remain dynamically imported.
 */
const VRM_TARGET_PARENTS: Readonly<
  Partial<Record<VRMHumanBoneName, VRMHumanBoneName | null>>
> = Object.freeze({
  hips: null,
  spine: "hips",
  chest: "spine",
  upperChest: "chest",
  neck: "upperChest",
  head: "neck",
  leftShoulder: "upperChest",
  leftUpperArm: "leftShoulder",
  leftLowerArm: "leftUpperArm",
  leftHand: "leftLowerArm",
  leftThumbMetacarpal: "leftHand",
  rightShoulder: "upperChest",
  rightUpperArm: "rightShoulder",
  rightLowerArm: "rightUpperArm",
  rightHand: "rightLowerArm",
  rightThumbMetacarpal: "rightHand",
  leftUpperLeg: "hips",
  leftLowerLeg: "leftUpperLeg",
  leftFoot: "leftLowerLeg",
  leftToes: "leftFoot",
  rightUpperLeg: "hips",
  rightLowerLeg: "rightUpperLeg",
  rightFoot: "rightLowerLeg",
  rightToes: "rightFoot",
});

export interface VrmRetargetBinding {
  readonly sourceJointName: string;
  readonly sourceJointIndex: number;
  readonly targetBone: VRMHumanBoneName;
  /**
   * Source joint mapped to the nearest target-hierarchy parent that is
   * actually present in this VRM. Null means the normalized-rig root.
   */
  readonly parentSourceJointIndex: number | null;
}

export interface VrmRetargetPlan {
  readonly bindings: readonly VrmRetargetBinding[];
  readonly hipsSourceJointIndex: number;
  readonly metaVersion: VrmMetaVersion;
  readonly positionScale: number;
  readonly sourceHipsHeight: number;
  readonly targetHipsHeight: number;
}

export interface CreateVrmRetargetPlanOptions {
  readonly presentBones: Iterable<VRMHumanBoneName>;
  readonly targetHipsHeight: number;
  readonly metaVersion: VrmMetaVersion;
  readonly sourceHipsHeight?: number;
}

export interface VrmRetargetRotation {
  readonly targetBone: VRMHumanBoneName;
  readonly rotation: QuaternionTuple;
}

export interface VrmRetargetFrame {
  /**
   * Absolute position for the normalized VRM hips node, not a rest-pose delta.
   */
  readonly hipsPosition: Vector3Tuple;
  readonly rotations: readonly VrmRetargetRotation[];
}

function positiveFinite(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive finite number.`);
  }
  return value;
}

/**
 * Build a model-specific binding plan. Missing optional VRM bones are skipped,
 * and children bind relative to their nearest mapped ancestor that is present.
 */
export function createVrmRetargetPlan(
  skeleton: SkeletonMetadata,
  options: CreateVrmRetargetPlanOptions,
): VrmRetargetPlan {
  const targetHipsHeight = positiveFinite(
    options.targetHipsHeight,
    "Target hips height",
  );
  const sourceHipsHeight = positiveFinite(
    options.sourceHipsHeight ?? CORE27_REST_HIPS_HEIGHT,
    "Source hips height",
  );
  const presentBones = new Set(options.presentBones);
  if (!presentBones.has("hips")) {
    throw new RangeError("The VRM does not expose its required hips bone.");
  }

  const sourceJointByName = new Map(
    skeleton.jointNames.map((name, index) => [name, index] as const),
  );
  const candidates = CORE27_VRM_BINDINGS.flatMap((definition) => {
    const sourceJointIndex = sourceJointByName.get(
      definition.sourceJointName,
    );
    return sourceJointIndex === undefined ||
      !presentBones.has(definition.targetBone)
      ? []
      : [
          {
            ...definition,
            sourceJointIndex,
          },
        ];
  });
  const hips = candidates.find(({ targetBone }) => targetBone === "hips");
  if (!hips) {
    throw new RangeError("The source skeleton does not contain a Hips joint.");
  }

  const candidateByTarget = new Map(
    candidates.map((candidate) => [candidate.targetBone, candidate] as const),
  );
  const bindings = candidates.map<VrmRetargetBinding>((candidate) => {
    let targetParent = VRM_TARGET_PARENTS[candidate.targetBone] ?? null;
    let parentSourceJointIndex: number | null = null;
    while (targetParent !== null) {
      const parentCandidate = candidateByTarget.get(targetParent);
      if (parentCandidate) {
        parentSourceJointIndex = parentCandidate.sourceJointIndex;
        break;
      }
      targetParent = VRM_TARGET_PARENTS[targetParent] ?? null;
    }
    return Object.freeze({
      sourceJointName: candidate.sourceJointName,
      sourceJointIndex: candidate.sourceJointIndex,
      targetBone: candidate.targetBone,
      parentSourceJointIndex,
    });
  });

  return Object.freeze({
    bindings: Object.freeze(bindings),
    hipsSourceJointIndex: hips.sourceJointIndex,
    metaVersion: options.metaVersion,
    positionScale: targetHipsHeight / sourceHipsHeight,
    sourceHipsHeight,
    targetHipsHeight,
  });
}

export function normalizeQuaternion(
  quaternion: QuaternionTuple,
): QuaternionTuple {
  const magnitude = Math.hypot(...quaternion);
  if (magnitude <= Number.EPSILON) return [0, 0, 0, 1];
  return [
    quaternion[0] / magnitude,
    quaternion[1] / magnitude,
    quaternion[2] / magnitude,
    quaternion[3] / magnitude,
  ];
}

export function multiplyQuaternions(
  left: QuaternionTuple,
  right: QuaternionTuple,
): QuaternionTuple {
  const [lx, ly, lz, lw] = left;
  const [rx, ry, rz, rw] = right;
  return normalizeQuaternion([
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
    lw * rw - lx * rx - ly * ry - lz * rz,
  ]);
}

export function invertQuaternion(
  quaternion: QuaternionTuple,
): QuaternionTuple {
  const normalized = normalizeQuaternion(quaternion);
  return [-normalized[0], -normalized[1], -normalized[2], normalized[3]];
}

export function slerpQuaternions(
  start: QuaternionTuple,
  end: QuaternionTuple,
  alpha: number,
): QuaternionTuple {
  const clampedAlpha = Math.max(0, Math.min(1, alpha));
  const from = normalizeQuaternion(start);
  let to = normalizeQuaternion(end);
  let cosine =
    from[0] * to[0] +
    from[1] * to[1] +
    from[2] * to[2] +
    from[3] * to[3];
  if (cosine < 0) {
    to = [-to[0], -to[1], -to[2], -to[3]];
    cosine = -cosine;
  }
  if (cosine > 0.9995) {
    return normalizeQuaternion([
      from[0] + (to[0] - from[0]) * clampedAlpha,
      from[1] + (to[1] - from[1]) * clampedAlpha,
      from[2] + (to[2] - from[2]) * clampedAlpha,
      from[3] + (to[3] - from[3]) * clampedAlpha,
    ]);
  }

  const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
  const sine = Math.sin(angle);
  const startWeight = Math.sin((1 - clampedAlpha) * angle) / sine;
  const endWeight = Math.sin(clampedAlpha * angle) / sine;
  return normalizeQuaternion([
    from[0] * startWeight + to[0] * endWeight,
    from[1] * startWeight + to[1] * endWeight,
    from[2] * startWeight + to[2] * endWeight,
    from[3] * startWeight + to[3] * endWeight,
  ]);
}

function matrix3Quaternion(
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
  return normalizeQuaternion([x, y, z, w]);
}

function rotationAtFrame(
  track: RotationTrack,
  frame: number,
  joint: number,
): QuaternionTuple {
  const components = track.shape[2];
  const offset = (frame * track.shape[1] + joint) * components;
  if (components === 4) {
    return normalizeQuaternion([
      track.values[offset],
      track.values[offset + 1],
      track.values[offset + 2],
      track.values[offset + 3],
    ]);
  }
  return matrix3Quaternion(track.values, offset);
}

/** Sample either supported rotation representation at a fractional frame. */
export function sampleRotationTrack(
  track: RotationTrack,
  frameCursor: number,
  joint: number,
): QuaternionTuple {
  if (!Number.isFinite(frameCursor)) {
    throw new RangeError("Rotation frame cursor must be finite.");
  }
  if (!Number.isInteger(joint) || joint < 0 || joint >= track.shape[1]) {
    throw new RangeError("Rotation joint index is outside the track.");
  }
  const cursor = Math.max(0, Math.min(track.shape[0] - 1, frameCursor));
  const frame0 = Math.floor(cursor);
  const frame1 = Math.min(frame0 + 1, track.shape[0] - 1);
  const first = rotationAtFrame(track, frame0, joint);
  return frame0 === frame1
    ? first
    : slerpQuaternions(
        first,
        rotationAtFrame(track, frame1, joint),
        cursor - frame0,
      );
}

function writeQuaternion(
  output: Float32Array,
  joint: number,
  value: QuaternionTuple,
): void {
  output.set(value, joint * 4);
}

function readQuaternion(
  values: Float32Array,
  joint: number,
): QuaternionTuple {
  const offset = joint * 4;
  return [
    values[offset],
    values[offset + 1],
    values[offset + 2],
    values[offset + 3],
  ];
}

/**
 * Sample source-global rotations. A supplied global track is authoritative;
 * otherwise globals are reconstructed recursively from the local track.
 */
export function sampleGlobalJointRotations(
  motion: StructuredMotionResult,
  frameCursor: number,
): Float32Array | null {
  const jointCount = motion.skeleton.jointNames.length;
  const output = new Float32Array(jointCount * 4);
  if (motion.globalRotations) {
    for (let joint = 0; joint < jointCount; joint += 1) {
      writeQuaternion(
        output,
        joint,
        sampleRotationTrack(motion.globalRotations, frameCursor, joint),
      );
    }
    return output;
  }
  if (!motion.localRotations) return null;

  const state = new Uint8Array(jointCount);
  const resolve = (joint: number): QuaternionTuple => {
    if (state[joint] === 2) return readQuaternion(output, joint);
    if (state[joint] === 1) {
      throw new RangeError("Source skeleton parent links contain a cycle.");
    }
    state[joint] = 1;
    const local = sampleRotationTrack(
      motion.localRotations!,
      frameCursor,
      joint,
    );
    const parent = motion.skeleton.parents[joint];
    const global =
      parent === -1
        ? local
        : multiplyQuaternions(resolve(parent), local);
    writeQuaternion(output, joint, global);
    state[joint] = 2;
    return global;
  };
  for (let joint = 0; joint < jointCount; joint += 1) resolve(joint);
  return output;
}

/** Interpolate one source joint's world-space position. */
export function sampleJointPosition(
  motion: StructuredMotionResult,
  frameCursor: number,
  joint: number,
): Vector3Tuple {
  if (!Number.isFinite(frameCursor)) {
    throw new RangeError("Position frame cursor must be finite.");
  }
  if (
    !Number.isInteger(joint) ||
    joint < 0 ||
    joint >= motion.skeleton.jointNames.length
  ) {
    throw new RangeError("Position joint index is outside the motion.");
  }
  const cursor = Math.max(0, Math.min(motion.frameCount - 1, frameCursor));
  const frame0 = Math.floor(cursor);
  const frame1 = Math.min(frame0 + 1, motion.frameCount - 1);
  const alpha = cursor - frame0;
  const stride = motion.skeleton.jointNames.length * 3;
  const offset0 = frame0 * stride + joint * 3;
  const offset1 = frame1 * stride + joint * 3;
  return [
    motion.positions[offset0] +
      (motion.positions[offset1] - motion.positions[offset0]) * alpha,
    motion.positions[offset0 + 1] +
      (motion.positions[offset1 + 1] - motion.positions[offset0 + 1]) *
        alpha,
    motion.positions[offset0 + 2] +
      (motion.positions[offset1 + 2] - motion.positions[offset0 + 2]) *
        alpha,
  ];
}

/**
 * VRM0's scene is rotated by PI around Y by VRMUtils.rotateVRM0. Its
 * normalized-rig input therefore needs the inverse basis transform too.
 */
export function toVrmPosition(
  position: Vector3Tuple,
  metaVersion: VrmMetaVersion,
): Vector3Tuple {
  return metaVersion === "0"
    ? [-position[0], position[1], -position[2]]
    : [...position];
}

export function toVrmQuaternion(
  quaternion: QuaternionTuple,
  metaVersion: VrmMetaVersion,
): QuaternionTuple {
  const normalized = normalizeQuaternion(quaternion);
  return metaVersion === "0"
    ? [-normalized[0], normalized[1], -normalized[2], normalized[3]]
    : normalized;
}

function assertPlanMatchesMotion(
  motion: StructuredMotionResult,
  plan: VrmRetargetPlan,
): void {
  for (const binding of plan.bindings) {
    if (
      motion.skeleton.jointNames[binding.sourceJointIndex] !==
      binding.sourceJointName
    ) {
      throw new RangeError(
        "The VRM retarget plan does not match the motion skeleton.",
      );
    }
  }
}

/**
 * Produce an allocation-only, renderer-independent normalized-rig pose.
 * `hipsPosition` can be copied directly to the normalized hips node.
 */
export function retargetMotionFrame(
  motion: StructuredMotionResult,
  frameCursor: number,
  plan: VrmRetargetPlan,
): VrmRetargetFrame {
  assertPlanMatchesMotion(motion, plan);
  const sourceHips = sampleJointPosition(
    motion,
    frameCursor,
    plan.hipsSourceJointIndex,
  );
  const hipsPosition = toVrmPosition(
    [
      sourceHips[0] * plan.positionScale,
      sourceHips[1] * plan.positionScale,
      sourceHips[2] * plan.positionScale,
    ],
    plan.metaVersion,
  );
  const sourceGlobals = sampleGlobalJointRotations(motion, frameCursor);
  if (!sourceGlobals) {
    return Object.freeze({
      hipsPosition,
      rotations: Object.freeze([]),
    });
  }

  const rotations = plan.bindings.map<VrmRetargetRotation>((binding) => {
    const sourceGlobal = readQuaternion(
      sourceGlobals,
      binding.sourceJointIndex,
    );
    const sourceParentGlobal =
      binding.parentSourceJointIndex === null
        ? ([0, 0, 0, 1] as const)
        : readQuaternion(sourceGlobals, binding.parentSourceJointIndex);
    const targetLocal = multiplyQuaternions(
      invertQuaternion(sourceParentGlobal),
      sourceGlobal,
    );
    return Object.freeze({
      targetBone: binding.targetBone,
      rotation: toVrmQuaternion(targetLocal, plan.metaVersion),
    });
  });
  return Object.freeze({
    hipsPosition,
    rotations: Object.freeze(rotations),
  });
}

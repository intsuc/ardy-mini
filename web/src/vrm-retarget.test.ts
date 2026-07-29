// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  CORE27_JOINT_COUNT,
  CORE27_SKELETON,
  type RotationTrack,
  type StructuredMotionResult,
} from "./motion-data";
import {
  CORE27_REST_HIPS_HEIGHT,
  createVrmRetargetPlan,
  multiplyQuaternions,
  retargetMotionFrame,
  sampleGlobalJointRotations,
  sampleRotationTrack,
  toVrmPosition,
  toVrmQuaternion,
  type QuaternionTuple,
} from "./vrm-retarget";

const IDENTITY: QuaternionTuple = [0, 0, 0, 1];

function quaternionFromAxisAngle(
  axis: readonly [number, number, number],
  angle: number,
): QuaternionTuple {
  const sine = Math.sin(angle / 2);
  return [
    axis[0] * sine,
    axis[1] * sine,
    axis[2] * sine,
    Math.cos(angle / 2),
  ];
}

function quaternionTrack(
  frames: readonly (readonly QuaternionTuple[])[],
): RotationTrack {
  return {
    values: Float32Array.from(frames.flatMap((frame) => frame.flat())),
    shape: [frames.length, frames[0].length, 4],
    format: "quaternion-xyzw",
  };
}

function identityFrame(): QuaternionTuple[] {
  return Array.from({ length: CORE27_JOINT_COUNT }, () => IDENTITY);
}

function motionWith(
  options: {
    frameCount?: number;
    positions?: Float32Array;
    localRotations?: RotationTrack;
    globalRotations?: RotationTrack;
  } = {},
): StructuredMotionResult {
  const frameCount = options.frameCount ?? 1;
  return {
    skeleton: CORE27_SKELETON,
    positions:
      options.positions ??
      new Float32Array(frameCount * CORE27_JOINT_COUNT * 3),
    positionsShape: [frameCount, CORE27_JOINT_COUNT, 3],
    frameCount,
    fps: 20,
    localRotations: options.localRotations,
    globalRotations: options.globalRotations,
  };
}

function allMappedBones() {
  return [
    "hips",
    "spine",
    "chest",
    "upperChest",
    "neck",
    "head",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "leftThumbMetacarpal",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "rightThumbMetacarpal",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
  ] as const;
}

function expectQuaternionClose(
  actual: QuaternionTuple,
  expected: QuaternionTuple,
): void {
  const direct = actual.every(
    (value, index) => Math.abs(value - expected[index]) < 1e-5,
  );
  const negated = actual.every(
    (value, index) => Math.abs(value + expected[index]) < 1e-5,
  );
  expect(direct || negated).toBe(true);
}

describe("Core27 to VRM binding plan", () => {
  it("maps Spine3 to upperChest and folds the skipped Spine2 segment", () => {
    const plan = createVrmRetargetPlan(CORE27_SKELETON, {
      presentBones: allMappedBones(),
      targetHipsHeight: CORE27_REST_HIPS_HEIGHT,
      metaVersion: "1",
    });
    const chest = plan.bindings.find(
      ({ targetBone }) => targetBone === "chest",
    );
    const upperChest = plan.bindings.find(
      ({ targetBone }) => targetBone === "upperChest",
    );

    expect(chest?.sourceJointName).toBe("Spine1");
    expect(upperChest?.sourceJointName).toBe("Spine3");
    expect(upperChest?.parentSourceJointIndex).toBe(
      CORE27_SKELETON.jointNames.indexOf("Spine1"),
    );
    expect(
      plan.bindings.some(({ sourceJointName }) => sourceJointName === "Spine2"),
    ).toBe(false);
  });

  it("uses the nearest mapped parent when optional VRM bones are absent", () => {
    const plan = createVrmRetargetPlan(CORE27_SKELETON, {
      presentBones: ["hips", "spine", "chest", "neck", "leftUpperArm"],
      targetHipsHeight: CORE27_REST_HIPS_HEIGHT,
      metaVersion: "1",
    });
    const neck = plan.bindings.find(({ targetBone }) => targetBone === "neck");
    const arm = plan.bindings.find(
      ({ targetBone }) => targetBone === "leftUpperArm",
    );
    const chestIndex = CORE27_SKELETON.jointNames.indexOf("Spine1");

    expect(neck?.parentSourceJointIndex).toBe(chestIndex);
    expect(arm?.parentSourceJointIndex).toBe(chestIndex);
  });
});

describe("rotation sampling", () => {
  it("SLERPs rotation tracks at fractional frames", () => {
    const end = quaternionFromAxisAngle([0, 1, 0], Math.PI);
    const sampled = sampleRotationTrack(
      quaternionTrack([[IDENTITY], [end]]),
      0.5,
      0,
    );

    expectQuaternionClose(
      sampled,
      quaternionFromAxisAngle([0, 1, 0], Math.PI / 2),
    );
  });

  it("reads the generated motion's row-major rotation matrices", () => {
    const angle = Math.PI / 2;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const track: RotationTrack = {
      values: new Float32Array([
        cosine,
        -sine,
        0,
        sine,
        cosine,
        0,
        0,
        0,
        1,
      ]),
      shape: [1, 1, 9],
      format: "matrix3x3-row-major",
    };

    expectQuaternionClose(
      sampleRotationTrack(track, 0, 0),
      quaternionFromAxisAngle([0, 0, 1], angle),
    );
  });

  it("uses a global track in preference to a conflicting local track", () => {
    const localFrame = identityFrame();
    const globalFrame = identityFrame();
    localFrame[0] = quaternionFromAxisAngle([1, 0, 0], Math.PI / 2);
    globalFrame[0] = quaternionFromAxisAngle([0, 1, 0], Math.PI / 2);
    const sampled = sampleGlobalJointRotations(
      motionWith({
        localRotations: quaternionTrack([localFrame]),
        globalRotations: quaternionTrack([globalFrame]),
      }),
      0,
    );

    expect(sampled).not.toBeNull();
    expectQuaternionClose(
      Array.from(sampled!.slice(0, 4)) as unknown as QuaternionTuple,
      globalFrame[0],
    );
  });

  it("reconstructs source-global rotations from a local-only track", () => {
    const frame = identityFrame();
    const hips = quaternionFromAxisAngle([0, 1, 0], Math.PI / 2);
    const spine = quaternionFromAxisAngle([1, 0, 0], Math.PI / 4);
    frame[0] = hips;
    frame[1] = spine;
    const sampled = sampleGlobalJointRotations(
      motionWith({ localRotations: quaternionTrack([frame]) }),
      0,
    );

    expect(sampled).not.toBeNull();
    expectQuaternionClose(
      Array.from(sampled!.slice(4, 8)) as unknown as QuaternionTuple,
      multiplyQuaternions(hips, spine),
    );
  });
});

describe("VRM coordinate conversion and root motion", () => {
  it("converts VRM0 normalized input around the Y axis", () => {
    const inverseAxisLength = 1 / Math.sqrt(14);
    const rotation = quaternionFromAxisAngle(
      [
        inverseAxisLength,
        2 * inverseAxisLength,
        3 * inverseAxisLength,
      ],
      Math.PI / 3,
    );
    expect(toVrmPosition([1, 2, 3], "0")).toEqual([-1, 2, -3]);
    expectQuaternionClose(toVrmQuaternion(rotation, "0"), [
      -rotation[0],
      rotation[1],
      -rotation[2],
      rotation[3],
    ]);
    expect(toVrmPosition([1, 2, 3], "1")).toEqual([1, 2, 3]);
  });

  it("interpolates and scales the hips from world-space positions", () => {
    const positions = new Float32Array(2 * CORE27_JOINT_COUNT * 3);
    const secondFrame = CORE27_JOINT_COUNT * 3;
    positions[0] = 1;
    positions[1] = 2;
    positions[2] = 3;
    positions[secondFrame] = 3;
    positions[secondFrame + 1] = 4;
    positions[secondFrame + 2] = 5;
    const plan = createVrmRetargetPlan(CORE27_SKELETON, {
      presentBones: ["hips"],
      targetHipsHeight: CORE27_REST_HIPS_HEIGHT / 2,
      metaVersion: "0",
    });
    const frame = retargetMotionFrame(
      motionWith({ frameCount: 2, positions }),
      0.5,
      plan,
    );

    expect(frame.hipsPosition).toEqual([-1, 1.5, -2]);
    expect(frame.rotations).toEqual([]);
  });
});

describe("retargeted local pose", () => {
  it("composes Spine2 and Spine3 into the VRM upperChest rotation", () => {
    const localFrame = identityFrame();
    const spine2 = quaternionFromAxisAngle([1, 0, 0], Math.PI / 3);
    const spine3 = quaternionFromAxisAngle([0, 1, 0], Math.PI / 4);
    localFrame[3] = spine2;
    localFrame[4] = spine3;
    const plan = createVrmRetargetPlan(CORE27_SKELETON, {
      presentBones: ["hips", "spine", "chest", "upperChest"],
      targetHipsHeight: CORE27_REST_HIPS_HEIGHT,
      metaVersion: "1",
    });
    const frame = retargetMotionFrame(
      motionWith({ localRotations: quaternionTrack([localFrame]) }),
      0,
      plan,
    );
    const upperChest = frame.rotations.find(
      ({ targetBone }) => targetBone === "upperChest",
    );

    expect(upperChest).toBeDefined();
    expectQuaternionClose(
      upperChest!.rotation,
      multiplyQuaternions(spine2, spine3),
    );
  });
});

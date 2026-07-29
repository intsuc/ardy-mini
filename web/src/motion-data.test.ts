// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  CORE27_SKELETON,
  isJointInContact,
  normalizeSkeletonMetadata,
  normalizeStructuredMotion,
} from "./motion-data";

const dynamicSkeleton = {
  id: "test-three",
  name: "Test three-joint skeleton",
  joint_names: ["Root", "Toe", "Hand"],
  parents: [-1, 0, 0],
  root_index: 0,
  contact_joint_indices: [1],
  contact_names: ["Toe"],
};

describe("dynamic skeleton metadata", () => {
  it("retains Core27 as the compatibility default", () => {
    expect(normalizeSkeletonMetadata(undefined)).toBe(CORE27_SKELETON);
    const manifestStyle = normalizeSkeletonMetadata({
      name: "cskel27",
      joint_names: [...CORE27_SKELETON.jointNames],
      parents: [...CORE27_SKELETON.parents],
      root_index: 0,
    });
    expect(manifestStyle.contactJointIndices).toEqual([25, 26, 21, 22]);
  });

  it("accepts manifest-style names and validates topology", () => {
    const skeleton = normalizeSkeletonMetadata(dynamicSkeleton);
    expect(skeleton.jointNames).toEqual(["Root", "Toe", "Hand"]);
    expect(skeleton.parents).toEqual([-1, 0, 0]);
    expect(skeleton.contactJointIndices).toEqual([1]);
    expect(Object.isFrozen(skeleton.jointNames)).toBe(true);
  });

  it("rejects cycles, duplicate names, and invalid contact joints", () => {
    expect(() =>
      normalizeSkeletonMetadata({
        ...dynamicSkeleton,
        parents: [-1, 2, 1],
      }),
    ).toThrow(/cycle/);
    expect(() =>
      normalizeSkeletonMetadata({
        ...dynamicSkeleton,
        joint_names: ["Root", "Toe", "Toe"],
      }),
    ).toThrow(/unique/);
    expect(() =>
      normalizeSkeletonMetadata({
        ...dynamicSkeleton,
        contact_joint_indices: [3],
      }),
    ).toThrow(/invalid/);
  });
});

describe("structured motion normalization", () => {
  it("preserves all output tracks and derives explicit shapes", () => {
    const frameCount = 2;
    const positions = Float32Array.from({ length: frameCount * 3 * 3 }, (_, index) => index / 10);
    const localRotations = new Float32Array(frameCount * 3 * 4);
    for (let offset = 3; offset < localRotations.length; offset += 4) localRotations[offset] = 1;
    const globalRotations = new Float32Array(frameCount * 3 * 9);
    for (let offset = 0; offset < globalRotations.length; offset += 9) {
      globalRotations[offset] = 1;
      globalRotations[offset + 4] = 1;
      globalRotations[offset + 8] = 1;
    }

    const motion = normalizeStructuredMotion({
      skeleton: dynamicSkeleton,
      positions,
      frameCount,
      fps: 30,
      normalized_motion: new Float32Array(frameCount * 5),
      local_rotations: localRotations,
      global_rotations: globalRotations,
      roots: new Float32Array(frameCount * 4),
      foot_contacts: [[1], [0]],
    });

    expect(motion.positions).toBe(positions);
    expect(motion.positionsShape).toEqual([2, 3, 3]);
    expect(motion.normalizedMotionShape).toEqual([2, 5]);
    expect(motion.localRotations?.shape).toEqual([2, 3, 4]);
    expect(motion.localRotations?.format).toBe("quaternion-xyzw");
    expect(motion.globalRotations?.shape).toEqual([2, 3, 9]);
    expect(motion.globalRotations?.format).toBe("matrix3x3-row-major");
    expect(motion.rootsShape).toEqual([2, 4]);
    expect(motion.contacts).toEqual(new Uint8Array([1, 0]));
    expect(motion.contactsShape).toEqual([2, 1]);
    expect(isJointInContact(motion, 0, 1)).toBe(true);
    expect(isJointInContact(motion, 1, 1)).toBe(false);
    expect(isJointInContact(motion, 0, 2)).toBe(false);
  });

  it("rejects contacts without a channel-to-joint mapping", () => {
    expect(() =>
      normalizeStructuredMotion({
        skeleton: {
          ...dynamicSkeleton,
          contact_joint_indices: [],
          contact_names: [],
        },
        positions: new Float32Array(3 * 3),
        contacts: [1],
      }),
    ).toThrow(/no contact joint metadata/);
  });

  it("rejects mismatched dynamic position and rotation tracks", () => {
    expect(() =>
      normalizeStructuredMotion({
        skeleton: dynamicSkeleton,
        positions: new Float32Array(8),
      }),
    ).toThrow(/T × 3 × 3/);
    expect(() =>
      normalizeStructuredMotion({
        skeleton: dynamicSkeleton,
        positions: new Float32Array(3 * 3),
        localRotations: new Float32Array(3 * 5),
      }),
    ).toThrow(/T × J × 4 quaternion/);
  });
});

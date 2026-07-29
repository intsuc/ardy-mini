// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  BODY_PROXY_DESCRIPTION,
  CORE27_FOOT_CONTACT_JOINTS,
  CORE27_JOINT_COUNT,
  CORE27_PARENTS,
  CORE27_SKELETON,
  frameAfterElapsed,
  normalizeMotionClip,
  normalizeSkeletonMetadata,
  referenceFrameAtPlayhead,
  skeletonInstanceCounts,
} from "./viewer";

describe("Core27 rendering contract", () => {
  it("contains one root and topologically ordered parent links", () => {
    expect(CORE27_PARENTS).toHaveLength(CORE27_JOINT_COUNT);
    expect(CORE27_PARENTS.filter((parent) => parent === -1)).toHaveLength(1);
    CORE27_PARENTS.forEach((parent, joint) => {
      if (joint > 0) {
        expect(parent).toBeGreaterThanOrEqual(0);
        expect(parent).toBeLessThan(joint);
      }
    });
    expect(CORE27_FOOT_CONTACT_JOINTS).toEqual([25, 26, 21, 22]);
  });
});

describe("motion payload normalization", () => {
  it("accepts worker joints and defaults to 20 FPS", () => {
    const positions = new Float32Array(2 * CORE27_JOINT_COUNT * 3);
    const clip = normalizeMotionClip({
      positions,
      frameCount: 2,
    });
    expect(clip.positions).toBe(positions);
    expect(clip.frameCount).toBe(2);
    expect(clip.fps).toBe(20);
  });

  it("accepts Python-style nested motion data and contact values", () => {
    const frame = Array.from({ length: CORE27_JOINT_COUNT }, () => [0, 1, 2]);
    const clip = normalizeMotionClip({
      motion: {
        posed_joints: [frame],
        foot_contacts: [[true, false, 1, 0]],
        fps: 25,
      },
    });
    expect(clip.positions).toHaveLength(CORE27_JOINT_COUNT * 3);
    expect(clip.contacts).toEqual(new Uint8Array([1, 0, 1, 0]));
    expect(clip.fps).toBe(25);
  });

  it("rejects malformed shapes and non-finite values", () => {
    expect(() => normalizeMotionClip({ positions: new Float32Array(4) })).toThrow(/T × 27 × 3/);
    const values = new Float32Array(CORE27_JOINT_COUNT * 3);
    values[8] = Number.NaN;
    expect(() => normalizeMotionClip({ positions: values })).toThrow(/non-finite/);
  });
});

describe("absolute-clock playback", () => {
  it("loops without accumulating timer drift", () => {
    expect(frameAfterElapsed(38, 0.2, 20, 1, 40, true)).toEqual({
      frame: 2,
      ended: false,
    });
  });

  it("stops exactly at the last frame when looping is off", () => {
    expect(frameAfterElapsed(38, 0.2, 20, 1, 40, false)).toEqual({
      frame: 39,
      ended: true,
    });
  });
});

describe("optional comparison layers", () => {
  it("describes the lightweight body display as a proxy rather than SMPL", () => {
    expect(BODY_PROXY_DESCRIPTION).toMatch(/body proxy/i);
    expect(BODY_PROXY_DESCRIPTION).toMatch(/not an SMPL/i);
  });

  it("sizes every instanced layer from dynamic skeleton metadata", () => {
    expect(skeletonInstanceCounts(CORE27_SKELETON)).toEqual({
      joints: CORE27_JOINT_COUNT,
      bones: CORE27_JOINT_COUNT - 1,
    });
    const branchedSkeleton = normalizeSkeletonMetadata({
      id: "branched-test",
      jointNames: ["root", "left", "right", "tip"],
      parents: [-1, 0, 0, 2],
      rootJointIndex: 0,
      contactJointIndices: [],
    });
    expect(skeletonInstanceCounts(branchedSkeleton)).toEqual({
      joints: 4,
      bones: 3,
    });
  });

  it("keeps a reference with a different FPS on the same wall-clock time", () => {
    expect(referenceFrameAtPlayhead(20, 20, 30, 100)).toBe(30);
    expect(referenceFrameAtPlayhead(80, 20, 30, 100)).toBe(99);
    expect(referenceFrameAtPlayhead(-4, 20, 30, 100)).toBe(0);
    expect(() => referenceFrameAtPlayhead(2, 0, 30, 100)).toThrow(/timing/);
  });
});

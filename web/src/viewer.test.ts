// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import {
  canPreserveMotionContinuity,
  cameraMovementDistance,
  CORE27_FOOT_CONTACT_JOINTS,
  CORE27_JOINT_COUNT,
  CORE27_PARENTS,
  CORE27_SKELETON,
  frameAfterElapsed,
  normalizeMotionClip,
  normalizeSkeletonMetadata,
  SkeletonViewer,
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

describe("camera movement", () => {
  const createViewer = (azimuth: number): SkeletonViewer => {
    const camera = new THREE.PerspectiveCamera();
    camera.position.set(0, 4, 0);
    camera.lookAt(0, 0, 0);
    const target = new THREE.Vector3();

    return Object.assign(Object.create(SkeletonViewer.prototype), {
      camera,
      controls: {
        target,
        getAzimuthalAngle: vi.fn(() => azimuth),
        getDistance: vi.fn(() => 4),
        update: vi.fn(),
      },
      cameraForward: new THREE.Vector3(),
      cameraRight: new THREE.Vector3(),
      cameraOffset: new THREE.Vector3(),
      upAxis: new THREE.Vector3(0, 1, 0),
      invalidate: vi.fn(),
    }) as SkeletonViewer;
  };

  const cameraPosition = (viewer: SkeletonViewer): THREE.Vector3 =>
    (
      viewer as unknown as {
        camera: THREE.PerspectiveCamera;
      }
    ).camera.position;

  const controlsTarget = (viewer: SkeletonViewer): THREE.Vector3 =>
    (
      viewer as unknown as {
        controls: { target: THREE.Vector3 };
      }
    ).controls.target;

  it("uses the orbit azimuth for W movement even when looking straight down", () => {
    const azimuth = Math.PI / 3;
    const viewer = createViewer(azimuth);
    const before = cameraPosition(viewer).clone();

    SkeletonViewer.prototype.moveCamera.call(viewer, 1, 0);

    const movement = cameraPosition(viewer).clone().sub(before);
    const expectedDirection = new THREE.Vector3(
      -Math.sin(azimuth),
      0,
      -Math.cos(azimuth),
    );
    expect(movement.y).toBeCloseTo(0);
    expect(movement.length()).toBeCloseTo(0.2);
    expect(movement.clone().normalize().dot(expectedDirection)).toBeCloseTo(1);
    expect(controlsTarget(viewer).toArray()).toEqual(movement.toArray());
    expect(
      (
        viewer as unknown as {
          controls: { getAzimuthalAngle: ReturnType<typeof vi.fn> };
        }
      ).controls.getAzimuthalAngle,
    ).toHaveBeenCalledOnce();
  });

  it("normalizes diagonal camera movement to the same distance as W", () => {
    const azimuth = -Math.PI / 5;
    const forwardViewer = createViewer(azimuth);
    const diagonalViewer = createViewer(azimuth);
    const forwardStart = cameraPosition(forwardViewer).clone();
    const diagonalStart = cameraPosition(diagonalViewer).clone();

    SkeletonViewer.prototype.moveCamera.call(forwardViewer, 1, 0);
    SkeletonViewer.prototype.moveCamera.call(diagonalViewer, 1, 1);

    const forwardDistance = cameraPosition(forwardViewer).distanceTo(
      forwardStart,
    );
    const diagonalDistance = cameraPosition(diagonalViewer).distanceTo(
      diagonalStart,
    );
    expect(diagonalDistance).toBeCloseTo(forwardDistance);
  });

  it("integrates held movement independently of the animation frame rate", () => {
    const at60Fps =
      60 * cameraMovementDistance(4, 1 / 60);
    const at30Fps =
      30 * cameraMovementDistance(4, 1 / 30);

    expect(at60Fps).toBeCloseTo(at30Fps);
    expect(cameraMovementDistance(4, 1)).toBeCloseTo(
      cameraMovementDistance(4, 0.05),
    );
  });
});

describe("streaming clip continuity", () => {
  it("preserves secondary motion for a same-rig update at the same playhead", () => {
    expect(
      canPreserveMotionContinuity({
        requested: true,
        hasPreviousClip: true,
        skeletonChanged: false,
        previousFrame: 57.25,
        nextFrame: 57.25,
      }),
    ).toBe(true);
  });

  it("resets secondary motion for a new clip, rig change, or actual seek", () => {
    const base = {
      requested: true,
      hasPreviousClip: true,
      skeletonChanged: false,
      previousFrame: 57.25,
      nextFrame: 57.25,
    };

    expect(
      canPreserveMotionContinuity({
        ...base,
        hasPreviousClip: false,
      }),
    ).toBe(false);
    expect(
      canPreserveMotionContinuity({
        ...base,
        skeletonChanged: true,
      }),
    ).toBe(false);
    expect(
      canPreserveMotionContinuity({
        ...base,
        nextFrame: 58.25,
      }),
    ).toBe(false);
  });

  it("does not reset the VRM spring manager when streaming extends a clip", () => {
    const previousClip = normalizeMotionClip({
      positions: new Float32Array(80 * CORE27_JOINT_COUNT * 3),
      frameCount: 80,
    });
    const extendedClip = normalizeMotionClip({
      positions: new Float32Array(120 * CORE27_JOINT_COUNT * 3),
      frameCount: 120,
    });
    const resetNormalizedPose = vi.fn();
    const resetSpringBones = vi.fn();
    const updatePose = vi.fn();
    const viewer = {
      clip: previousClip,
      skeleton: previousClip.skeleton,
      frameCursor: 57.25,
      playing: true,
      reducedMotion: false,
      hasCameraFollowAnchor: true,
      lastAnimationTime: 1_000,
      lastReportedFrame: 57,
      vrm: {
        humanoid: { resetNormalizedPose },
        springBoneManager: { reset: resetSpringBones },
      },
      rebuildVrmRetargetPlan: vi.fn(),
      createSkeletonMeshes: vi.fn(),
      setOrientationAxes: vi.fn(),
      buildTrajectory: vi.fn(),
      updatePose,
      applyOutputVisibility: vi.fn(),
      resetCamera: vi.fn(),
      invalidate: vi.fn(),
      emitPlaybackState: vi.fn(),
    } as unknown as SkeletonViewer;

    SkeletonViewer.prototype.setMotion.call(viewer, extendedClip, {
      playing: true,
      resetCamera: false,
      preserveContinuity: true,
    });

    expect(resetNormalizedPose).not.toHaveBeenCalled();
    expect(resetSpringBones).not.toHaveBeenCalled();
    expect(updatePose).toHaveBeenCalledWith(57.25);
    expect(
      (viewer as unknown as { hasCameraFollowAnchor: boolean })
        .hasCameraFollowAnchor,
    ).toBe(true);
    expect(
      (viewer as unknown as { lastAnimationTime: number | null })
        .lastAnimationTime,
    ).toBe(1_000);
  });
});

describe("dynamic skeleton layers", () => {
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
});

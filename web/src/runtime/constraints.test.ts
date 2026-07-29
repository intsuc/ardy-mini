// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  buildConstraintWindowBuffers,
  createCapturedConstraint,
  createRootConstraint,
  interpolateRootWaypoints,
  waypointsFromTargetVelocity,
} from "./constraints";
import type { BrowserModelPackManifest } from "./manifest";

function manifest(): BrowserModelPackManifest {
  const motionDim = 12;
  const stats = {
    mean: new Array(motionDim).fill(0),
    std: new Array(motionDim).fill(1),
    normalization_denominator: new Array(motionDim).fill(2),
  };
  return {
    format: "ardy-browser-model-pack",
    schema_version: 1,
    model: { id: "fixture", variant: "constraints" },
    files: {},
    tokenizer: { directory: "tokenizer", max_length: 8 },
    graphs: {} as BrowserModelPackManifest["graphs"],
    dimensions: {
      fps: 20,
      num_frames_per_token: 4,
      max_tokens: 20,
      max_frames: 80,
      constraint_max_tokens: 50,
      constraint_max_frames: 200,
      generation_tokens: 10,
      generation_frames: 40,
      history_tokens: 10,
      history_frames: 40,
      root_features_per_frame: 5,
      nframe_root_dim: 20,
      latent_dim: 128,
      hybrid_dim: 148,
      motion_dim: motionDim,
      body_dim: 7,
      text_condition_dim: 2048,
      num_joints: 2,
    },
    generation: {
      min_frames: 40,
      max_frames: 200,
      default_cfg_weight: 2,
      denoising_steps: 10,
    },
    diffusion: {
      timesteps: [0],
      alphas_cumprod: [1],
      alphas_cumprod_prev: [1],
    },
    recenter: {
      root_mean: [0, 0, 0, 0, 0],
      root_std: [1, 1, 1, 1, 1],
      position_indices: [0, 1, 2],
      heading_indices: [3, 4],
    },
    latent_quantization: {
      levels: new Array(128).fill(4),
      mean: new Array(128).fill(0),
      std: new Array(128).fill(1),
    },
    motion_layout: {
      root_pos: [0, 3],
      global_root_heading: [3, 5],
      local_joints_positions: [5, 8],
      global_rot_data: [8, 12],
    },
    stats: {
      motion: stats,
      global_root: {
        mean: new Array(5).fill(0),
        std: new Array(5).fill(1),
        normalization_denominator: new Array(5).fill(1),
      },
      local_root: stats,
      body: stats,
      post_quantization: stats,
    },
    skeleton: {
      name: "Tiny",
      root_index: 0,
      parents: [-1, 0],
      joint_names: ["Root", "LeftHand"],
      neutral_joints: [
        [0, 0, 0],
        [0, 1, 0],
      ],
    },
  };
}

describe("browser kinematic constraints", () => {
  it("projects world root targets into the recentered ONNX window", () => {
    const pack = manifest();
    const constraint = createRootConstraint(pack, {
      id: "goal",
      frame: 84,
      x: 6,
      z: -2,
      heading: Math.PI / 2,
    });
    const buffers = buildConstraintWindowBuffers(
      pack,
      [constraint],
      0,
      40,
      80,
      Float32Array.of(2, 0, 1),
    );
    const offset = 84 * pack.dimensions.motion_dim;
    expect(buffers.motionMask[offset]).toBe(1);
    expect(buffers.motionMask[offset + 2]).toBe(1);
    // (world 6 - recenter 2) / denominator 2
    expect(buffers.observedMotion[offset]).toBe(2);
    // (world -2 - recenter 1) / denominator 2
    expect(buffers.observedMotion[offset + 2]).toBe(-1.5);
    expect(buffers.futureFrames).toBe(8);
    expect(buffers.appliedConstraintIds).toEqual(["goal"]);
  });

  it("densifies root waypoints and derives heading from target velocity", () => {
    const pack = manifest();
    const dense = interpolateRootWaypoints(
      pack,
      [
        { id: "a", frame: 0, x: 0, z: 0, heading: 0 },
        { id: "b", frame: 4, x: 4, z: 2, heading: Math.PI },
      ],
      true,
    );
    expect(dense).toHaveLength(5);
    expect(dense[2].frame).toBe(2);
    expect(dense[2].values[0]).toBe(1);
    expect(dense[2].values[2]).toBe(0.5);

    const velocity = waypointsFromTargetVelocity({
      startFrame: 20,
      startX: 1,
      startZ: 2,
      velocityX: 2,
      velocityZ: 0,
      fps: 20,
      durationSeconds: 1,
      intervalFrames: 10,
      includeHeading: true,
    });
    expect(velocity.map((waypoint) => waypoint.frame)).toEqual([30, 40]);
    expect(velocity[1].x).toBe(3);
    expect(velocity[1].heading).toBeCloseTo(Math.PI / 2);
  });

  it("interpolates heading over the shortest angular arc", () => {
    const pack = manifest();
    const dense = interpolateRootWaypoints(
      pack,
      [
        { id: "a", frame: 0, x: 0, z: 0, heading: (179 * Math.PI) / 180 },
        { id: "b", frame: 2, x: 0, z: 0, heading: (-179 * Math.PI) / 180 },
      ],
      true,
    );
    const heading = Math.atan2(
      dense[1].values[4] * 2,
      dense[1].values[3] * 2,
    );
    expect(Math.abs(heading)).toBeCloseTo(Math.PI, 5);
  });

  it("smooths dense corners inside the native six-centimetre envelope", () => {
    const pack = manifest();
    const dense = interpolateRootWaypoints(
      pack,
      [
        { id: "a", frame: 0, x: 0, z: 0 },
        { id: "b", frame: 4, x: 4, z: 0 },
        { id: "c", frame: 8, x: 4, z: 4 },
      ],
      true,
    );
    const corner = dense.find((constraint) => constraint.frame === 4);
    expect(corner).toBeDefined();
    const x = corner!.values[0] * 2;
    const z = corner!.values[2] * 2;
    expect(x).toBeLessThan(4);
    expect(z).toBeGreaterThan(0);
    expect(Math.hypot(x - 4, z)).toBeLessThanOrEqual(0.060001);
  });

  it("ramps target velocity from the current root velocity", () => {
    const velocity = waypointsFromTargetVelocity({
      startFrame: 0,
      startX: 0,
      startZ: 0,
      startVelocityX: 0,
      startVelocityZ: 0,
      velocityX: 2,
      velocityZ: 0,
      fps: 20,
      durationSeconds: 2,
      transitionSeconds: 2,
      intervalFrames: 20,
    });
    expect(velocity.map((waypoint) => waypoint.frame)).toEqual([20, 40]);
    expect(velocity[0].x).toBeCloseTo(0.525, 6);
    expect(velocity[1].x).toBeCloseTo(2.05, 6);
  });

  it("captures the complete foot chain and pelvis rotation", () => {
    const pack = manifest();
    pack.dimensions.motion_dim = 30;
    pack.dimensions.body_dim = 25;
    pack.dimensions.num_joints = 3;
    pack.motion_layout = {
      root_pos: [0, 3],
      global_root_heading: [3, 5],
      local_joints_positions: [5, 11],
      global_rot_data: [11, 29],
    };
    pack.stats!.motion = {
      mean: new Array(30).fill(0),
      std: new Array(30).fill(1),
      normalization_denominator: new Array(30).fill(1),
    };
    pack.skeleton = {
      name: "Feet",
      root_index: 0,
      parents: [-1, 0, 1],
      joint_names: ["Hips", "LeftFoot", "LeftToeBase"],
      neutral_joints: [
        [0, 0, 0],
        [0, -1, 0],
        [0, -1, 0.2],
      ],
    };
    const captured = createCapturedConstraint(
      pack,
      "left-foot",
      "left-foot",
      4,
      Float32Array.from({ length: 30 }, (_, index) => index),
      0,
    );

    expect(Array.from(captured.mask.slice(5, 11))).toEqual(
      new Array(6).fill(1),
    );
    expect(Array.from(captured.mask.slice(11, 17))).toEqual(
      new Array(6).fill(1),
    );
    expect(Array.from(captured.mask.slice(17, 23))).toEqual(
      new Array(6).fill(1),
    );
  });

  it("rejects malformed sparse constraints before allocating graph inputs", () => {
    const pack = manifest();
    const malformed = createRootConstraint(pack, {
      id: "bad",
      frame: 10,
      x: 0,
      z: 0,
    });
    malformed.values[0] = Number.NaN;
    expect(() =>
      buildConstraintWindowBuffers(
        pack,
        [malformed],
        0,
        0,
        40,
        new Float32Array(3),
      ),
    ).toThrow(/non-finite/);

    malformed.values[0] = 0;
    malformed.mask.fill(0.25);
    expect(() =>
      buildConstraintWindowBuffers(
        pack,
        [malformed],
        0,
        0,
        40,
        new Float32Array(3),
      ),
    ).toThrow(/observe at least one/);
  });
});

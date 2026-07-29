// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { MotionConstraint } from "./motion-constraint";
import {
  GRAPH_PRECISION_CONTRACT,
  MIXED_PRECISION_FORMAT,
  MIXED_PRECISION_POLICY_VERSION,
  MIXED_PRECISION_PUBLIC_IO_DTYPE,
  REQUIRED_WEBGPU_FEATURE,
  type BrowserGraphPrecisionSummary,
  type BrowserModelPackManifest,
} from "./manifest";
import { postprocessMotion } from "./postprocess";

function precision(): BrowserModelPackManifest["precision"] {
  const summary = <GraphName extends keyof BrowserModelPackManifest["graphs"]>(
    graphName: GraphName,
  ): BrowserGraphPrecisionSummary<GraphName> => {
    const isIdentity =
      GRAPH_PRECISION_CONTRACT[graphName].conversion_mode ===
      "fp32-identity";
    const sourceInitializers = {
      count_by_dtype: { float: 1 },
      bytes_by_dtype: { float: 4 },
      total_count: 1,
      total_bytes: 4,
    };
    return {
      schema_version: 1,
      graph_name: graphName,
      policy_id: GRAPH_PRECISION_CONTRACT[graphName].policy_id,
      conversion_mode: GRAPH_PRECISION_CONTRACT[graphName].conversion_mode,
      source_sha256: isIdentity ? "1".repeat(64) : "0".repeat(64),
      output_sha256: "1".repeat(64),
      source_size_bytes: isIdentity ? 1 : 2,
      output_size_bytes: 1,
      size_reduction_bytes: isIdentity ? 0 : 1,
      size_reduction_fraction: isIdentity ? 0 : 0.5,
      source_initializers: sourceInitializers,
      output_initializers: isIdentity
        ? sourceInitializers
        : {
            count_by_dtype: { float16: 1 },
            bytes_by_dtype: { float16: 2 },
            total_count: 1,
            total_bytes: 2,
          },
    };
  };
  return {
    format: MIXED_PRECISION_FORMAT,
    policy_version: MIXED_PRECISION_POLICY_VERSION,
    public_io_dtype: MIXED_PRECISION_PUBLIC_IO_DTYPE,
    required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
    source_onnx_bytes: 5,
    mixed_onnx_bytes: 3,
    saved_onnx_bytes: 2,
    saved_onnx_fraction: 0.4,
    toolchain: {
      torch: "fixture",
      onnx: "fixture",
      onnxruntime: "fixture",
    },
    graphs: {
      text_encoder: summary("text_encoder"),
      denoiser: summary("denoiser"),
      decoder: summary("decoder"),
    },
  };
}

function manifest(): BrowserModelPackManifest {
  return {
    format: "ardy-browser-model-pack",
    schema_version: 2,
    model: { id: "test", variant: "test" },
    files: {},
    tokenizer: { directory: "tokenizer", max_length: 8 },
    graphs: {} as BrowserModelPackManifest["graphs"],
    precision: precision(),
    dimensions: {
      fps: 20,
      num_frames_per_token: 4,
      max_tokens: 20,
      max_frames: 80,
      generation_tokens: 10,
      generation_frames: 40,
      history_tokens: 10,
      history_frames: 40,
      root_features_per_frame: 5,
      nframe_root_dim: 20,
      latent_dim: 128,
      hybrid_dim: 148,
      motion_dim: 5,
      body_dim: 0,
      text_condition_dim: 1024,
      num_joints: 2,
    },
    generation: {
      min_frames: 1,
      max_frames: 80,
      default_cfg_weight: 2.5,
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
    latent_quantization: { levels: [3], mean: [0], std: [1] },
    motion_layout: {
      root_pos: [0, 3],
      global_root_heading: [3, 5],
    },
    stats: {
      motion: {
        mean: [0, 0, 0, 0, 0],
        std: [1, 1, 1, 1, 1],
        normalization_denominator: [1, 1, 1, 1, 1],
      },
    } as BrowserModelPackManifest["stats"],
    runtime: {
      contract_revision: 3,
      text_only: true,
      required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
    },
  };
}

function rootConstraint(
  frame: number,
  x: number,
  z: number,
  kind: MotionConstraint["kind"] = "root",
): MotionConstraint {
  return {
    id: `root-${frame}`,
    kind,
    frame,
    values: new Float32Array([x, 0, z, 0, 0]),
    mask: new Float32Array([1, 0, 1, 0, 0]),
  };
}

function stationaryMotion(frameCount: number): {
  positions: Float32Array;
  roots: Float32Array;
} {
  const positions = new Float32Array(frameCount * 2 * 3);
  const roots = new Float32Array(frameCount * 3);
  for (let frame = 0; frame < frameCount; frame += 1) {
    positions[(frame * 2 + 1) * 3 + 1] = -1;
  }
  return { positions, roots };
}

describe("postprocessMotion", () => {
  it("smoothly corrects a trajectory through an exact full-body target", () => {
    const source = stationaryMotion(5);
    const result = postprocessMotion(
      {
        ...source,
        roots: { values: source.roots, components: 3 },
        frameCount: 5,
        jointCount: 2,
        constraints: [rootConstraint(2, 2, 0, "full-body")],
        constraintManifest: manifest(),
      },
      {
        constraintBlendFrames: 2,
        footLockStrength: 0,
      },
    );

    expect(Array.from(result.roots ?? [])).toEqual([
      0, 0, 0,
      1, 0, 0,
      2, 0, 0,
      1, 0, 0,
      0, 0, 0,
    ]);
    expect(result.positions[(2 * 2 + 1) * 3]).toBe(2);
    expect(result.metrics.rootConstraintFrames).toBe(1);
    expect(result.metrics.rootConstraintMeanErrorBefore).toBe(2);
    expect(result.metrics.rootConstraintMeanErrorAfter).toBe(0);
    expect(source.roots.every((value) => value === 0)).toBe(true);
    expect(source.positions[(2 * 2 + 1) * 3]).toBe(0);
  });

  it("honors the root margin for non-full-body targets", () => {
    const source = stationaryMotion(3);
    const result = postprocessMotion(
      {
        positions: source.positions,
        roots: { values: source.roots, components: 3 },
        frameCount: 3,
        jointCount: 2,
        constraints: [rootConstraint(1, 1, 0)],
        constraintManifest: manifest(),
      },
      {
        rootMargin: 0.1,
        constraintBlendFrames: 0,
        footLockStrength: 0,
      },
    );

    expect(result.roots?.[3]).toBeCloseTo(0.9, 6);
    expect(result.metrics.rootConstraintMeanErrorAfter).toBeCloseTo(0.1, 6);
  });

  it("locks contacted feet and reduces horizontal skating", () => {
    const frameCount = 5;
    const source = stationaryMotion(frameCount);
    for (let frame = 0; frame < frameCount; frame += 1) {
      source.positions[(frame * 2 + 1) * 3] = frame * 0.1;
    }
    const result = postprocessMotion(
      {
        positions: source.positions,
        roots: { values: source.roots, components: 3 },
        frameCount,
        jointCount: 2,
        contacts: {
          values: new Float32Array(frameCount).fill(1),
          channels: 1,
          jointIndices: [1],
        },
      },
      {
        footLockStrength: 1,
        footLockRampFrames: 1,
        maxFootCorrectionPerFrame: 1,
      },
    );

    expect(result.metrics.contactSamples).toBe(4);
    expect(result.metrics.footSlidingDistanceBefore).toBeCloseTo(0.4, 6);
    expect(result.metrics.footSlidingDistanceAfter).toBeCloseTo(0, 6);
    for (let frame = 0; frame < frameCount; frame += 1) {
      expect(result.positions[(frame * 2 + 1) * 3]).toBeCloseTo(0, 6);
    }
  });

  it("does not let foot locking move an exact constrained root", () => {
    const frameCount = 4;
    const source = stationaryMotion(frameCount);
    for (let frame = 0; frame < frameCount; frame += 1) {
      source.positions[(frame * 2 + 1) * 3] = frame;
    }
    const result = postprocessMotion(
      {
        positions: source.positions,
        roots: { values: source.roots, components: 3 },
        frameCount,
        jointCount: 2,
        constraints: [rootConstraint(2, 2, 0, "full-body")],
        constraintManifest: manifest(),
        contacts: {
          values: new Uint8Array(frameCount).fill(1),
          channels: 1,
          jointIndices: [1],
        },
      },
      {
        constraintBlendFrames: 0,
        footLockStrength: 1,
        footLockRampFrames: 1,
        maxFootCorrectionPerFrame: 10,
      },
    );

    expect(result.roots?.[2 * 3]).toBeCloseTo(2, 6);
    expect(result.metrics.rootConstraintMeanErrorAfter).toBeCloseTo(0, 6);
  });

  it("rejects unsafe data instead of producing NaN", () => {
    const source = stationaryMotion(2);
    source.positions[4] = Number.NaN;
    expect(() =>
      postprocessMotion({
        positions: source.positions,
        frameCount: 2,
        jointCount: 2,
      }),
    ).toThrow(/non-finite/);

    expect(() =>
      postprocessMotion(
        {
          positions: new Float32Array(12),
          frameCount: 2,
          jointCount: 2,
        },
        { contactThreshold: Number.NaN },
      ),
    ).toThrow(/Contact threshold/);
  });

  it("is deterministic and ignores constraints outside the current chunk", () => {
    const source = stationaryMotion(3);
    const input = {
      positions: source.positions,
      frameCount: 3,
      jointCount: 2,
      frameOffset: 40,
      constraints: [rootConstraint(10, 100, 100)],
      constraintManifest: manifest(),
    } as const;
    const first = postprocessMotion(input);
    const second = postprocessMotion(input);
    expect(first.positions).toEqual(second.positions);
    expect(first.rootTranslations).toEqual(new Float32Array(9));
    expect(first.metrics.rootConstraintFrames).toBe(0);
  });
});

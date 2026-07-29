// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { BrowserDimensions } from "./manifest";
import { PortableRandom } from "./random";
import {
  copyTailHistory,
  createArWindow,
  createConditionedArWindow,
  createMotionPadMask,
  decoderValidTokensForFrames,
  recenterAndRequantize,
  roundTiesToEven,
} from "./windows";

const dimensions: BrowserDimensions = {
  fps: 20,
  num_frames_per_token: 1,
  max_tokens: 4,
  max_frames: 4,
  generation_tokens: 2,
  generation_frames: 2,
  history_tokens: 2,
  history_frames: 2,
  root_features_per_frame: 5,
  nframe_root_dim: 5,
  latent_dim: 2,
  hybrid_dim: 7,
  motion_dim: 6,
  body_dim: 1,
  text_condition_dim: 8,
  num_joints: 2,
};

describe("AR window construction", () => {
  it("places an initial generation at the start of the fixed window", () => {
    const window = createArWindow(dimensions, new PortableRandom(7));
    expect(window.historyFrames).toBe(0);
    expect([...window.historyMask]).toEqual([0, 0, 0, 0]);
    expect([...window.generationMask]).toEqual([1, 1, 0, 0]);
    expect([...window.generationTokenMask]).toEqual([1, 1, 0, 0]);
    expect([...window.x.slice(14)]).toEqual(new Array(14).fill(0));
  });

  it("places continuation history before fresh generation noise", () => {
    const history = new Float32Array(14).fill(0.25);
    const window = createArWindow(dimensions, new PortableRandom(7), history);
    expect(window.x.slice(0, 14)).toEqual(history);
    expect([...window.historyMask]).toEqual([1, 1, 0, 0]);
    expect([...window.generationMask]).toEqual([0, 0, 1, 1]);
    expect([...window.historyTokenMask]).toEqual([1, 1, 0, 0]);
    expect([...window.generationTokenMask]).toEqual([0, 0, 1, 1]);
    expect(createMotionPadMask(dimensions, 4)).toEqual(
      new Float32Array([1, 1, 1, 1]),
    );
  });

  it("rounds a partial final decoder window to whole tokens", () => {
    const core40 = {
      ...dimensions,
      num_frames_per_token: 4,
      max_tokens: 20,
      max_frames: 80,
      generation_tokens: 10,
      generation_frames: 40,
      history_tokens: 10,
      history_frames: 40,
    };
    expect(decoderValidTokensForFrames(core40, 10, 1)).toBe(11);
    const mask = createMotionPadMask(core40, 11);
    expect(mask.slice(0, 44)).toEqual(new Float32Array(44).fill(1));
    expect(mask.slice(44)).toEqual(new Float32Array(36));
  });

  it("marks only future tokens that contain a sparse observation", () => {
    const conditionedDimensions: BrowserDimensions = {
      ...dimensions,
      constraint_max_tokens: 6,
      constraint_max_frames: 6,
    };
    const motionMask = new Float32Array(6 * dimensions.motion_dim);
    const observedMotion = new Float32Array(motionMask.length);
    // Two history + two generation frames put future at indices 4 and 5.
    motionMask[5 * dimensions.motion_dim + 2] = 1;
    observedMotion[5 * dimensions.motion_dim + 2] = 0.25;
    const window = createConditionedArWindow(
      conditionedDimensions,
      new PortableRandom(7),
      new Float32Array(2 * dimensions.hybrid_dim),
      2,
      motionMask,
      observedMotion,
    );
    expect([...window.futureMask]).toEqual([0, 0, 0, 0, 1, 1]);
    expect([...window.futureTokenMask]).toEqual([0, 0, 0, 0, 0, 1]);
    expect(window.observedMotion).toBe(observedMotion);
  });
});

describe("recenter and FSQ", () => {
  it("matches ARDY translation, heading, ties-to-even, and tail cropping", () => {
    const hybrid = new Float32Array([
      0, 0, 1, 1, 0, 0.25, 0.75,
      1, 0, 2, 1, 0, -0.25, -0.75,
      2, 0, 3, 0, 1, 0.25, 0.75,
      3, 0, 4, 1, 0, -0.25, -0.75,
    ]);
    const result = recenterAndRequantize(
      hybrid,
      4,
      dimensions,
      {
        root_mean: [0, 0, 0, 0, 0],
        root_std: [1, 1, 1, 1, 1],
        position_indices: [0, 1, 2],
        heading_indices: [3, 4],
      },
      {
        levels: [4, 4],
        mean: [0, 0],
        std: [1, 1],
      },
      new Float32Array([10, 0, 20]),
      2,
    );
    expect([...result.globalTranslation]).toEqual([13, 0, 24]);
    expect(result.firstHeadingAngle).toBeCloseTo(Math.PI / 2);
    expect(hybrid[0]).toBe(-3);
    expect(hybrid[2]).toBe(-3);
    expect(hybrid[21]).toBe(0);
    expect(hybrid[23]).toBe(0);
    expect(hybrid[19]).toBe(0);
    expect(hybrid[20]).toBe(1);
    expect(copyTailHistory(hybrid, 4, dimensions)).toEqual(
      hybrid.slice(14, 28),
    );
  });

  it("uses banker's rounding like torch.round", () => {
    expect(roundTiesToEven(0.5)).toBe(0);
    expect(roundTiesToEven(1.5)).toBe(2);
    expect(roundTiesToEven(-0.5)).toBe(0);
    expect(roundTiesToEven(-1.5)).toBe(-2);
  });
});

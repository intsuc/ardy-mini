// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { applyDdimStepInPlace, ddimStepForIndex } from "./ddim";

describe("DDIM", () => {
  it("looks alpha values up by descending model timestep", () => {
    expect(
      ddimStepForIndex(
        [2, 1, 0],
        [0.9, 0.6, 0.25],
        [1, 0.9, 0.6],
        0,
      ),
    ).toEqual({ timestep: 2, alpha: 0.25, alphaPrevious: 0.6 });
  });

  it("updates only the requested generation range", () => {
    const sample = new Float32Array([11, 4, 4, 12]);
    const predicted = new Float32Array([0, 2, 3, 0]);
    applyDdimStepInPlace(
      sample,
      predicted,
      { alpha: 0.25, alphaPrevious: 1 },
      1,
      3,
    );
    expect(sample[0]).toBe(11);
    expect(sample[1]).toBe(2);
    expect(sample[2]).toBe(3);
    expect(sample[3]).toBe(12);
  });
});

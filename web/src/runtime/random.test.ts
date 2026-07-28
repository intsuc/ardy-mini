// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { PortableRandom, seedToUint32 } from "./random";

describe("PortableRandom", () => {
  it("has a stable uint32 stream", () => {
    const random = new PortableRandom(1);
    expect(Array.from({ length: 5 }, () => random.nextUint32())).toEqual([
      2693262067,
      11749833,
      2265367787,
      4213581821,
      4159151403,
    ]);
  });

  it("maps string seeds deterministically and fills finite float32 noise", () => {
    expect(seedToUint32("walk forward")).toBe(seedToUint32("walk forward"));
    expect(seedToUint32("walk forward")).not.toBe(seedToUint32("walk backward"));
    const left = new Float32Array(32);
    const right = new Float32Array(32);
    new PortableRandom("same").fillNormal(left);
    new PortableRandom("same").fillNormal(right);
    expect(left).toEqual(right);
    expect([...left].every(Number.isFinite)).toBe(true);
    expect(new Set(left).size).toBeGreaterThan(20);
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

const UINT32_RANGE = 0x1_0000_0000;

export interface PortableRandomState {
  seed: number;
  state: number;
  spareNormal?: number;
}

export function seedToUint32(seed: number | string): number {
  if (typeof seed === "number") {
    return Math.trunc(seed) >>> 0;
  }
  const bytes = new TextEncoder().encode(seed);
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/**
 * Small reproducible generator used instead of browser- or ORT-owned randomness.
 *
 * Uniform values use Mulberry32; Gaussian values use the Marsaglia polar method.
 * All generated tensors are explicitly rounded to float32 at their write boundary.
 */
export class PortableRandom {
  readonly seed: number;
  #state: number;
  #spareNormal: number | undefined;

  constructor(seed: number | string) {
    this.seed = seedToUint32(seed);
    this.#state = this.seed;
  }

  snapshot(): PortableRandomState {
    return {
      seed: this.seed,
      state: this.#state,
      ...(this.#spareNormal === undefined
        ? {}
        : { spareNormal: this.#spareNormal }),
    };
  }

  restore(snapshot: PortableRandomState): void {
    if (
      snapshot.seed !== this.seed ||
      !Number.isSafeInteger(snapshot.state) ||
      snapshot.state < 0 ||
      snapshot.state >= UINT32_RANGE ||
      (snapshot.spareNormal !== undefined &&
        !Number.isFinite(snapshot.spareNormal))
    ) {
      throw new TypeError("Portable random snapshot is invalid");
    }
    this.#state = snapshot.state >>> 0;
    this.#spareNormal = snapshot.spareNormal;
  }

  nextUint32(): number {
    this.#state = (this.#state + 0x6d2b79f5) >>> 0;
    let value = this.#state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  }

  nextFloat(): number {
    return (this.nextUint32() + 1) / (UINT32_RANGE + 1);
  }

  nextNormal(): number {
    if (this.#spareNormal !== undefined) {
      const result = this.#spareNormal;
      this.#spareNormal = undefined;
      return result;
    }
    let x = 0;
    let y = 0;
    let radiusSquared = 0;
    do {
      x = 2 * this.nextFloat() - 1;
      y = 2 * this.nextFloat() - 1;
      radiusSquared = x * x + y * y;
    } while (radiusSquared <= Number.EPSILON || radiusSquared >= 1);
    const scale = Math.sqrt((-2 * Math.log(radiusSquared)) / radiusSquared);
    this.#spareNormal = y * scale;
    return x * scale;
  }

  fillNormal(target: Float32Array, start = 0, end = target.length): void {
    if (
      !Number.isSafeInteger(start) ||
      !Number.isSafeInteger(end) ||
      start < 0 ||
      end < start ||
      end > target.length
    ) {
      throw new RangeError("Invalid Gaussian fill range");
    }
    for (let index = start; index < end; index += 1) {
      target[index] = Math.fround(this.nextNormal());
    }
  }
}

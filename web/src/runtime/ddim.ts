// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export interface DdimStep {
  timestep: number;
  alpha: number;
  alphaPrevious: number;
}

export function ddimStepForIndex(
  timesteps: readonly number[],
  alphasCumprod: readonly number[],
  alphasCumprodPrevious: readonly number[],
  inferenceIndex: number,
): DdimStep {
  if (
    !Number.isSafeInteger(inferenceIndex) ||
    inferenceIndex < 0 ||
    inferenceIndex >= timesteps.length
  ) {
    throw new RangeError("DDIM inference index is out of range");
  }
  const timestep = timesteps[inferenceIndex];
  const alpha = alphasCumprod[timestep];
  const alphaPrevious = alphasCumprodPrevious[timestep];
  if (
    !Number.isFinite(alpha) ||
    !Number.isFinite(alphaPrevious) ||
    alpha <= 0 ||
    alpha > 1 ||
    alphaPrevious <= 0 ||
    alphaPrevious > 1
  ) {
    throw new RangeError(`Invalid DDIM alpha at timestep ${timestep}`);
  }
  return { timestep, alpha, alphaPrevious };
}

/**
 * Deterministic DDIM update (eta=0), matching ARDY's Diffusion.ddim_sample.
 * History/future slots remain byte-for-byte unchanged.
 */
export function applyDdimStepInPlace(
  sample: Float32Array,
  predictedX0: Float32Array,
  step: Pick<DdimStep, "alpha" | "alphaPrevious">,
  start = 0,
  end = sample.length,
): void {
  if (sample.length !== predictedX0.length) {
    throw new RangeError("DDIM sample and prediction lengths differ");
  }
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    end > sample.length
  ) {
    throw new RangeError("Invalid DDIM update range");
  }
  const sqrtAlpha = Math.sqrt(step.alpha);
  const sqrtReciprocalMinusOne = Math.sqrt(1 / step.alpha - 1);
  const sqrtPrevious = Math.sqrt(step.alphaPrevious);
  const sqrtOneMinusPrevious = Math.sqrt(1 - step.alphaPrevious);

  for (let index = start; index < end; index += 1) {
    const predicted = predictedX0[index];
    const epsilon =
      sqrtReciprocalMinusOne === 0
        ? 0
        : (sample[index] / sqrtAlpha - predicted) / sqrtReciprocalMinusOne;
    sample[index] = Math.fround(
      predicted * sqrtPrevious + epsilon * sqrtOneMinusPrevious,
    );
  }
}

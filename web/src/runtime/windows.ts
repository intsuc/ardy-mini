// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type {
  BrowserDimensions,
  BrowserLatentQuantization,
  BrowserRecenterConfig,
} from "./manifest";
import { PortableRandom } from "./random";

export interface ArWindow {
  x: Float32Array;
  historyMask: Float32Array;
  generationMask: Float32Array;
  historyTokenMask: Float32Array;
  generationTokenMask: Float32Array;
  historyFrames: number;
  generationFrames: number;
  historyTokens: number;
  generationTokens: number;
  generationTokenOffset: number;
}

function assertHistory(
  history: Float32Array | undefined,
  dimensions: BrowserDimensions,
): number {
  if (history === undefined) {
    return 0;
  }
  if (history.length % dimensions.hybrid_dim !== 0) {
    throw new RangeError("History length is not divisible by hybrid_dim");
  }
  const tokens = history.length / dimensions.hybrid_dim;
  if (tokens > dimensions.history_tokens) {
    throw new RangeError(
      `Continuation history cannot exceed ${dimensions.history_tokens} tokens`,
    );
  }
  return tokens;
}

export function createArWindow(
  dimensions: BrowserDimensions,
  random: PortableRandom,
  history?: Float32Array,
): ArWindow {
  const historyTokens = assertHistory(history, dimensions);
  const historyFrames = historyTokens * dimensions.num_frames_per_token;
  const generationTokens = dimensions.generation_tokens;
  const generationFrames = dimensions.generation_frames;
  const generationTokenOffset = historyTokens;
  if (generationTokenOffset + generationTokens > dimensions.max_tokens) {
    throw new RangeError("History and generation do not fit in the AR window");
  }

  const x = new Float32Array(dimensions.max_tokens * dimensions.hybrid_dim);
  if (history !== undefined) {
    x.set(history);
  }
  const noiseStart = generationTokenOffset * dimensions.hybrid_dim;
  const noiseEnd = (generationTokenOffset + generationTokens) * dimensions.hybrid_dim;
  random.fillNormal(x, noiseStart, noiseEnd);

  const historyMask = new Float32Array(dimensions.max_frames);
  const generationMask = new Float32Array(dimensions.max_frames);
  const historyTokenMask = new Float32Array(dimensions.max_tokens);
  const generationTokenMask = new Float32Array(dimensions.max_tokens);
  historyMask.fill(1, 0, historyFrames);
  generationMask.fill(1, historyFrames, historyFrames + generationFrames);
  historyTokenMask.fill(1, 0, historyTokens);
  generationTokenMask.fill(
    1,
    generationTokenOffset,
    generationTokenOffset + generationTokens,
  );

  return {
    x,
    historyMask,
    generationMask,
    historyTokenMask,
    generationTokenMask,
    historyFrames,
    generationFrames,
    historyTokens,
    generationTokens,
    generationTokenOffset,
  };
}

export function createMotionPadMask(
  dimensions: BrowserDimensions,
  validTokens: number,
): Float32Array {
  if (
    !Number.isSafeInteger(validTokens) ||
    validTokens <= 0 ||
    validTokens > dimensions.max_tokens
  ) {
    throw new RangeError("Invalid valid-token count");
  }
  const result = new Float32Array(dimensions.max_frames);
  result.fill(1, 0, validTokens * dimensions.num_frames_per_token);
  return result;
}

export function decoderValidTokensForFrames(
  dimensions: BrowserDimensions,
  historyTokens: number,
  visibleGenerationFrames: number,
): number {
  if (
    !Number.isSafeInteger(historyTokens) ||
    historyTokens < 0 ||
    historyTokens > dimensions.history_tokens ||
    !Number.isSafeInteger(visibleGenerationFrames) ||
    visibleGenerationFrames <= 0 ||
    visibleGenerationFrames > dimensions.generation_frames
  ) {
    throw new RangeError("Invalid decoder history/generation bounds");
  }
  const generationTokens = Math.ceil(
    visibleGenerationFrames / dimensions.num_frames_per_token,
  );
  const validTokens = historyTokens + generationTokens;
  if (validTokens > dimensions.max_tokens) {
    throw new RangeError("Visible decoder tokens exceed the fixed window");
  }
  return validTokens;
}

/** ECMAScript's Math.round is not PyTorch's ties-to-even rounding. */
export function roundTiesToEven(value: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) {
    return floor;
  }
  if (fraction > 0.5) {
    return floor + 1;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

export interface RecenterResult {
  globalTranslation: Float32Array;
  firstHeadingAngle: number;
}

/**
 * Recenter normalized root x/z around the final valid frame and requantize FSQ
 * body latents. Returns the world-space offset and heading for the next window.
 */
export function recenterAndRequantize(
  hybrid: Float32Array,
  validTokens: number,
  dimensions: BrowserDimensions,
  recenter: BrowserRecenterConfig,
  quantization: BrowserLatentQuantization,
  globalTranslation: Float32Array,
  nextHistoryStartToken: number,
): RecenterResult {
  if (hybrid.length !== dimensions.max_tokens * dimensions.hybrid_dim) {
    throw new RangeError("Hybrid window has the wrong element count");
  }
  if (
    !Number.isSafeInteger(validTokens) ||
    validTokens <= 0 ||
    validTokens > dimensions.max_tokens ||
    !Number.isSafeInteger(nextHistoryStartToken) ||
    nextHistoryStartToken < 0 ||
    nextHistoryStartToken >= validTokens
  ) {
    throw new RangeError("Invalid recenter window bounds");
  }
  if (globalTranslation.length !== 3) {
    throw new RangeError("Global translation must contain three values");
  }

  const {
    hybrid_dim: hybridDim,
    nframe_root_dim: rootTokenDim,
    root_features_per_frame: rootFrameDim,
    num_frames_per_token: framesPerToken,
  } = dimensions;
  const validFrames = validTokens * framesPerToken;
  const [positionX, positionY, positionZ] = recenter.position_indices;
  const [headingCos, headingSin] = recenter.heading_indices;

  const rootOffset = (frame: number, feature: number): number => {
    const token = Math.floor(frame / framesPerToken);
    const frameInToken = frame % framesPerToken;
    return token * hybridDim + frameInToken * rootFrameDim + feature;
  };
  const rawRoot = (frame: number, feature: number): number => {
    const normalized = hybrid[rootOffset(frame, feature)];
    return normalized * recenter.root_std[feature] + recenter.root_mean[feature];
  };

  const centerFrame = validFrames - 1;
  const center = new Float32Array([
    rawRoot(centerFrame, positionX),
    0,
    rawRoot(centerFrame, positionZ),
  ]);
  for (let frame = 0; frame < validFrames; frame += 1) {
    for (const [feature, centerValue] of [
      [positionX, center[0]],
      [positionZ, center[2]],
    ] as const) {
      const translated = rawRoot(frame, feature) - centerValue;
      hybrid[rootOffset(frame, feature)] = Math.fround(
        (translated - recenter.root_mean[feature]) / recenter.root_std[feature],
      );
    }
  }

  const latentOffset = rootTokenDim;
  for (let token = 0; token < validTokens; token += 1) {
    const tokenOffset = token * hybridDim;
    for (let feature = 0; feature < dimensions.latent_dim; feature += 1) {
      const index = tokenOffset + latentOffset + feature;
      const raw =
        hybrid[index] * quantization.std[feature] + quantization.mean[feature];
      const halfWidth = Math.floor(quantization.levels[feature] / 2);
      const clamped = Math.max(-1, Math.min(1, raw));
      const discrete = roundTiesToEven(clamped * halfWidth) / halfWidth;
      hybrid[index] = Math.fround(
        (discrete - quantization.mean[feature]) / quantization.std[feature],
      );
    }
  }

  const firstHistoryFrame = nextHistoryStartToken * framesPerToken;
  const rawCos = rawRoot(firstHistoryFrame, headingCos);
  const rawSin = rawRoot(firstHistoryFrame, headingSin);
  return {
    globalTranslation: new Float32Array([
      Math.fround(globalTranslation[0] + center[0]),
      Math.fround(globalTranslation[1] + center[1]),
      Math.fround(globalTranslation[2] + center[2]),
    ]),
    firstHeadingAngle: Math.atan2(rawSin, rawCos),
  };
}

export function copyTailHistory(
  hybrid: Float32Array,
  validTokens: number,
  dimensions: BrowserDimensions,
  requestedTokens = dimensions.history_tokens,
): Float32Array {
  const historyTokens = requestedTokens;
  if (
    !Number.isSafeInteger(historyTokens) ||
    historyTokens <= 0 ||
    historyTokens > dimensions.history_tokens
  ) {
    throw new RangeError("Invalid requested history-token count");
  }
  if (validTokens < historyTokens || validTokens > dimensions.max_tokens) {
    throw new RangeError("Cannot crop requested history from hybrid window");
  }
  const start = (validTokens - historyTokens) * dimensions.hybrid_dim;
  const end = validTokens * dimensions.hybrid_dim;
  return hybrid.slice(start, end);
}

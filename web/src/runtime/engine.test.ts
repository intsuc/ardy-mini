// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";

import {
  BrowserArdyGenerationSession,
  type RuntimeContinuationState,
} from "./engine";
import type { BrowserModelPackManifest } from "./manifest";
import { ort, type RuntimeSessions } from "./sessions";
import type { LocalTokenizer } from "./tokenizer";

function manifest(): BrowserModelPackManifest {
  return {
    format: "ardy-browser-model-pack",
    schema_version: 2,
    model: { id: "fixture", variant: "session" },
    files: {},
    tokenizer: { directory: "tokenizer", max_length: 8 },
    graphs: {} as BrowserModelPackManifest["graphs"],
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
      motion_dim: 330,
      body_dim: 325,
      text_condition_dim: 2048,
      num_joints: 27,
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
    runtime: { contract_revision: 3, text_only: true },
  };
}

function session(): BrowserArdyGenerationSession {
  return new BrowserArdyGenerationSession(
    manifest(),
    {} as LocalTokenizer,
    {} as RuntimeSessions,
    {
      seed: 7,
      initialTranslation: Float32Array.of(1, 0, -2),
      initialHeading: 0.5,
    },
  );
}

function continuation(): RuntimeContinuationState {
  return {
    frameCount: 12,
    hybridTokens: Float32Array.from(
      { length: 3 * 148 },
      (_, index) => index / 100,
    ),
    hybridDim: 148,
    random: { seed: 7, state: 123, spareNormal: 0.25 },
    initialTranslation: [1, 0, -2],
    initialHeading: 0.5,
  };
}

describe("stateful browser generation session", () => {
  it("round-trips portable continuation state without sharing buffers", () => {
    const runtimeSession = session();
    const source = continuation();
    runtimeSession.restore(source);
    const restored = runtimeSession.continuation();
    expect(restored).toEqual(source);
    expect(restored.hybridTokens).not.toBe(source.hybridTokens);
    source.hybridTokens[0] = 999;
    expect(runtimeSession.continuation().hybridTokens[0]).toBe(0);
  });

  it("branches at the preceding complete four-frame token", () => {
    const runtimeSession = session();
    runtimeSession.restore(continuation());
    expect(runtimeSession.branch(11)).toBe(8);
    expect(runtimeSession.frameCount).toBe(8);
    expect(runtimeSession.continuation().hybridTokens).toHaveLength(2 * 148);
  });

  it("rejects incompatible or non-finite continuation buffers", () => {
    const runtimeSession = session();
    expect(() =>
      runtimeSession.restore({
        ...continuation(),
        hybridDim: 149,
      }),
    ).toThrow(/does not match/);
    const invalid = continuation();
    invalid.hybridTokens[3] = Number.NaN;
    expect(() => runtimeSession.restore(invalid)).toThrow(/non-finite/);

    expect(() =>
      runtimeSession.restore({
        ...continuation(),
        frameCount: 8,
      }),
    ).toThrow(/does not match/);
    expect(() =>
      runtimeSession.restore({
        ...continuation(),
        frameCount: 0,
      }),
    ).toThrow(/does not match/);
    expect(
      () =>
        new BrowserArdyGenerationSession(
          manifest(),
          {} as LocalTokenizer,
          {} as RuntimeSessions,
          {
            seed: 1,
            initialTranslation: [Number.MAX_VALUE, 0, 0],
          },
        ),
    ).toThrow(/float32/);
  });
});

function generationManifest(): BrowserModelPackManifest {
  return {
    format: "ardy-browser-model-pack",
    schema_version: 2,
    model: { id: "fixture", variant: "generation" },
    files: {},
    tokenizer: { directory: "tokenizer", max_length: 8 },
    graphs: {
      text_encoder: {
        model: "text.onnx",
        inputs: {
          inputIds: "input_ids",
          attentionMask: "attention_mask",
          tokenTypeIds: "token_type_ids",
        },
        outputs: { textConditions: "text_conditions" },
      },
      denoiser: {
        model: "denoiser.onnx",
        inputs: {
          cfgWeight: "cfg_weight",
          x: "x",
          historyLength: "history_len",
          generationLength: "generation_len",
          historyMask: "history_mask",
          generationMask: "generation_mask",
          historyTokenMask: "history_token_mask",
          generationTokenMask: "generation_token_mask",
          textConditions: "text_conditions",
          timestep: "timestep",
          firstHeadingAngle: "first_heading_angle",
        },
        outputs: { predX0: "pred_x0" },
      },
      decoder: {
        model: "decoder.onnx",
        inputs: {
          hybridTokens: "hybrid_tokens",
          motionPadMask: "motion_pad_mask",
          globalTranslation: "global_translation",
        },
        outputs: {
          normalizedMotion: "normalized_motion",
          posedJoints: "posed_joints",
          localRotations: "local_rotations",
          globalRotations: "global_rotations",
          rootPositions: "root_positions",
          footContacts: "foot_contacts",
          globalRootHeading: "global_root_heading",
        },
      },
    },
    dimensions: {
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
      latent_dim: 1,
      hybrid_dim: 6,
      motion_dim: 6,
      body_dim: 1,
      text_condition_dim: 1,
      num_joints: 1,
    },
    generation: {
      min_frames: 1,
      max_frames: 4,
      default_cfg_weight: 2,
      denoising_steps: 1,
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
      levels: [3],
      mean: [0],
      std: [1],
    },
    runtime: { contract_revision: 3, text_only: true },
  };
}

describe("stateful generation coordinates", () => {
  it("preserves an external y translation after history is reconstructed", async () => {
    const pack = generationManifest();
    const decoderTranslations: number[][] = [];
    const denoiserWindows: Float32Array[] = [];
    const tokenizer = {
      encode: vi.fn(async () => ({
        inputIds: BigInt64Array.of(1n),
        attentionMask: BigInt64Array.of(1n),
        tokenTypeIds: BigInt64Array.of(0n),
        sequenceLength: 1,
      })),
    } as unknown as LocalTokenizer;
    const sessions = {
      textEncoder: {
        run: vi.fn(async () => ({
          text_conditions: new ort.Tensor(
            "float32",
            Float32Array.of(0),
            [1, 1, 1],
          ),
        })),
      },
      denoiser: {
        run: vi.fn(
          async (feeds: Record<string, ort.Tensor>) => {
            denoiserWindows.push(
              new Float32Array(feeds.x.data as Float32Array),
            );
            return {
              pred_x0: new ort.Tensor(
                "float32",
                new Float32Array(
                  pack.dimensions.max_tokens * pack.dimensions.hybrid_dim,
                ),
                [1, pack.dimensions.max_tokens, pack.dimensions.hybrid_dim],
              ),
            };
          },
        ),
      },
      decoder: {
        run: vi.fn(
          async (feeds: Record<string, ort.Tensor>) => {
            const translation = Array.from(
              feeds.global_translation.data as Float32Array,
            );
            decoderTranslations.push(translation);
            const motion = new Float32Array(
              pack.dimensions.max_frames * pack.dimensions.motion_dim,
            );
            const joints = new Float32Array(
              pack.dimensions.max_frames * pack.dimensions.num_joints * 3,
            );
            const rotations = new Float32Array(
              pack.dimensions.max_frames * pack.dimensions.num_joints * 9,
            );
            const rootPositions = new Float32Array(
              pack.dimensions.max_frames * 3,
            );
            for (let frame = 0; frame < pack.dimensions.max_frames; frame += 1) {
              const motionOffset = frame * pack.dimensions.motion_dim;
              motion.set(
                [
                  translation[0] + frame,
                  translation[1],
                  translation[2],
                  1,
                  0,
                ],
                motionOffset,
              );
              joints.set(
                [translation[0] + frame, translation[1], translation[2]],
                frame * 3,
              );
              rootPositions.set(
                [translation[0] + frame, translation[1], translation[2]],
                frame * 3,
              );
            }
            return {
              normalized_motion: new ort.Tensor(
                "float32",
                motion,
                [1, pack.dimensions.max_frames, pack.dimensions.motion_dim],
              ),
              posed_joints: new ort.Tensor(
                "float32",
                joints,
                [
                  1,
                  pack.dimensions.max_frames,
                  pack.dimensions.num_joints,
                  3,
                ],
              ),
              local_rotations: new ort.Tensor(
                "float32",
                rotations,
                [
                  1,
                  pack.dimensions.max_frames,
                  pack.dimensions.num_joints,
                  3,
                  3,
                ],
              ),
              global_rotations: new ort.Tensor(
                "float32",
                rotations,
                [
                  1,
                  pack.dimensions.max_frames,
                  pack.dimensions.num_joints,
                  3,
                  3,
                ],
              ),
              root_positions: new ort.Tensor(
                "float32",
                rootPositions,
                [1, pack.dimensions.max_frames, 3],
              ),
              foot_contacts: new ort.Tensor(
                "bool",
                new Uint8Array(pack.dimensions.max_frames * 4),
                [1, pack.dimensions.max_frames, 4],
              ),
              global_root_heading: new ort.Tensor(
                "float32",
                new Float32Array(pack.dimensions.max_frames * 2),
                [1, pack.dimensions.max_frames, 2],
              ),
            };
          },
        ),
      },
    } as unknown as RuntimeSessions;
    const runtimeSession = new BrowserArdyGenerationSession(
      pack,
      tokenizer,
      sessions,
      {
        seed: 7,
        initialTranslation: [1, 2.5, -3],
      },
    );

    await runtimeSession.generate({ prompt: "walk", durationFrames: 2 });
    await runtimeSession.generate({ prompt: "walk", durationFrames: 2 });
    await runtimeSession.generate({
      prompt: "walk",
      durationFrames: 2,
      historyFrames: 0,
    });

    expect(decoderTranslations).toHaveLength(3);
    expect(decoderTranslations[0]).toEqual([1, 2.5, -3]);
    expect(decoderTranslations[1]).toEqual([2, 2.5, -3]);
    expect(decoderTranslations[2]).toEqual([5, 2.5, -3]);
    // The first two tokens in the second denoiser window are continuation
    // history. Its root y must be model-local (0), not the stored world y
    // (2.5), because the decoder applies the external y translation again.
    expect(denoiserWindows[1][1]).toBe(0);
    expect(
      denoiserWindows[1][pack.dimensions.hybrid_dim + 1],
    ).toBe(0);
  });
});

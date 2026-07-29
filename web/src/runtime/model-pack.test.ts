// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { sha256Hex } from "./hash";
import {
  MODEL_PACK_FORMAT,
  MODEL_PACK_SCHEMA_VERSION,
  type BrowserModelPackManifest,
  validateModelPackManifest,
} from "./manifest";
import { loadModelPackFromMemory } from "./model-pack";

const encoder = new TextEncoder();

async function fixture(): Promise<{
  entries: Map<string, Uint8Array>;
  manifest: BrowserModelPackManifest;
}> {
  const assets = new Map<string, Uint8Array>([
    ["tokenizer/tokenizer.json", encoder.encode("{}")],
    ["tokenizer/tokenizer_config.json", encoder.encode("{}")],
    ["graphs.onnx", encoder.encode("fake onnx")],
  ]);
  const files: BrowserModelPackManifest["files"] = {};
  for (const [path, bytes] of assets) {
    files[path] = {
      size_bytes: bytes.byteLength,
      sha256: await sha256Hex(bytes),
    };
  }
  const alphas = [0.99, 0.95, 0.88, 0.78, 0.66, 0.53, 0.4, 0.28, 0.17, 0.08];
  const manifest: BrowserModelPackManifest = {
    format: MODEL_PACK_FORMAT,
    schema_version: MODEL_PACK_SCHEMA_VERSION,
    model: { id: "fixture", variant: "test" },
    files,
    tokenizer: { directory: "tokenizer", max_length: 128 },
    graphs: {
      text_encoder: {
        model: "graphs.onnx",
        inputs: {
          inputIds: "input_ids",
          attentionMask: "attention_mask",
          tokenTypeIds: "token_type_ids",
        },
        outputs: { textConditions: "text_conditions" },
      },
      denoiser: {
        model: "graphs.onnx",
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
        model: "graphs.onnx",
        inputs: {
          hybridTokens: "hybrid_tokens",
          motionPadMask: "motion_pad_mask",
          globalTranslation: "global_translation",
        },
        outputs: {
          normalizedMotion: "normalized_motion",
          posedJoints: "posed_joints",
        },
      },
    },
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
      timesteps: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      alphas_cumprod: alphas,
      alphas_cumprod_prev: [1, ...alphas.slice(0, -1)],
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
    notices: ["fixture only"],
    license_notices: [
      { component: "fixture", license: "MIT", notice: "fixture only" },
    ],
  };
  const entries = new Map(assets);
  entries.set("manifest.json", encoder.encode(JSON.stringify(manifest)));
  return { entries, manifest };
}

describe("model-pack validation", () => {
  it("accepts and hashes an in-memory v1 pack", async () => {
    const { entries } = await fixture();
    const progress: string[] = [];
    const pack = await loadModelPackFromMemory(entries, (event) =>
      progress.push(`${event.stage}:${event.completed}/${event.total}`),
    );
    expect(pack.manifest.model.id).toBe("fixture");
    expect(new TextDecoder().decode(await pack.read("graphs.onnx"))).toBe(
      "fake onnx",
    );
    expect(progress.some((value) => value.startsWith("hashing-pack:3/3"))).toBe(
      true,
    );
  });

  it("rejects tampered bytes and ascending sampler timesteps", async () => {
    const { entries, manifest } = await fixture();
    entries.set("graphs.onnx", encoder.encode("tampered!"));
    await expect(loadModelPackFromMemory(entries)).rejects.toThrow(
      /SHA-256 mismatch/,
    );
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        diffusion: {
          ...manifest.diffusion,
          timesteps: [...manifest.diffusion.timesteps].reverse(),
        },
      }),
    ).toThrow(/strictly decreasing/);
  });

  it("rejects dimensions outside the fixed Core40 v1 contract", async () => {
    const { manifest } = await fixture();
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        dimensions: {
          ...manifest.dimensions,
          num_joints: 26,
        },
      }),
    ).toThrow(/browser Core40 v1 runtime/);
  });

  it("accepts the revision-2 constraint and structured-output extension", async () => {
    const { manifest } = await fixture();
    const extended = {
      ...manifest,
      graphs: {
        ...manifest.graphs,
        constraint_denoiser: {
          model: "graphs.onnx",
          inputs: {
            textCfgWeight: "text_cfg_weight",
            constraintCfgWeight: "constraint_cfg_weight",
            x: "x",
            historyLength: "history_len",
            generationLength: "generation_len",
            futureLength: "future_len",
            historyMask: "history_mask",
            generationMask: "generation_mask",
            futureMask: "future_mask",
            historyTokenMask: "history_token_mask",
            generationTokenMask: "generation_token_mask",
            futureTokenMask: "future_token_mask",
            textConditions: "text_conditions",
            textConditionMask: "text_condition_mask",
            timestep: "timestep",
            firstHeadingAngle: "first_heading_angle",
            motionMask: "motion_mask",
            observedMotion: "observed_motion",
          },
          outputs: { predX0: "pred_x0" },
        },
        decoder: {
          ...manifest.graphs.decoder,
          outputs: {
            ...manifest.graphs.decoder.outputs,
            localRotations: "local_rotations",
            globalRotations: "global_rotations",
            rootPositions: "root_positions",
            footContacts: "foot_contacts",
            globalRootHeading: "global_root_heading",
          },
        },
      },
      dimensions: {
        ...manifest.dimensions,
        constraint_max_tokens: 50,
        constraint_max_frames: 200,
      },
      generation: {
        ...manifest.generation,
        default_text_cfg_weight: 2,
        default_constraint_cfg_weight: 2,
      },
      capabilities: {
        text_conditioning: true,
        kinematic_constraints: true,
        detailed_motion_outputs: true,
      },
      runtime: {
        contract_revision: 2,
        onnx_opset: 18,
        batch_size: 1,
        constraints_supported: true,
        separated_cfg: true,
        future_constraints_supported: true,
        detailed_motion_outputs: true,
      },
    };
    expect(validateModelPackManifest(extended).runtime?.contract_revision).toBe(2);
    expect(
      validateModelPackManifest(extended).graphs.decoder.outputs.footContacts,
    ).toBe("foot_contacts");
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { createModelTestFixture } from "./model-assets.test-fixture";
import {
  MODEL_FILES_FORMAT,
  MODEL_FILES_SCHEMA_VERSION,
  normalizeModelPath,
  validateModelManifest,
} from "./manifest";

describe("browser model-files manifest", () => {
  it("accepts the fixed Core40 model and transport contracts", async () => {
    const { manifest } = await createModelTestFixture();
    const validated = validateModelManifest(manifest);

    expect(validated.format).toBe(MODEL_FILES_FORMAT);
    expect(validated.schema_version).toBe(MODEL_FILES_SCHEMA_VERSION);
    expect(validated.model).toMatchObject({
      id: "fixture",
      revision: "fixture-revision-1",
    });
    expect(
      validated.files["denoiser.onnx"].transport,
    ).toMatchObject({
      path: "denoiser.onnx.gz",
      compression: "gzip",
    });
  });

  it("requires an immutable revision and the current files format", async () => {
    const { manifest } = await createModelTestFixture();
    expect(() =>
      validateModelManifest({
        ...manifest,
        model: { id: manifest.model.id, variant: manifest.model.variant },
      }),
    ).toThrow(/model\.revision/);
    expect(() =>
      validateModelManifest({
        ...manifest,
        format: "ardy-browser-model-files-old",
      }),
    ).toThrow(/format/);
    expect(() =>
      validateModelManifest({ ...manifest, schema_version: 2 }),
    ).toThrow(/schema_version/);
  });

  it("rejects unsafe, duplicate, or unsupported transports", async () => {
    const { manifest } = await createModelTestFixture();
    const files = structuredClone(manifest.files);
    files["decoder.onnx"].transport.path = "../decoder.onnx.gz";
    expect(() =>
      validateModelManifest({ ...manifest, files }),
    ).toThrow(/unsafe relative asset path/);

    const duplicate = structuredClone(manifest.files);
    duplicate["decoder.onnx"].transport.path =
      duplicate["denoiser.onnx"].transport.path;
    expect(() =>
      validateModelManifest({ ...manifest, files: duplicate }),
    ).toThrow(/duplicate transport path/);

    const unsupported = structuredClone(manifest.files);
    unsupported["decoder.onnx"].transport.compression =
      "brotli" as "gzip";
    expect(() =>
      validateModelManifest({ ...manifest, files: unsupported }),
    ).toThrow(/compression must be "gzip"/);
  });

  it("binds graph precision metadata to raw model hashes and sizes", async () => {
    const { manifest } = await createModelTestFixture();
    const files = structuredClone(manifest.files);
    files["decoder.onnx"].size_bytes += 1;
    expect(() =>
      validateModelManifest({ ...manifest, files }),
    ).toThrow(/output_size_bytes/);

    const precision = structuredClone(manifest.precision);
    precision.graphs.decoder.output_sha256 = "0".repeat(64);
    expect(() =>
      validateModelManifest({ ...manifest, precision }),
    ).toThrow(/output_sha256/);
  });

  it("rejects non-Core40 dimensions, sampler order, and old fields", async () => {
    const { manifest } = await createModelTestFixture();
    expect(() =>
      validateModelManifest({
        ...manifest,
        dimensions: { ...manifest.dimensions, max_tokens: 21 },
      }),
    ).toThrow(/max_frames|must be 20/);
    expect(() =>
      validateModelManifest({
        ...manifest,
        diffusion: {
          ...manifest.diffusion,
          timesteps: [...manifest.diffusion.timesteps].reverse(),
        },
      }),
    ).toThrow(/decreasing/);
    expect(() =>
      validateModelManifest({
        ...manifest,
        runtime: {
          ...manifest.runtime,
          constraints_supported: true,
        },
      }),
    ).toThrow(/constraints_supported/);
  });

  it("normalizes only canonical relative model paths", () => {
    expect(normalizeModelPath("tokenizer/tokenizer.json")).toBe(
      "tokenizer/tokenizer.json",
    );
    for (const value of [
      "",
      "/absolute",
      "a/../b",
      "a//b",
      "%2e%2e/model.onnx",
      "data:model",
      "model.onnx?download=true",
      "model.onnx#fragment",
    ]) {
      expect(() => normalizeModelPath(value)).toThrow();
    }
    expect(normalizeModelPath("a\\b")).toBe("a/b");
  });
});

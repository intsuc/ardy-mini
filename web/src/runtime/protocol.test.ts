// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  MAX_WORKER_PROMPT_LENGTH,
  parseWorkerCommand,
  serializeWorkerError,
  WORKER_PROTOCOL_VERSION,
} from "./protocol";

describe("worker protocol v4", () => {
  it("accepts one tar.gz archive and rejects legacy load options", () => {
    const archive = new File(["gzip"], "model-pack.tar.gz", {
      type: "application/gzip",
    });
    expect(
      parseWorkerCommand({
        type: "loadModelPack",
        requestId: "load",
        archive,
      }),
    ).toMatchObject({
      type: "loadModelPack",
      archive,
    });
    expect(() =>
      parseWorkerCommand({
        type: "loadModelPack",
        requestId: "legacy-load",
        files: [archive],
      }),
    ).toThrow(/archive/);
    expect(() =>
      parseWorkerCommand({
        type: "loadModelPack",
        requestId: "legacy-backend",
        archive,
        backend: "wasm",
      }),
    ).toThrow(/requires WebGPU/);
    expect(() =>
      parseWorkerCommand({
        type: "loadModelPack",
        requestId: "legacy-paths",
        archive,
        wasmPaths: "/ort/",
      }),
    ).toThrow(/requires WebGPU/);
    expect(() =>
      parseWorkerCommand({
        type: "loadModelPack",
        requestId: "wrong-extension",
        archive: new File(["gzip"], "model-pack.zip"),
      }),
    ).toThrow(/\.tar\.gz/);
  });

  it("parses a replace-compatible generation command", () => {
    expect(WORKER_PROTOCOL_VERSION).toBe(4);
    const command = parseWorkerCommand({
      type: "generate",
      requestId: "legacy",
      prompt: "A person walks forward.",
      seed: 42,
      durationSeconds: 4,
      cfgWeight: 2,
    });

    expect(command).toMatchObject({
      type: "generate",
      requestId: "legacy",
      seed: 42,
      durationSeconds: 4,
      cfgWeight: 2,
    });
    expect(command).not.toHaveProperty("mode");
  });

  it("normalizes a branch command and clones mutable inputs", () => {
    const translation = new Float64Array([1, 2, 3]);
    const command = parseWorkerCommand({
      type: "generate",
      requestId: "branch",
      mode: "branch",
      branchFrame: 80,
      prompt: "Turn left and raise both hands.",
      seed: "take-two",
      durationFrames: 40,
      cfgWeight: 2.5,
      historyFrames: 40,
    });

    expect(command).toMatchObject({
      type: "generate",
      mode: "branch",
      branchFrame: 80,
      historyFrames: 40,
    });

    const replace = parseWorkerCommand({
      type: "generate",
      requestId: "replace",
      mode: "replace",
      prompt: "Stand still.",
      seed: 0,
      durationFrames: 40,
      initialTranslation: translation,
      initialHeading: Math.PI / 2,
    });
    if (replace.type !== "generate") {
      throw new Error("unreachable");
    }
    expect(replace.initialTranslation).toBeInstanceOf(Float32Array);
    expect(Array.from(replace.initialTranslation ?? [])).toEqual([1, 2, 3]);
  });

  it("rejects ambiguous modes, durations, weights, and transforms", () => {
    const base = {
      type: "generate",
      requestId: "bad",
      prompt: "Walk.",
      seed: 1,
      durationFrames: 40,
    };
    expect(() =>
      parseWorkerCommand({ ...base, mode: "branch" }),
    ).toThrow(/branchFrame/);
    expect(() =>
      parseWorkerCommand({ ...base, mode: "append", branchFrame: 0 }),
    ).toThrow(/only valid in branch mode/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        durationSeconds: 2,
      }),
    ).toThrow(/exactly one/);
    expect(() =>
      parseWorkerCommand({ ...base, cfgWeight: 101 }),
    ).toThrow(/between 0 and 100/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        mode: "append",
        initialTranslation: [0, 0, 0],
      }),
    ).toThrow(/only valid in replace mode/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        prompt: "x".repeat(MAX_WORKER_PROMPT_LENGTH + 1),
      }),
    ).toThrow(/at most/);
  });

  it("validates and clones reset and continuation commands", () => {
    const reset = parseWorkerCommand({
      type: "resetSession",
      requestId: "reset",
      seed: "fresh",
      initialTranslation: [1, 0, -1],
      initialHeading: 0.5,
    });
    expect(reset).toMatchObject({
      type: "resetSession",
      seed: "fresh",
      initialHeading: 0.5,
    });

    const hybridTokens = new Float32Array(2 * 148);
    const restored = parseWorkerCommand({
      type: "restoreContinuation",
      requestId: "restore",
      continuation: {
        frameCount: 8,
        hybridTokens,
        hybridDim: 148,
        random: { seed: 7, state: 11, spareNormal: 0.25 },
        initialTranslation: [1, 0, 2],
        initialHeading: 0.75,
      },
    });
    if (restored.type !== "restoreContinuation") {
      throw new Error("unreachable");
    }
    expect(restored.continuation.hybridTokens).toEqual(hybridTokens);
    expect(restored.continuation.hybridTokens).not.toBe(hybridTokens);
    expect(restored.continuation.random).toEqual({
      seed: 7,
      state: 11,
      spareNormal: 0.25,
    });

    expect(() =>
      parseWorkerCommand({
        type: "restoreContinuation",
        requestId: "bad-restore",
        continuation: {
          frameCount: 8,
          hybridTokens: new Float32Array(149),
          hybridDim: 148,
          random: { seed: 7, state: 11 },
          initialTranslation: [0, 0, 0],
          initialHeading: 0,
        },
      }),
    ).toThrow(/divisible/);
  });

  it("serializes errors without trusting thrown values", () => {
    expect(serializeWorkerError("request", new RangeError("bad"))).toMatchObject({
      type: "error",
      requestId: "request",
      error: { name: "RangeError", message: "bad" },
    });
    expect(serializeWorkerError("request", { secret: true })).toEqual({
      type: "error",
      requestId: "request",
      error: { name: "Error", message: "[object Object]" },
    });
  });
});

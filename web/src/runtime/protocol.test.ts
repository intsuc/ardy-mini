// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  MAX_WORKER_CONSTRAINTS,
  MAX_WORKER_PROMPT_LENGTH,
  parseWorkerCommand,
  serializeWorkerError,
  WORKER_PROTOCOL_VERSION,
} from "./protocol";

function constraint(id = "root-1") {
  const values = new Float32Array(330);
  const mask = new Float32Array(330);
  values[0] = 0.25;
  mask[0] = 1;
  return {
    id,
    kind: "root",
    frame: 40,
    endFrame: 44,
    values,
    mask,
  };
}

describe("worker protocol v2", () => {
  it("keeps the protocol-v1 generate shape as a replace-compatible command", () => {
    expect(WORKER_PROTOCOL_VERSION).toBe(2);
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

  it("normalizes a constrained branch command and clones mutable inputs", () => {
    const sourceConstraint = constraint();
    const translation = new Float64Array([1, 2, 3]);
    const command = parseWorkerCommand({
      type: "generate",
      requestId: "branch",
      mode: "branch",
      branchFrame: 80,
      prompt: "Turn left and raise both hands.",
      seed: "take-two",
      durationFrames: 40,
      textCfgWeight: 2.5,
      constraintCfgWeight: 1.25,
      historyFrames: 40,
      futureFrames: 120,
      constraints: [sourceConstraint],
    });

    expect(command).toMatchObject({
      type: "generate",
      mode: "branch",
      branchFrame: 80,
      historyFrames: 40,
      futureFrames: 120,
    });
    if (command.type !== "generate") {
      throw new Error("unreachable");
    }
    expect(command.constraints?.[0].values).toEqual(sourceConstraint.values);
    expect(command.constraints?.[0].values).not.toBe(sourceConstraint.values);

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
      parseWorkerCommand({
        ...base,
        cfgWeight: 2,
        textCfgWeight: 3,
      }),
    ).toThrow(/must not both/);
    expect(() =>
      parseWorkerCommand({ ...base, textCfgWeight: 101 }),
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

  it("validates sparse constraint tensors before model allocation", () => {
    const base = {
      type: "generate",
      requestId: "constraint",
      mode: "replace",
      prompt: "Walk.",
      seed: 1,
      durationFrames: 40,
    };
    expect(() =>
      parseWorkerCommand({
        ...base,
        constraints: [
          {
            ...constraint(),
            values: [0],
          },
        ],
      }),
    ).toThrow(/Float32Array/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        constraints: [
          {
            ...constraint(),
            mask: new Float32Array(329),
          },
        ],
      }),
    ).toThrow(/equal lengths/);
    const nonFinite = constraint();
    nonFinite.values[0] = Number.NaN;
    expect(() =>
      parseWorkerCommand({ ...base, constraints: [nonFinite] }),
    ).toThrow(/finite/);
    const invalidMask = constraint();
    invalidMask.mask[0] = 2;
    expect(() =>
      parseWorkerCommand({ ...base, constraints: [invalidMask] }),
    ).toThrow(/between 0 and 1/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        constraints: [constraint("same"), constraint("same")],
      }),
    ).toThrow(/unique/);
    expect(() =>
      parseWorkerCommand({
        ...base,
        constraints: Array(MAX_WORKER_CONSTRAINTS + 1).fill(constraint()),
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

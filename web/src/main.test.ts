// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  canAttemptGeneration,
  canContinueGeneration,
  canonicalizePackFiles,
  formatBytes,
  formatTime,
  sanitizeImportedEditorState,
  shouldAutoplayMotion,
  validateGenerationForm,
} from "./main";
import type { MotionEditorState } from "./editor-state";

function directoryFile(path: string, contents = "x"): File {
  const file = new File([contents], path.split("/").at(-1) ?? path);
  Object.defineProperty(file, "webkitRelativePath", {
    configurable: true,
    value: path,
  });
  return file;
}

describe("generation form validation", () => {
  it("accepts the supported prompt, duration, and uint32 seed domain", () => {
    expect(validateGenerationForm("  A person walks forward.  ", "4", "4294967295")).toEqual({
      values: {
        prompt: "A person walks forward.",
        durationSeconds: 4,
        seed: 4294967295,
      },
    });
  });

  it("accepts multilingual prompts without changing their contents", () => {
    expect(validateGenerationForm("人物が歩く。", "4", "2")).toEqual({
      values: {
        prompt: "人物が歩く。",
        durationSeconds: 4,
        seed: 2,
      },
    });
  });

  it("rejects blank and overlong prompts, invalid windows, and invalid seeds", () => {
    expect(validateGenerationForm("", "4", "2").promptError).toMatch(/Describe/);
    expect(validateGenerationForm("a".repeat(281), "4", "2").promptError).toMatch(/280/);
    expect(validateGenerationForm("A person walks.", "3", "2").promptError).toMatch(/2 to 10/);
    expect(validateGenerationForm("A person walks.", "4", "-1").seedError).toMatch(/whole-number/);
  });

  it("keeps non-empty multilingual input submittable", () => {
    expect(canAttemptGeneration("人物が歩く。", true, true, false, false)).toBe(true);
    expect(canAttemptGeneration("   ", true, true, false, false)).toBe(false);
    expect(canAttemptGeneration("A person walks.", true, false, false, false)).toBe(false);
  });

  it("only enables append and branch actions for a live continuation", () => {
    expect(canContinueGeneration(true, false, true, true)).toBe(true);
    expect(canContinueGeneration(true, false, true, false)).toBe(false);
    expect(canContinueGeneration(true, true, true, true)).toBe(false);
    expect(canContinueGeneration(false, false, true, true)).toBe(false);
  });
});

describe("model-pack file normalization", () => {
  it("strips the directory-picker root and retains nested graph paths", () => {
    const files = canonicalizePackFiles([
      directoryFile("ardy-pack/graphs/decoder.onnx", "decoder"),
      directoryFile("ardy-pack/manifest.json", "{}"),
      directoryFile("ardy-pack/tokenizer/tokenizer.json", "{}"),
    ]);

    expect(files.map((file) => file.name)).toEqual([
      "manifest.json",
      "graphs/decoder.onnx",
      "tokenizer/tokenizer.json",
    ]);
  });

  it("requires exactly one manifest", () => {
    expect(() => canonicalizePackFiles([directoryFile("ardy-pack/model.onnx")])).toThrow(/manifest/);
    expect(() =>
      canonicalizePackFiles([
        directoryFile("one/manifest.json"),
        directoryFile("two/manifest.json"),
      ]),
    ).toThrow(/more than one/);
  });
});

describe("display formatting", () => {
  it("formats playback time and binary sizes", () => {
    expect(formatTime(4.25)).toBe("00:04.25");
    expect(formatTime(65)).toBe("01:05.00");
    expect(formatBytes(1024 ** 3)).toBe("1.0 GiB");
    expect(formatBytes(0)).toBe("0 B");
  });

  it("does not autoplay generated motion when reduced motion is requested", () => {
    expect(shouldAutoplayMotion(false)).toBe(true);
    expect(shouldAutoplayMotion(true)).toBe(false);
  });
});

describe("removed motion parameters", () => {
  it("keeps display preferences but discards hidden imported controls", () => {
    const imported = {
      initialTransform: {
        position: [4, 2, -3],
        headingRadians: Math.PI / 2,
      },
      waypoints: [
        {
          id: "old-waypoint",
          frame: 12,
          position: [1, 0, 2],
          enabled: true,
        },
      ],
      constraints: [
        {
          id: "old-constraint",
          kind: "root",
          startFrame: 4,
          endFrame: 8,
          position: [3, 0, 1],
          enabled: true,
        },
      ],
      outputVisibility: {
        skeleton: false,
        mesh: true,
        reference: true,
        trajectory: false,
        contacts: false,
        orientationAxes: true,
        constraints: true,
        initialTransform: true,
        waypoints: true,
      },
    } satisfies MotionEditorState;

    const sanitized = sanitizeImportedEditorState(imported);

    expect(sanitized.initialTransform).toEqual({
      position: [0, 0, 0],
      headingRadians: 0,
    });
    expect(sanitized.waypoints).toEqual([]);
    expect(sanitized.constraints).toEqual([]);
    expect(sanitized.outputVisibility).toMatchObject({
      skeleton: false,
      mesh: true,
      reference: false,
      trajectory: false,
      contacts: false,
      orientationAxes: true,
    });
  });
});

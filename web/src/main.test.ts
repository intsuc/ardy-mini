// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  canAttemptGeneration,
  canonicalizePackFiles,
  formatBytes,
  formatTime,
  shouldAutoplayMotion,
  validateGenerationForm,
} from "./main";

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

  it("rejects blank/non-Latin prompts, invalid windows, and invalid seeds", () => {
    expect(validateGenerationForm("", "4", "2").promptError).toMatch(/Describe/);
    expect(validateGenerationForm("人物が歩く。", "4", "2").promptError).toMatch(/English/);
    expect(validateGenerationForm("A person walks.", "3", "2").promptError).toMatch(/2 to 10/);
    expect(validateGenerationForm("A person walks.", "4", "-1").seedError).toMatch(/whole-number/);
  });

  it("keeps non-empty invalid input submittable so inline validation is reachable", () => {
    expect(canAttemptGeneration("人物が歩く。", true, true, false, false)).toBe(true);
    expect(canAttemptGeneration("   ", true, true, false, false)).toBe(false);
    expect(canAttemptGeneration("A person walks.", true, false, false, false)).toBe(false);
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

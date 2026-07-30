// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  canAttemptGeneration,
  canContinueGeneration,
  cameraMoveForCode,
  formatBytes,
  formatTime,
  isModelPackArchive,
  isVrmFile,
  resolveGenerationProgressState,
  shouldAutoplayMotion,
  shouldResetMotionPresentation,
  shouldShowIdleGenerationStatus,
  validateGenerationForm,
} from "./main";

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

describe("model-pack archive selection", () => {
  it("accepts only a non-empty .tar.gz file", () => {
    expect(
      isModelPackArchive(
        new File(["gzip bytes"], "ardy-minilm-core40-browser-v1.tar.gz"),
      ),
    ).toBe(true);
    expect(isModelPackArchive(new File(["x"], "legacy-pack.zip"))).toBe(false);
    expect(isModelPackArchive(new File([], "empty.tar.gz"))).toBe(false);
  });
});

describe("VRM file selection", () => {
  it("accepts the VRM extension case-insensitively", () => {
    expect(isVrmFile(new File(["vrm"], "avatar.vrm"))).toBe(true);
    expect(isVrmFile(new File(["vrm"], "AVATAR.VRM"))).toBe(true);
    expect(isVrmFile(new File(["glb"], "avatar.glb"))).toBe(false);
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

describe("generation progress presentation", () => {
  it("preserves partial playback-only motion instead of resetting to idle", () => {
    expect(resolveGenerationProgressState(false, true, 0.35)).toBe(
      "playback-only",
    );
    expect(
      shouldShowIdleGenerationStatus(false, "playback-only"),
    ).toBe(false);
    expect(shouldShowIdleGenerationStatus(false, "idle")).toBe(true);
  });
});

describe("camera keyboard mapping", () => {
  it("maps physical WASD keys to view-relative movement", () => {
    expect(cameraMoveForCode("KeyW")).toEqual([1, 0]);
    expect(cameraMoveForCode("KeyA")).toEqual([0, -1]);
    expect(cameraMoveForCode("KeyS")).toEqual([-1, 0]);
    expect(cameraMoveForCode("KeyD")).toEqual([0, 1]);
    expect(cameraMoveForCode("KeyZ")).toBeNull();
  });
});

describe("streaming presentation policy", () => {
  it("resets only the first visual update of a replacement generation", () => {
    expect(shouldResetMotionPresentation("replace", true)).toBe(true);
    expect(shouldResetMotionPresentation("replace", false)).toBe(false);
    expect(shouldResetMotionPresentation("append", true)).toBe(false);
    expect(shouldResetMotionPresentation("append", false)).toBe(false);
    expect(shouldResetMotionPresentation("branch", true)).toBe(false);
    expect(shouldResetMotionPresentation("branch", false)).toBe(false);
  });
});

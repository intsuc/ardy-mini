// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  canAttemptGeneration,
  canContinueGeneration,
  cameraMoveForCode,
  cameraMovementForCodes,
  formatBytes,
  formatTime,
  isModelPackArchive,
  isVrmFile,
  livePromptBranchFrame,
  resolveGenerationProgressState,
  resolvePromptActionState,
  shouldAutoplayMotion,
  shouldResetMotionPresentation,
  shouldShowIdleGenerationStatus,
  validateGenerationForm,
} from "./main";

describe("generation form validation", () => {
  it("accepts the supported prompt and uint32 seed domain", () => {
    expect(
      validateGenerationForm("  A person walks forward.  ", "4294967295"),
    ).toEqual({
      values: {
        prompt: "A person walks forward.",
        seed: 4294967295,
      },
    });
  });

  it("accepts multilingual prompts without changing their contents", () => {
    expect(validateGenerationForm("人物が歩く。", "2")).toEqual({
      values: {
        prompt: "人物が歩く。",
        seed: 2,
      },
    });
  });

  it("rejects blank and overlong prompts and invalid seeds", () => {
    expect(validateGenerationForm("", "2").promptError).toMatch(/Describe/);
    expect(validateGenerationForm("a".repeat(281), "2").promptError).toMatch(
      /280/,
    );
    expect(validateGenerationForm("A person walks.", "-1").seedError).toMatch(
      /whole-number/,
    );
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

describe("continuous prompt actions", () => {
  it("starts without motion and updates only a dirty live prompt", () => {
    expect(
      resolvePromptActionState(false, false, "  Walk forward. ", null),
    ).toEqual({
      label: "Start motion",
      dirty: true,
      canSubmit: true,
    });
    expect(
      resolvePromptActionState(true, true, "Walk forward.", "Walk forward."),
    ).toEqual({
      label: "Update motion",
      dirty: false,
      canSubmit: false,
    });
    expect(
      resolvePromptActionState(true, true, "Turn left.", "Walk forward."),
    ).toEqual({
      label: "Update motion",
      dirty: true,
      canSubmit: true,
    });
  });

  it("does not update a playback-only motion", () => {
    expect(
      resolvePromptActionState(true, false, "Turn left.", "Walk forward.")
        .canSubmit,
    ).toBe(false);
    expect(
      resolvePromptActionState(false, false, "a".repeat(281), null).canSubmit,
    ).toBe(false);
  });

  it("branches a live prompt after a short lookahead without passing the end", () => {
    expect(livePromptBranchFrame(12.9, 80)).toBe(32);
    expect(livePromptBranchFrame(75, 80)).toBe(80);
    expect(livePromptBranchFrame(-5, 80)).toBe(20);
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

  it("combines simultaneous movement keys and cancels opposing inputs", () => {
    expect(cameraMovementForCodes(new Set(["KeyW", "KeyD"]))).toEqual([1, 1]);
    expect(cameraMovementForCodes(new Set(["KeyW", "KeyS"]))).toEqual([0, 0]);
    expect(
      cameraMovementForCodes(new Set(["KeyW", "KeyA", "KeyS", "KeyD"])),
    ).toEqual([0, 0]);
  });

  it("ignores unrelated keys and preserves the remaining held direction", () => {
    const heldCodes = new Set(["KeyW", "KeyD", "KeyZ"]);
    expect(cameraMovementForCodes(heldCodes)).toEqual([1, 1]);

    heldCodes.delete("KeyW");
    expect(cameraMovementForCodes(heldCodes)).toEqual([0, 1]);

    heldCodes.delete("KeyD");
    expect(cameraMovementForCodes(heldCodes)).toEqual([0, 0]);
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

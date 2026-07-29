// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

test("renders the model-gated motion studio and reports runtime capabilities", async ({ page }) => {
  await page.goto("/");

  await expect
    .poll(() => page.evaluate(() => globalThis.crossOriginIsolated))
    .toBe(true);
  await expect(page.getByRole("heading", { name: /Describe a motion/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Load the local model" })).toBeVisible();
  await expect(page.getByLabel("Motion prompt")).toBeEditable();
  await expect(page.getByRole("button", { name: "Generate motion" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Choose model-pack folder" })).toBeVisible();
  await expect(page.getByText(/artifacts\/browser\/core40/)).toBeVisible();
  await expect(page.getByText(/about 798 MiB/)).toBeVisible();
  await expect(page.getByText(/WASM (threads ready|single-thread)/)).toBeVisible();
  await expect(page.locator("#model-state")).toHaveText(/Checking|Not loaded/);
  await expect
    .poll(() =>
      page.evaluate(() => {
        const setup = document.querySelector(".model-setup");
        const prompt = document.querySelector("#prompt");
        return Boolean(setup && prompt && setup.compareDocumentPosition(prompt) & Node.DOCUMENT_POSITION_FOLLOWING);
      }),
    )
    .toBe(true);
});

test("validates prompt and exposes deterministic controls without model weights", async ({ page }) => {
  await page.goto("/");

  const prompt = page.getByLabel("Motion prompt");
  await prompt.fill("A person walks forward confidently.");
  await expect(page.locator("#prompt-count")).toHaveText("35 / 280");

  await page.getByLabel("Duration").fill("8");
  await expect(page.locator("#duration-output")).toHaveText("8 seconds");
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toBeHidden();
  await page.getByText("Runtime settings").click();
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toHaveValue("2");
  await expect(page.getByLabel("Inference backend")).toHaveValue("auto");

  await page.getByRole("button", { name: "Dance" }).click();
  await expect(prompt).toHaveValue("A person performs a joyful dance.");
});

test("keeps keyboard semantics and focus indication intact", async ({ page }) => {
  await page.goto("/");

  const settings = page.locator("summary");
  await settings.focus();
  await page.keyboard.press("Space");
  await expect(page.locator(".advanced-settings")).toHaveAttribute("open", "");

  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await expect(canvas).toBeFocused();
  await expect
    .poll(() => canvas.evaluate((element) => getComputedStyle(element).outlineStyle))
    .not.toBe("none");
  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("=");
  await page.keyboard.press("Home");
});

test("keeps model import errors beside the model setup action", async ({ page }, testInfo) => {
  await page.goto("/");

  const invalidPack = testInfo.outputPath("invalid-pack");
  await mkdir(invalidPack, { recursive: true });
  await writeFile(path.join(invalidPack, "not-a-model-pack.txt"), "not a model pack");
  await page.locator("#model-file-input").setInputFiles(invalidPack);

  const modelError = page.locator("#model-error-banner");
  await expect(modelError).toBeVisible();
  await expect(modelError).toBeFocused();
  await expect(modelError).toContainText("manifest.json");
  await expect(page.locator("#error-banner")).toBeHidden();
  await expect
    .poll(() =>
      page.evaluate(() => {
        const setup = document.querySelector(".model-setup");
        const error = document.querySelector("#model-error-banner");
        return Boolean(setup && error && setup.contains(error));
      }),
    )
    .toBe(true);
});

test("honors reduced motion in CSS and keeps mobile playback controls on one row", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect
    .poll(() => page.locator(".loading-orbit").evaluate((element) => getComputedStyle(element).animationName))
    .toBe("none");
  await expect(page.locator("#privacy-badge")).toBeVisible();
  await expect(page.locator("#gpu-badge")).toBeHidden();

  const controls = [
    page.locator("#play-pause"),
    page.locator("#playback-speed"),
    page.locator("#loop-toggle"),
    page.locator("#reset-camera"),
  ];
  const boxes = await Promise.all(controls.map((control) => control.boundingBox()));
  for (const box of boxes) {
    expect(box).not.toBeNull();
    expect(box?.width ?? 0).toBeGreaterThanOrEqual(44);
    expect(box?.height ?? 0).toBeGreaterThanOrEqual(44);
  }
  expect(Math.max(...boxes.map((box) => box?.y ?? 0)) - Math.min(...boxes.map((box) => box?.y ?? 0))).toBeLessThan(2);
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
    .toBe(true);
});

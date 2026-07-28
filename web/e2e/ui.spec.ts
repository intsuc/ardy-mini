// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

test("renders the model-gated motion studio and reports runtime capabilities", async ({ page }) => {
  await page.goto("/");

  await expect
    .poll(() => page.evaluate(() => globalThis.crossOriginIsolated))
    .toBe(true);
  await expect(page.getByRole("heading", { name: /Describe a motion/i })).toBeVisible();
  await expect(page.getByLabel("Motion prompt")).toBeEditable();
  await expect(page.getByRole("button", { name: "Generate motion" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Import model pack" })).toBeVisible();
  await expect(page.getByText(/WASM (threads ready|single-thread)/)).toBeVisible();
  await expect(page.locator("#model-state")).toHaveText(/Checking|Not loaded/);
});

test("validates prompt and exposes deterministic controls without model weights", async ({ page }) => {
  await page.goto("/");

  const prompt = page.getByLabel("Motion prompt");
  await prompt.fill("A person walks forward confidently.");
  await expect(page.locator("#prompt-count")).toHaveText("35 / 280");

  await page.getByLabel("Duration").fill("8");
  await expect(page.locator("#duration-output")).toHaveText("8 seconds");
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toHaveValue("2");
  await expect(page.getByLabel("Inference backend")).toHaveValue("auto");

  await page.getByRole("button", { name: "Dance" }).click();
  await expect(prompt).toHaveValue("A person performs a joyful dance.");
});

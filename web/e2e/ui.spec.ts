// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

async function setRange(page: Page, selector: string, value: number): Promise<void> {
  await page.locator(selector).evaluate((element, nextValue) => {
    const input = element as HTMLInputElement;
    input.value = String(nextValue);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
}

test("renders the two-pane technical workspace without motion parameters", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");

  await expect
    .poll(() =>
      page
        .evaluate(() => globalThis.crossOriginIsolated)
        .catch(() => false),
    )
    .toBe(true);
  await expect(page.locator("#model-state")).toHaveText("Not loaded");
  await expect(page.locator("#generator-panel")).toBeVisible();
  await expect(page.locator("#viewport-panel")).toBeVisible();
  await expect(page.locator(".inspector-panel")).toHaveCount(0);

  const panePositions = await Promise.all(
    ["#generator-panel", "#viewport-panel"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(panePositions.every(Boolean)).toBe(true);
  expect(panePositions[0]!.x).toBeLessThan(panePositions[1]!.x);

  await expect(page.getByLabel("Motion description")).toBeEditable();
  await expect(
    page.getByText(
      "Clear, typo-free English. Apply updates while streaming.",
      { exact: true },
    ),
  ).toHaveCount(0);
  await expect(page.locator(".setup-note")).toContainText(
    "about 1.4 GiB, four ONNX graphs",
  );
  await expect(page.locator("#privacy-badge")).toContainText("Local");
  await expect(page.locator("#isolation-label")).toContainText(
    /WASM (threads ready|single-thread)/,
  );
  await expect(page.locator("#import-model")).toContainText("Choose model pack");

  for (const selector of [
    "#generate",
    "#apply-prompt",
    "#restart-generation",
    "#restart-from-now",
    "#export-session",
    "#export-motion",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  await expect(page.locator("#new-session")).toBeEnabled();
  await expect(page.locator("#import-session")).toBeEnabled();

  for (const selector of [
    "#initial-x",
    "#initial-z",
    "#initial-heading",
    "#text-cfg",
    "#constraint-cfg",
    "#history-frames",
    "#future-crop",
    "#replan-buffer",
    "#replan-threshold",
    "#constraint-timeline",
    ".constraint-track",
    "#constraint-type",
    "#constraint-frame",
    "#constraint-end-frame",
    "#add-constraint",
    "#delete-constraint",
    "#clear-constraints",
    "#waypoint-mode",
    "#waypoint-interval",
    "#waypoint-dense",
    "#add-waypoint",
    "#target-velocity",
    "#target-heading",
    "#apply-target-velocity",
    "#postprocess-enabled",
    "#root-height-margin",
    "#contact-threshold",
  ]) {
    await expect(page.locator(selector)).toHaveCount(0);
  }
  for (const label of [
    "Control",
    "Motion parameters",
    "Initial transform",
    "Guidance and planning",
    "Constraints",
    "Root control",
    "Postprocess",
  ]) {
    await expect(page.getByText(label, { exact: true })).toHaveCount(0);
  }

  await expect(page.locator("#preview-settings")).toBeVisible();
  await page.locator("#preview-settings > summary").click();
  await expect(page.getByText("VRM avatar", { exact: true })).toBeVisible();
  await expect(page.locator("#import-vrm")).toBeEnabled();
  await expect(page.getByText("Foot contacts", { exact: true })).toBeVisible();
  await expect(page.getByText("Orientations", { exact: true })).toBeVisible();
  await expect(page.getByText("Body proxy", { exact: true })).toBeVisible();
});

test("retains the square Lyra treatment on standard shadcn surfaces", async ({
  page,
}) => {
  await page.goto("/");

  const radii = await page.evaluate(() =>
    [
      "#model-card",
      "#privacy-badge",
      "#generate",
      "#prompt",
      "#seed",
      "#playback-speed",
    ].map((selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing Lyra surface: ${selector}`);
      }
      return getComputedStyle(element).borderRadius;
    }),
  );

  expect(radii).toEqual(["0px", "0px", "0px", "0px", "0px", "0px"]);
});

test("exposes deterministic inputs and enforces the prompt contract", async ({
  page,
}) => {
  await page.goto("/");

  const prompt = page.getByLabel("Motion description");
  await prompt.fill("A person walks forward confidently.");
  await expect(page.locator("#prompt-count")).toHaveText("35 / 280");

  await setRange(page, "#duration", 8);
  await expect(page.locator("#duration-output")).toHaveText("8 seconds");
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toHaveValue("2");
  await expect(page.getByLabel("Backend")).toHaveValue("auto");

  const promptExample = page.getByRole("combobox", {
    name: "Example prompt",
  });
  await promptExample.click();
  await expect(page.getByRole("option")).toHaveCount(10);
  await page.getByRole("option", { name: "Joyful dance" }).click();
  await expect(prompt).toHaveValue("A person performs a joyful dance.");

  const validation = await page.evaluate(async () => {
    const { validateGenerationForm } = await import("/src/main.ts");
    return {
      empty: validateGenerationForm("", "2", "2").promptError,
      multilingual: validateGenerationForm("人物が歩く。", "2", "2").values,
      long: validateGenerationForm("a".repeat(281), "2", "2").promptError,
      seed: validateGenerationForm("A person walks.", "2", "-1").seedError,
      duration: validateGenerationForm("A person walks.", "3", "2").promptError,
      valid: validateGenerationForm(
        "A person walks forward.",
        "10",
        "4294967295",
      ).values,
    };
  });
  expect(validation.empty).toContain("Describe the motion");
  expect(validation.multilingual).toEqual({
    prompt: "人物が歩く。",
    durationSeconds: 2,
    seed: 2,
  });
  expect(validation.long).toContain("280 characters");
  expect(validation.seed).toContain("whole-number seed");
  expect(validation.duration).toContain("2 to 10 seconds");
  expect(validation.valid).toEqual({
    prompt: "A person walks forward.",
    durationSeconds: 10,
    seed: 4_294_967_295,
  });
});

test("keeps labels, keyboard focus, and canvas controls accessible", async ({
  page,
}) => {
  await page.goto("/");

  await page.keyboard.press("Tab");
  await expect(page.locator(".skip-link")).toBeFocused();

  const runtimeNotes = page.locator("#runtime-settings");
  const summary = runtimeNotes.locator("summary");
  await summary.focus();
  await page.keyboard.press("Space");
  await expect(runtimeNotes).toHaveAttribute("open", "");

  for (const label of [
    "Motion description",
    "Duration",
    "Seed",
    "Backend",
  ]) {
    await expect(page.getByLabel(label, { exact: true })).toHaveCount(1);
  }

  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await expect(canvas).toBeFocused();
  await expect
    .poll(() =>
      canvas.evaluate((element) => getComputedStyle(element).outlineStyle),
    )
    .not.toBe("none");
  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("=");
  await page.keyboard.press("Home");
});

test("keeps invalid model-pack errors beside the model setup action", async ({
  page,
}, testInfo) => {
  await page.goto("/");

  const invalidPack = testInfo.outputPath("invalid-pack");
  await mkdir(invalidPack, { recursive: true });
  await writeFile(
    path.join(invalidPack, "not-a-model-pack.txt"),
    "not a model pack",
  );
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

test("keeps saved model actions inside the input panel at minimum width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/");
  await page.locator("#remove-model").evaluate((element) => {
    (element as HTMLButtonElement).hidden = false;
  });

  const [panel, importButton, removeButton] = await Promise.all(
    ["#generator-panel", "#import-model", "#remove-model"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(panel).not.toBeNull();
  expect(importButton).not.toBeNull();
  expect(removeButton).not.toBeNull();
  expect(importButton!.x + importButton!.width).toBeLessThanOrEqual(
    panel!.x + panel!.width,
  );
  expect(removeButton!.x + removeButton!.width).toBeLessThanOrEqual(
    panel!.x + panel!.width,
  );
  expect(
    await page
      .locator("#generator-panel")
      .evaluate((element) => element.scrollWidth <= element.clientWidth),
  ).toBe(true);
});

test("stacks the workspace before the two-pane layout can overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 800, height: 900 });
  await page.goto("/");

  const panes = await Promise.all(
    ["#generator-panel", "#viewport-panel"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(panes.every(Boolean)).toBe(true);
  expect(panes[0]!.y).toBeLessThan(panes[1]!.y);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
});

test("honors reduced motion and remains touch-safe without mobile overflow", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect
    .poll(() =>
      page
        .locator(".loading-indicator")
        .evaluate((element) => getComputedStyle(element).animationName),
    )
    .toBe("none");
  await expect
    .poll(() =>
      page
        .locator("#generate")
        .evaluate((element) => getComputedStyle(element).transitionDuration),
    )
    .toBe("0s");

  await expect(page.locator("#generation-progress")).toBeVisible();
  await expect(page.locator("#generation-progress")).not.toHaveAttribute(
    "hidden",
    "",
  );
  const idleGenerationStatus =
    await page.locator("#generation-progress").boundingBox();
  await page.locator("#generation-progress").evaluate((element) => {
    element.setAttribute("data-state", "active");
  });
  const activeGenerationStatus =
    await page.locator("#generation-progress").boundingBox();
  expect(activeGenerationStatus).toEqual(idleGenerationStatus);

  await page.locator("#remove-model").evaluate((element) => {
    (element as HTMLButtonElement).hidden = false;
  });
  const modelActionBounds = await Promise.all(
    ["#import-model", "#remove-model", "#generator-panel"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(modelActionBounds.every(Boolean)).toBe(true);
  expect(
    modelActionBounds[1]!.x + modelActionBounds[1]!.width,
  ).toBeLessThanOrEqual(
    modelActionBounds[2]!.x + modelActionBounds[2]!.width,
  );
  await expect
    .poll(() =>
      page.locator("#generator-panel").evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
      ),
    )
    .toBe(true);

  const panes = await Promise.all(
    ["#generator-panel", "#viewport-panel"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(panes.every(Boolean)).toBe(true);
  expect(panes[0]!.y).toBeLessThan(panes[1]!.y);
  await expect(page.locator(".inspector-panel")).toHaveCount(0);

  await page.locator("#preview-settings > summary").click();
  const preview = await page.locator("#viewport").boundingBox();
  expect(preview).not.toBeNull();
  expect(preview!.height).toBeGreaterThanOrEqual(384);

  const mobileFormFontSizes = await page.evaluate(() =>
    ["#prompt", "#seed", "#backend", "#prompt-example"].map((selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing mobile form control: ${selector}`);
      }
      return getComputedStyle(element).fontSize;
    }),
  );
  expect(mobileFormFontSizes).toEqual(["16px", "16px", "16px", "16px"]);

  const tapTargetSelectors = [
    "#new-session",
    "#import-session",
    "#export-session",
    "#import-model",
    "#prompt-example",
    "#apply-prompt",
    "#generate",
    "#play-pause",
    "#playback-speed",
    "#loop-toggle",
    "#reset-camera",
    "#import-vrm",
  ];
  const boxes = await Promise.all(
    tapTargetSelectors.map((selector) => page.locator(selector).boundingBox()),
  );
  for (const box of boxes) {
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(44);
    expect(box!.height).toBeGreaterThanOrEqual(44);
  }

  const playbackBoxes = await Promise.all(
    ["#play-pause", "#playback-speed", "#loop-toggle", "#reset-camera"].map(
      (selector) => page.locator(selector).boundingBox(),
    ),
  );
  expect(
    Math.max(...playbackBoxes.map((box) => box!.y)) -
      Math.min(...playbackBoxes.map((box) => box!.y)),
  ).toBeLessThan(2);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
});

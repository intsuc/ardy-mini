// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import {
  allowRequiredWebGpuFeatureForPreflight,
  setSliderValue,
  waitForPreviewReady,
} from "./control-helpers";
import {
  createMockModelFiles,
  developmentModelPath,
  installMockModelWorker,
  missingDevelopmentModelRoute,
  routeMockModelFiles,
} from "./model-files-fixture";

test.beforeEach(async ({ page }) => {
  await allowRequiredWebGpuFeatureForPreflight(page);
  await page.route(missingDevelopmentModelRoute, async (route) => {
    await route.fulfill({ status: 404, body: "Not found" });
  });
});

function formatModelBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

interface CameraMovementProbeState {
  initialPosition: [number, number, number] | null;
  inputs: Array<[number, number]>;
  position: [number, number, number];
}

async function installCameraMovementProbe(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const viewerModuleUrl = performance
      .getEntriesByType("resource")
      .map((entry) => entry.name)
      .find((url) => new URL(url).pathname.endsWith("/src/viewer.ts"));
    if (!viewerModuleUrl)
      throw new Error("Viewer module URL was not observed.");
    const { SkeletonViewer } = await import(/* @vite-ignore */ viewerModuleUrl);
    interface ProbeViewer {
      camera: {
        position: { x: number; y: number; z: number };
      };
    }
    interface Probe {
      initialPosition: [number, number, number] | null;
      inputs: Array<[number, number]>;
      viewer: ProbeViewer | null;
    }
    const prototype = SkeletonViewer.prototype as unknown as {
      setCameraMovement(
        this: ProbeViewer,
        forward: number,
        right: number,
      ): void;
    };
    const original = prototype.setCameraMovement;
    if (typeof original !== "function") {
      throw new Error("SkeletonViewer.setCameraMovement is unavailable.");
    }
    const probe: Probe = {
      initialPosition: null,
      inputs: [],
      viewer: null,
    };
    (
      globalThis as typeof globalThis & {
        __cameraMovementProbe?: Probe;
      }
    ).__cameraMovementProbe = probe;
    prototype.setCameraMovement = function (
      this: ProbeViewer,
      forward: number,
      right: number,
    ): void {
      probe.viewer = this;
      if (probe.initialPosition === null && (forward !== 0 || right !== 0)) {
        const { x, y, z } = this.camera.position;
        probe.initialPosition = [x, y, z];
      }
      probe.inputs.push([forward, right]);
      original.call(this, forward, right);
    };
  });
}

async function cameraMovementProbeState(
  page: Page,
): Promise<CameraMovementProbeState> {
  return page.evaluate(() => {
    interface Probe {
      initialPosition: [number, number, number] | null;
      inputs: Array<[number, number]>;
      viewer: {
        camera: {
          position: { x: number; y: number; z: number };
        };
      } | null;
    }
    const probe = (
      globalThis as typeof globalThis & {
        __cameraMovementProbe?: Probe;
      }
    ).__cameraMovementProbe;
    if (!probe?.viewer) throw new Error("Camera movement was not observed.");
    const { x, y, z } = probe.viewer.camera.position;
    return {
      initialPosition: probe.initialPosition,
      inputs: probe.inputs.map(
        ([forward, right]) => [forward, right] as [number, number],
      ),
      position: [x, y, z] as [number, number, number],
    };
  });
}

function horizontalDistance(
  first: readonly [number, number, number],
  second: readonly [number, number, number],
): number {
  return Math.hypot(second[0] - first[0], second[2] - first[2]);
}

async function waitForAnimationFrames(
  page: Page,
  count: number,
): Promise<void> {
  await page.evaluate(async (frameCount) => {
    for (let frame = 0; frame < frameCount; frame += 1) {
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    }
  }, count);
}

async function gotoReadyApp(page: Page): Promise<void> {
  const modelFiles = createMockModelFiles();
  await page.unroute(missingDevelopmentModelRoute);
  await routeMockModelFiles(page, modelFiles);
  await installMockModelWorker(page, modelFiles.manifest);
  await page.goto("/");

  const startupDialog = page.getByRole("alertdialog");
  await expect(startupDialog).toBeVisible();
  await startupDialog
    .getByRole("button", {
      name: /^(?:Download model|Resume download|Resume setup|Try again)$/,
    })
    .click();
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(startupDialog).toHaveCount(0);
}

async function openSettings(
  page: Page,
  tab: "motion" | "view" = "motion",
): Promise<void> {
  const trigger = page.locator("#settings-trigger");
  await expect(trigger).toBeVisible();
  if ((await trigger.getAttribute("aria-expanded")) !== "true") {
    await trigger.click();
  }
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#preview-settings")).toBeVisible();
  const tabTrigger = page.getByRole("tab", {
    name: tab === "motion" ? "Motion" : "View",
  });
  await tabTrigger.click();
  await expect(tabTrigger).toHaveAttribute("aria-selected", "true");
}

async function openPromptExamples(page: Page): Promise<void> {
  const trigger = page.getByRole("combobox", {
    name: "Choose an example prompt",
    exact: true,
  });
  await expect(trigger).toBeVisible();
  await trigger.click();
  await expect(
    page.getByRole("combobox", {
      name: "Search example prompts",
      exact: true,
    }),
  ).toBeVisible();
}

test("renders one preview-first workspace with responsive settings", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await gotoReadyApp(page);

  await expect
    .poll(() =>
      page.evaluate(() => globalThis.crossOriginIsolated).catch(() => false),
    )
    .toBe(true);
  const viewportPanel = page.locator("#viewport-panel");
  const viewportPanelBox = await viewportPanel.boundingBox();
  expect(viewportPanelBox).not.toBeNull();
  expect(viewportPanelBox!.x).toBe(0);
  expect(viewportPanelBox!.y).toBe(0);
  expect(viewportPanelBox!.width).toBeCloseTo(1440, 1);
  expect(viewportPanelBox!.height).toBeCloseTo(900, 1);
  await expect(page.locator("#generator-panel")).toHaveCount(0);
  await expect(page.locator("#sidebar-toggle")).toHaveCount(0);
  await expect(page.locator("#model-cache")).toHaveCount(0);

  await expect(
    page.getByRole("textbox", {
      name: "Motion description",
      exact: true,
    }),
  ).toBeEditable();
  await expect(page.locator("header")).toHaveCount(0);
  for (const removedText of [
    "ARDY Mini",
    "Input",
    "Output",
    "3D preview",
    "No motion",
    "No motion loaded",
    "Load the Core40 model, enter a prompt, then generate.",
  ]) {
    await expect(
      page.locator("body").getByText(removedText, { exact: true }),
    ).toHaveCount(0);
  }
  await expect(page.locator("#motion-badge")).toHaveCount(0);
  await expect(page.locator("#runtime-metric")).toHaveCount(0);
  await expect(
    page.getByText("Clear, typo-free English. Apply updates while streaming.", {
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(page.locator("#privacy-badge")).toHaveCount(0);
  await expect(page.locator("#gpu-badge")).toHaveCount(0);
  await expect(page.locator("#isolation-badge")).toHaveCount(0);
  await expect(page.locator("#backend")).toHaveCount(0);
  await expect(page.locator("#model-cache-state")).toHaveCount(0);
  await expect(page.locator("#model-runtime-state")).toHaveCount(0);
  await expect(page.locator("#download-model")).toHaveCount(0);
  await expect(page.getByText("20 FPS", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Core40", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Runtime notes", { exact: true })).toHaveCount(0);
  await expect(page.locator("#runtime-settings")).toHaveCount(0);

  await expect(page.locator("#generate")).toBeEnabled();
  await expect(page.locator("#generation-actions-menu")).toBeDisabled();
  const [generateBox, generationMenuBox] = await Promise.all([
    page.locator("#generate").boundingBox(),
    page.locator("#generation-actions-menu").boundingBox(),
  ]);
  expect(generateBox).not.toBeNull();
  expect(generationMenuBox).not.toBeNull();
  expect(generationMenuBox!.y).toBeCloseTo(generateBox!.y, 1);
  expect(generationMenuBox!.height).toBeCloseTo(generateBox!.height, 1);
  const [promptGroupBox, playbackBarBox] = await Promise.all([
    page.locator('[data-slot="input-group"]:has(#prompt)').boundingBox(),
    page.locator(".playback-bar").boundingBox(),
  ]);
  expect(promptGroupBox).not.toBeNull();
  expect(playbackBarBox).not.toBeNull();
  expect(
    playbackBarBox!.y - promptGroupBox!.y - promptGroupBox!.height,
  ).toBeLessThanOrEqual(12);
  for (const selector of [
    "#apply-prompt",
    "#stream-generation",
    "#duration",
    "#loop-toggle",
    "#new-session",
    "#import-session",
    "#export-session",
    "#export-motion",
    "#show-mesh",
    "#show-reference",
    "#import-reference",
    "#loading-overlay",
    "#camera-hint",
  ]) {
    await expect(page.locator(selector)).toHaveCount(0);
  }

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

  await expect(page.locator("#preview-settings")).toBeHidden();
  await expect(page.locator("#preview-settings")).toHaveAttribute(
    "data-slot",
    "popover-content",
  );
  await expect(page.locator("#target-buffer")).toHaveAttribute(
    "data-slot",
    "slider",
  );
  await expect(page.locator("#button-shortcut")).toHaveCount(0);
  await expect(page.locator("#generate")).toHaveAttribute(
    "aria-keyshortcuts",
    "Control+Enter Meta+Enter",
  );
  await expect(page.locator("#preview-diagnostics")).toBeHidden();
  await openSettings(page, "view");
  const vrmAvatarLegend = page.locator('[data-slot="field-legend"]', {
    hasText: "VRM avatar",
  });
  const displayLegend = page.locator('[data-slot="field-legend"]', {
    hasText: "Display",
  });
  await expect(vrmAvatarLegend).toBeVisible();
  await expect(vrmAvatarLegend).toHaveAttribute("data-variant", "label");
  await expect
    .poll(async () => {
      const [vrmColor, displayColor] = await Promise.all([
        vrmAvatarLegend.evaluate((element) => getComputedStyle(element).color),
        displayLegend.evaluate((element) => getComputedStyle(element).color),
      ]);
      return vrmColor === displayColor;
    })
    .toBe(true);
  await expect(page.locator("#import-vrm")).toBeEnabled();
  await expect(page.locator("#import-vrm svg")).toHaveCount(0);
  await expect(page.getByText("Foot contacts", { exact: true })).toBeVisible();
  await expect(page.getByText("Orientations", { exact: true })).toBeVisible();
  for (const selector of [
    "#show-vrm",
    "#show-skeleton",
    "#show-contacts",
    "#show-orientations",
    "#show-trajectory",
  ]) {
    await expect(page.locator(selector)).toHaveAttribute(
      "data-slot",
      "checkbox",
    );
  }
  await expect(page.locator("#show-orientations")).toBeChecked();
  await expect(page.getByText("Body proxy", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Reference motion", { exact: true })).toHaveCount(
    0,
  );
  await expect(page.locator("#clear-model-cache")).toBeVisible();
});

test("blocks model loading with a non-dismissible dialog when WebGPU is unavailable", async ({
  page,
}) => {
  const modelFiles = createMockModelFiles();
  await page.unroute(missingDevelopmentModelRoute);
  await routeMockModelFiles(page, modelFiles);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: undefined,
    });
  });
  await page.goto("/");

  const unavailableReason =
    "Use a browser and device that support WebGPU, then reload the page.";
  const dialog = page.getByRole("alertdialog", {
    name: "WebGPU is required",
  });

  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText(unavailableReason);
  await expect(page.getByText(unavailableReason, { exact: true })).toHaveCount(
    1,
  );
  await expect(
    page.getByRole("alertdialog", {
      name: "Download model files?",
    }),
  ).toHaveCount(0);
  await expect(page.locator("#confirm-model-download")).toHaveCount(0);
  const workspace = page.locator("main.workspace");
  await expect(workspace).toHaveAttribute(
    "data-ready",
    "false",
  );
  await expect(workspace).toBeVisible();
  await expect(workspace).toHaveAttribute("inert", "");
  await expect(workspace).toHaveAttribute("aria-hidden", "true");

  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
  await page.mouse.click(2, 2);
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button")).toHaveCount(0);
});

test("uses the worker's FP32 capability result even when main-thread preflight reports shader-f16", async ({
  page,
}) => {
  const fp16ModelFiles = createMockModelFiles({ variant: "fp16" });
  const fp32ModelFiles = createMockModelFiles({ variant: "fp32" });
  expect(fp32ModelFiles.transportSizeBytes).toBeGreaterThan(
    fp16ModelFiles.transportSizeBytes,
  );

  const modelRequests: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.includes("/models/ardy-minilm-core40-browser-v1/")) {
      modelRequests.push(path);
    }
  });
  await page.unroute(missingDevelopmentModelRoute);
  await routeMockModelFiles(page, fp32ModelFiles);
  await installMockModelWorker(page, fp32ModelFiles.manifest);

  await page.goto("/");
  expect(
    await page.evaluate(
      () =>
        (
          globalThis as typeof globalThis & {
            __ardyE2eMainPreflightShaderF16?: boolean;
          }
        ).__ardyE2eMainPreflightShaderF16,
    ),
  ).toBe(true);

  const startupDialog = page.getByRole("alertdialog", {
    name: "Download model files?",
  });
  await expect(startupDialog).toBeVisible();
  await expect(
    page.getByRole("alertdialog", {
      name: "WebGPU is required",
    }),
  ).toHaveCount(0);
  await expect(startupDialog).toContainText(
    `ARDY Mini needs a ${formatModelBytes(fp32ModelFiles.transportSizeBytes)} model download.`,
  );
  await expect(startupDialog).not.toContainText(
    /(?:FP16|FP32|shader-f16|precision)/i,
  );
  expect(modelRequests).toContain(
    `${developmentModelPath("fp32")}model.json.gz`,
  );
  expect(
    modelRequests.some((path) =>
      path.startsWith(developmentModelPath("fp16")),
    ),
  ).toBe(false);

  await page.locator("#confirm-model-download").click();
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "true",
  );
  expect(
    modelRequests.filter((path) =>
      path.startsWith(developmentModelPath("fp32")) &&
      !path.endsWith("model.json.gz"),
    ),
  ).toHaveLength(5);
  expect(await page.locator("body").innerText()).not.toMatch(
    /(?:FP16|FP32|shader-f16|precision)/i,
  );
});

test("gates startup through model download and manages the browser cache", async ({
  page,
}) => {
  const modelFiles = createMockModelFiles();
  const modelRequests: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.includes("/models/ardy-minilm-core40-browser-v1/")) {
      modelRequests.push(path);
    }
  });
  await page.unroute(missingDevelopmentModelRoute);
  await routeMockModelFiles(page, modelFiles);
  await installMockModelWorker(page, modelFiles.manifest);

  await page.goto("/");

  const startupDialog = page.getByRole("alertdialog");
  await expect(
    startupDialog.getByRole("heading", {
      name: "Download model files?",
    }),
  ).toBeVisible();
  await expect(startupDialog).toContainText(
    `ARDY Mini needs a ${formatModelBytes(modelFiles.transportSizeBytes)} model download.`,
  );
  expect(modelRequests).toContain(
    `${developmentModelPath("fp16")}model.json.gz`,
  );
  await expect(
    startupDialog.getByRole("button", {
      name: "Not now",
      exact: true,
    }),
  ).toHaveCount(0);
  const workspace = page.locator("main.workspace");
  await expect(workspace).toHaveAttribute(
    "data-ready",
    "false",
  );
  await expect(workspace).toBeVisible();
  await expect(workspace).toHaveAttribute("inert", "");
  await expect(workspace).toHaveAttribute("aria-hidden", "true");
  const startupOverlay = page.locator(
    '[data-slot="alert-dialog-overlay"]',
  );
  await expect(startupOverlay).toBeVisible();
  await expect(startupOverlay).toHaveClass(/bg-black\/10/);
  await expect
    .poll(() =>
      startupOverlay.evaluate(
        (element) => getComputedStyle(element).backdropFilter,
      ),
    )
    .toContain("blur(");
  expect(
    modelRequests.filter((path) => !path.endsWith("model.json.gz")),
  ).toEqual([]);

  await page.evaluate(async () => {
    const { modelUiControl } =
      await import("/src/ui-control-store.ts");
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 1,
      totalFiles: 5,
      cachedBytes: 1_024,
      totalBytes: 4_096,
    });
  });
  await expect(
    startupDialog.getByRole("heading", {
      name: "Resume model download?",
    }),
  ).toBeVisible();
  await expect(page.locator("#confirm-model-download")).toHaveText(
    "Resume download",
  );
  const clearPartialCache = page.locator(
    "#clear-partial-model-cache",
  );
  await expect(clearPartialCache).toBeVisible();
  await clearPartialCache.click();
  await expect(
    startupDialog.getByRole("heading", {
      name: "Download model files?",
    }),
  ).toBeVisible();
  await expect(clearPartialCache).toHaveCount(0);

  await page.evaluate(async () => {
    const { modelUiControl } =
      await import("/src/ui-control-store.ts");
    modelUiControl.dispatch({ type: "cache-check-started" });
  });
  await expect(
    startupDialog.getByRole("heading", {
      name: "Checking this browser",
    }),
  ).toBeVisible();
  const checkingSpinner = startupDialog.locator(
    '[data-slot="spinner"]',
  );
  await expect(checkingSpinner).toBeVisible();
  const [checkingDialogBox, checkingSpinnerBox] = await Promise.all([
    startupDialog.boundingBox(),
    checkingSpinner.boundingBox(),
  ]);
  expect(checkingDialogBox).not.toBeNull();
  expect(checkingSpinnerBox).not.toBeNull();
  expect(
    Math.abs(
      checkingSpinnerBox!.x +
        checkingSpinnerBox!.width / 2 -
        (checkingDialogBox!.x + checkingDialogBox!.width / 2),
    ),
  ).toBeLessThanOrEqual(1);

  await page.evaluate(async () => {
    const { modelUiControl } =
      await import("/src/ui-control-store.ts");
    modelUiControl.dispatch({
      type: "cache-error",
      operation: "initialization",
    });
  });
  await expect(
    startupDialog.getByRole("heading", {
      name: "ARDY Mini couldn’t start",
    }),
  ).toBeVisible();
  await expect(startupDialog).toContainText(
    "could not be verified or initialized for WebGPU",
  );
  await expect(page.locator("#confirm-model-download")).toHaveText(
    "Try again",
  );

  await page.locator("#confirm-model-download").click();
  await expect(startupDialog).toBeVisible();
  await expect(page.locator("#model-download-progress")).toBeVisible();
  await expect(page.locator("#cancel-model-download")).toBeVisible();
  await expect(
    startupDialog.getByRole("heading", {
      name: /^(?:Downloading model files|Preparing model)$/,
    }),
  ).toBeVisible();
  await expect(
    startupDialog.getByRole("heading", {
      name: "Preparing model",
    }),
  ).toBeVisible();
  const preparationProgress = page.locator(
    "#model-preparation-progress",
  );
  await Promise.all([
    expect(preparationProgress).toBeVisible(),
    expect(
      preparationProgress.locator(
        '[data-slot="progress-track"]',
      ),
    ).toBeVisible(),
    expect(
      preparationProgress.locator(
        '[data-slot="progress-label"], [data-slot="progress-value"]',
      ),
    ).toHaveCount(0),
    expect(
      startupDialog.locator('[data-slot="spinner"]'),
    ).toHaveCount(0),
    expect(page.locator("#cancel-model-download")).toHaveCount(0),
  ]);
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(startupDialog).toHaveCount(0);
  await expect(page.locator("#model-download-progress")).toHaveCount(0);
  const payloadRequestCount = modelRequests.filter(
    (path) => !path.endsWith("model.json.gz"),
  ).length;
  expect(payloadRequestCount).toBe(5);

  await page.reload();
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await expect(page.getByRole("alertdialog")).toHaveCount(0);
  expect(
    modelRequests.filter((path) => !path.endsWith("model.json.gz")),
  ).toHaveLength(payloadRequestCount);

  await openSettings(page);
  const clearCache = page.locator("#clear-model-cache");
  await expect(clearCache).toBeVisible();
  await clearCache.click();
  const clearDialog = page.getByRole("alertdialog", {
    name: "Clear cached model files?",
  });
  const cancelClear = clearDialog.getByRole("button", {
    name: "Cancel",
    exact: true,
  });
  await expect(clearDialog).toBeVisible();
  await expect(cancelClear).toBeFocused();
  await clearDialog
    .getByRole("button", { name: "Clear model cache", exact: true })
    .click();
  await expect(clearDialog).toBeHidden();
  await expect(clearCache).toHaveText("Cache cleared");
  await expect(clearCache).toBeDisabled();
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "true",
  );

  await page.reload();
  await expect(
    page.getByRole("alertdialog", {
      name: "Download model files?",
    }),
  ).toBeVisible();
  await expect(page.locator("main.workspace")).toHaveAttribute(
    "data-ready",
    "false",
  );
});

test("rejects renderer initialization instead of using a WebGL fallback", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript(() => {
    const requestedContexts: string[] = [];
    type ContextGetter = (contextId: string, ...options: unknown[]) => unknown;
    const canvasPrototype = HTMLCanvasElement.prototype as unknown as {
      getContext: ContextGetter;
    };
    const originalGetContext = canvasPrototype.getContext;
    canvasPrototype.getContext = function (contextId, ...options): unknown {
      requestedContexts.push(contextId);
      return Reflect.apply(originalGetContext, this, [contextId, ...options]);
    };
    Object.defineProperty(globalThis, "__rendererContexts", {
      configurable: true,
      value: requestedContexts,
    });
    let adapterRequestCount = 0;
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: {
        requestAdapter: async () => {
          adapterRequestCount += 1;
          if (adapterRequestCount > 1) return null;
          return {
            features: {
              has: (feature: string) => feature === "shader-f16",
            },
          };
        },
      },
    });
  });
  await page.goto("/");

  const dialog = page.getByRole("alertdialog", {
    name: "WebGPU preview unavailable",
  });
  await expect(dialog).toBeVisible();
  await expect(page.locator("#error-banner")).toHaveCount(0);
  const requestedContexts = await page.evaluate(
    () =>
      (
        globalThis as typeof globalThis & {
          __rendererContexts?: string[];
        }
      ).__rendererContexts ?? [],
  );
  expect(requestedContexts).not.toContain("webgl");
  expect(requestedContexts).not.toContain("webgl2");
  expect(pageErrors).toEqual([]);
});

test("retains the square Lyra treatment on standard shadcn surfaces", async ({
  page,
}) => {
  await gotoReadyApp(page);
  await openSettings(page);

  const radii = await page.evaluate(() =>
    [
      "#generate",
      "#prompt",
      '[data-slot="combobox-trigger"]',
      "#seed",
      "#playback-speed",
      "#settings-trigger",
      "#reset-camera",
    ].map((selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing Lyra surface: ${selector}`);
      }
      return getComputedStyle(element).borderRadius;
    }),
  );

  expect(radii).toEqual(["0px", "0px", "0px", "0px", "0px", "0px", "0px"]);
});

test("renders a camera-relative pristine ground while shadows follow motion", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  const shaderConsoleErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      /(?:WebGPU|WGSL|shader|validation)/i.test(message.text())
    ) {
      shaderConsoleErrors.push(message.text());
    }
  });
  await page.goto("/");

  const groundState = await page.evaluate(async () => {
    const viewerModule = await import("/src/viewer.ts");
    const { createVrmRetargetPlan } = await import("/src/vrm-retarget.ts");
    const { CORE27_JOINT_COUNT, CORE27_SKELETON, SkeletonViewer } =
      viewerModule;
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);

    const viewer = await SkeletonViewer.create(canvas);
    try {
      interface DebugObject {
        readonly name: string;
        readonly type: string;
        readonly isMesh?: boolean;
        readonly position: { readonly x: number; readonly z: number };
        readonly scale: { readonly x: number; readonly y: number };
        readonly receiveShadow?: boolean;
        readonly geometry?: {
          readonly type: string;
          readonly parameters: {
            readonly width?: number;
            readonly height?: number;
          };
        };
        readonly material?: {
          readonly roughness?: number;
          readonly metalness?: number;
          readonly userData?: Record<string, unknown>;
        };
        readonly target?: { readonly name: string };
        readonly shadow?: {
          readonly camera: {
            readonly left: number;
            readonly right: number;
            readonly top: number;
            readonly bottom: number;
          };
        };
      }
      const internal = viewer as unknown as {
        scene: {
          getObjectByName(name: string): DebugObject | undefined;
          traverse(callback: (object: DebugObject) => void): void;
        };
        camera: {
          readonly far: number;
          readonly position: {
            readonly x: number;
            readonly y: number;
            readonly z: number;
          };
        };
        controls: {
          readonly maxDistance: number;
          readonly target: {
            readonly x: number;
            readonly y: number;
            readonly z: number;
          };
        };
        renderer: {
          readonly isWebGPURenderer: boolean;
          readonly backend: {
            readonly isWebGPUBackend?: boolean;
            readonly isWebGLBackend?: boolean;
            readonly device?: GPUDevice;
          };
          readonly shadowMap: { readonly enabled: boolean };
          readonly debug: {
            getShaderAsync(
              scene: unknown,
              camera: unknown,
              object: unknown,
            ): Promise<{
              readonly fragmentShader: string | null;
              readonly vertexShader: string | null;
            }>;
          };
          onError(info: string | { readonly message?: string }): void;
          compileAsync(scene: unknown, camera: unknown): Promise<unknown>;
        };
        renderPipeline: {
          readonly pipeline: {
            readonly outputColorTransform: boolean;
          };
          render(): void;
        };
        vrm: unknown;
        vrmRetargetPlan: unknown;
        vrmRoot: {
          visible: boolean;
          readonly position: { readonly x: number; readonly z: number };
        };
      };
      const rendererErrors: string[] = [];
      internal.renderer.onError = (info) => {
        rendererErrors.push(
          typeof info === "string"
            ? info
            : (info.message ?? JSON.stringify(info)),
        );
      };

      const rootX = 24;
      const rootZ = -18;
      const positions = new Float32Array(CORE27_JOINT_COUNT * 3);
      for (let joint = 0; joint < CORE27_JOINT_COUNT; joint += 1) {
        const offset = joint * 3;
        positions[offset] = rootX;
        positions[offset + 1] = 0.04 * joint;
        positions[offset + 2] = rootZ;
      }
      viewer.setMotion(
        {
          skeleton: CORE27_SKELETON,
          positions,
          positionsShape: [1, CORE27_JOINT_COUNT, 3],
          frameCount: 1,
          fps: 20,
        },
        { playing: false, resetCamera: true },
      );
      await new Promise<void>((resolve) => window.setTimeout(resolve, 750));

      const ground = internal.scene.getObjectByName(
        "camera-relative-ground-grid",
      );
      const rig = internal.scene.getObjectByName("shadow-follow-rig");
      const light = internal.scene.getObjectByName("shadow-key-light");
      const snapshot = () => ({
        ground: ground ? { x: ground.position.x, z: ground.position.z } : null,
        rig: rig ? { x: rig.position.x, z: rig.position.z } : null,
        camera: {
          x: internal.camera.position.x,
          z: internal.camera.position.z,
        },
        target: {
          x: internal.controls.target.x,
          z: internal.controls.target.z,
        },
      });
      const initial = snapshot();
      const currentVrmState = () => ({
        rig: rig ? { x: rig.position.x, z: rig.position.z } : null,
        rootOffset: {
          x: internal.vrmRoot.position.x,
          z: internal.vrmRoot.position.z,
        },
      });
      let vrm1State;
      let vrm0State;
      let hiddenVrmRig;
      let reshownVrmRig;
      const fakeHips = {
        position: {
          fromArray(_position: readonly number[]): void {},
        },
      };
      try {
        internal.vrm = {
          humanoid: {
            getNormalizedBoneNode(name: string) {
              return name === "hips" ? fakeHips : null;
            },
          },
          update(): void {},
        };
        internal.vrmRetargetPlan = createVrmRetargetPlan(CORE27_SKELETON, {
          presentBones: ["hips"],
          sourceHipsHeight: 1,
          targetHipsHeight: 0.5,
          metaVersion: "1",
        });
        internal.vrmRoot.visible = true;
        viewer.seek(0);
        vrm1State = currentVrmState();
        internal.vrmRetargetPlan = createVrmRetargetPlan(CORE27_SKELETON, {
          presentBones: ["hips"],
          sourceHipsHeight: 1,
          targetHipsHeight: 0.5,
          metaVersion: "0",
        });
        viewer.seek(0);
        vrm0State = currentVrmState();
        viewer.setVrmVisible(false);
        hiddenVrmRig = currentVrmState().rig;
        viewer.setVrmVisible(true);
        reshownVrmRig = currentVrmState().rig;
      } finally {
        internal.vrm = null;
        internal.vrmRetargetPlan = null;
        internal.vrmRoot.visible = false;
      }
      viewer.orbit(3, 1);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
      const afterOrbit = snapshot();
      for (let step = 0; step < 48; step += 1) {
        viewer.moveCamera(1, 0);
      }
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
      const afterMove = snapshot();

      const gridHelpers: string[] = [];
      let namedGroundCount = 0;
      internal.scene.traverse((object) => {
        if (object.type === "GridHelper") {
          gridHelpers.push(object.name);
        }
        if (object.name === "camera-relative-ground-grid") {
          namedGroundCount += 1;
        }
      });

      const device = internal.renderer.backend.device;
      device?.pushErrorScope("validation");
      let validationError: string | null = null;
      let groundShader: {
        readonly fragmentShader: string | null;
        readonly vertexShader: string | null;
      } | null = null;
      try {
        await internal.renderer.compileAsync(internal.scene, internal.camera);
        if (ground) {
          groundShader = await internal.renderer.debug.getShaderAsync(
            internal.scene,
            internal.camera,
            ground,
          );
        }
        internal.renderPipeline.render();
        await device?.queue.onSubmittedWorkDone();
      } finally {
        validationError = (await device?.popErrorScope())?.message ?? null;
      }
      const width =
        (ground?.geometry?.parameters.width ?? 0) *
        Math.abs(ground?.scale.x ?? 0);
      const height =
        (ground?.geometry?.parameters.height ?? 0) *
        Math.abs(ground?.scale.y ?? 0);

      return {
        renderer: {
          isWebGPURenderer: internal.renderer.isWebGPURenderer,
          isWebGPUBackend: internal.renderer.backend.isWebGPUBackend === true,
          isWebGLBackend: internal.renderer.backend.isWebGLBackend === true,
          outputColorTransform:
            internal.renderPipeline.pipeline.outputColorTransform,
          generatedWgsl:
            Boolean(groundShader?.vertexShader?.includes("@vertex")) &&
            Boolean(groundShader?.fragmentShader?.includes("@fragment")),
        },
        rendererShadows: internal.renderer.shadowMap.enabled,
        ground: ground
          ? {
              geometry: ground.geometry?.type,
              width,
              height,
              receiveShadow: ground.receiveShadow,
              roughness: ground.material?.roughness,
              metalness: ground.material?.metalness,
              pristineGrid: ground.material?.userData?.pristineGrid,
            }
          : null,
        requiredDiameter:
          2 * (internal.camera.far + internal.controls.maxDistance + 5),
        initial,
        vrm1State,
        vrm0State,
        hiddenVrmRig,
        reshownVrmRig,
        afterOrbit,
        afterMove,
        light: light
          ? {
              target: light.target?.name,
              left: light.shadow?.camera.left,
              right: light.shadow?.camera.right,
              top: light.shadow?.camera.top,
              bottom: light.shadow?.camera.bottom,
            }
          : null,
        namedGroundCount,
        gridHelpers,
        rendererErrors,
        validationError,
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  expect(groundState.renderer).toEqual({
    isWebGPURenderer: true,
    isWebGPUBackend: true,
    isWebGLBackend: false,
    outputColorTransform: false,
    generatedWgsl: true,
  });
  expect(groundState.rendererShadows).toBe(true);
  expect(groundState.namedGroundCount).toBe(1);
  expect(groundState.gridHelpers).toEqual([]);
  expect(groundState.ground).toMatchObject({
    geometry: "PlaneGeometry",
    receiveShadow: true,
    roughness: 1,
    metalness: 0,
    pristineGrid: true,
  });
  expect(groundState.ground!.width).toBeGreaterThanOrEqual(
    groundState.requiredDiameter,
  );
  expect(groundState.ground!.height).toBeGreaterThanOrEqual(
    groundState.requiredDiameter,
  );
  expect(groundState.initial.rig).toEqual({ x: 24, z: -18 });
  expect(groundState.vrm1State).toEqual({
    rig: { x: 24, z: -18 },
    rootOffset: { x: 12, z: -9 },
  });
  expect(groundState.vrm0State).toEqual({
    rig: { x: 24, z: -18 },
    rootOffset: { x: 12, z: -9 },
  });
  expect(groundState.hiddenVrmRig).toEqual({ x: 24, z: -18 });
  expect(groundState.reshownVrmRig).toEqual({ x: 24, z: -18 });
  expect(groundState.afterOrbit.rig).toEqual({ x: 24, z: -18 });
  expect(groundState.afterMove.rig).toEqual({ x: 24, z: -18 });

  const snappedTarget = (value: number): number => Math.round(value / 5) * 5;
  for (const state of [
    groundState.initial,
    groundState.afterOrbit,
    groundState.afterMove,
  ]) {
    expect(state.ground?.x).toBeCloseTo(snappedTarget(state.target.x));
    expect(state.ground?.z).toBeCloseTo(snappedTarget(state.target.z));
  }
  expect(groundState.afterOrbit.ground).toEqual(groundState.initial.ground);
  expect(groundState.afterMove.ground).not.toEqual(groundState.initial.ground);
  expect(groundState.afterMove.camera).not.toEqual(groundState.initial.camera);
  expect(groundState.light).toEqual({
    target: "shadow-key-light-target",
    left: -4,
    right: 4,
    top: 4,
    bottom: -4,
  });
  expect(groundState.rendererErrors).toEqual([]);
  expect(groundState.validationError).toBeNull();
  expect(shaderConsoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("follows the root while preserving manual camera composition", async ({
  page,
}) => {
  await page.goto("/");

  const cameraState = await page.evaluate(async () => {
    const { CORE27_JOINT_COUNT, CORE27_SKELETON, SkeletonViewer } =
      await import("/src/viewer.ts");
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);
    const viewer = await SkeletonViewer.create(canvas);
    try {
      const positions = new Float32Array(2 * CORE27_JOINT_COUNT * 3);
      for (let frame = 0; frame < 2; frame += 1) {
        for (let joint = 0; joint < CORE27_JOINT_COUNT; joint += 1) {
          const offset = (frame * CORE27_JOINT_COUNT + joint) * 3;
          positions[offset] = frame * 6;
          positions[offset + 1] = joint * 0.04;
          positions[offset + 2] = frame * -4;
        }
      }
      viewer.setMotion(
        {
          skeleton: CORE27_SKELETON,
          positions,
          positionsShape: [2, CORE27_JOINT_COUNT, 3],
          frameCount: 2,
          fps: 20,
        },
        { playing: false, resetCamera: true },
      );
      await new Promise<void>((resolve) => window.setTimeout(resolve, 750));
      const internal = viewer as unknown as {
        camera: { position: { x: number; y: number; z: number } };
        controls: { target: { x: number; y: number; z: number } };
      };
      const snapshot = () => ({
        camera: [
          internal.camera.position.x,
          internal.camera.position.y,
          internal.camera.position.z,
        ],
        target: [
          internal.controls.target.x,
          internal.controls.target.y,
          internal.controls.target.z,
        ],
      });
      const initial = snapshot();
      viewer.seek(1);
      const followed = snapshot();
      viewer.moveCamera(1, 0);
      const moved = snapshot();
      viewer.resetCamera({ animated: false });
      const reset = snapshot();
      viewer.setReducedMotion(true);
      viewer.setPlaying(true);
      return {
        initial,
        followed,
        moved,
        reset,
        playing: viewer.getPlaybackState().playing,
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  const delta = (after: number[], before: number[]): number[] =>
    after.map((value, index) => value - before[index]);
  const relative = (state: { camera: number[]; target: number[] }): number[] =>
    delta(state.camera, state.target);
  expect(
    delta(cameraState.followed.camera, cameraState.initial.camera),
  ).toEqual([6, 0, -4]);
  expect(
    delta(cameraState.followed.target, cameraState.initial.target),
  ).toEqual([6, 0, -4]);
  relative(cameraState.followed).forEach((value, index) => {
    expect(value).toBeCloseTo(relative(cameraState.initial)[index]);
  });
  const manualCameraDelta = delta(
    cameraState.moved.camera,
    cameraState.followed.camera,
  );
  const manualTargetDelta = delta(
    cameraState.moved.target,
    cameraState.followed.target,
  );
  expect(manualCameraDelta[1]).toBeCloseTo(0);
  expect(
    Math.hypot(manualCameraDelta[0], manualCameraDelta[2]),
  ).toBeGreaterThan(0);
  manualTargetDelta.forEach((value, index) => {
    expect(value).toBeCloseTo(manualCameraDelta[index]);
  });
  expect(cameraState.reset.target[0]).toBeCloseTo(6);
  expect(cameraState.reset.target[2]).toBeCloseTo(-4);
  expect(cameraState.playing).toBe(false);
});

test("exposes deterministic inputs and enforces the prompt contract", async ({
  page,
}) => {
  await gotoReadyApp(page);

  const prompt = page.getByRole("textbox", {
    name: "Motion description",
    exact: true,
  });
  await expect(prompt.locator("xpath=..")).toHaveAttribute(
    "data-slot",
    "input-group",
  );
  await expect(page.locator("#seed").locator("xpath=..")).toHaveAttribute(
    "data-slot",
    "input-group",
  );
  await expect(prompt).toHaveValue(
    "A person walks forward, then waves with their right hand.",
  );
  await expect(page.locator("#prompt-label")).toHaveClass(/sr-only/);
  await expect(page.locator("#prompt-count")).toHaveCount(0);
  await expect(page.locator("#generation-progress")).toHaveCount(0);
  await expect(page.locator("#cancel-generation")).toHaveCount(0);
  await expect(page.locator("#prompt-error")).toBeHidden();
  await expect(page.locator("#generate").locator("xpath=..")).toHaveAttribute(
    "data-slot",
    "button-group",
  );

  await prompt.fill("");
  await page.locator("#generation-form").evaluate((form) => {
    form.dispatchEvent(
      new SubmitEvent("submit", {
        bubbles: true,
        cancelable: true,
      }),
    );
  });
  await expect(page.locator("#prompt-error")).toHaveText(
    "Describe the motion you want to generate.",
  );
  await expect(page.locator("#prompt-error")).toBeVisible();
  await prompt.fill(
    "A person walks forward, then waves with their right hand.",
  );
  await expect(page.locator("#prompt-error")).toBeHidden();

  await openSettings(page);
  await setSliderValue(page, "#target-buffer", 120);
  await expect(page.locator("#target-buffer-output")).toHaveText("6 seconds");
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toHaveValue("2");

  const randomizeSeed = page.locator("#randomize-seed");
  const randomizeSeedIcon = randomizeSeed.locator("svg");
  await expect(randomizeSeedIcon).toHaveCount(1);
  await expect(randomizeSeedIcon).not.toHaveAttribute("data-icon");
  const [randomizeSeedBox, randomizeSeedIconBox] = await Promise.all([
    randomizeSeed.boundingBox(),
    randomizeSeedIcon.boundingBox(),
  ]);
  expect(randomizeSeedBox).not.toBeNull();
  expect(randomizeSeedIconBox).not.toBeNull();
  expect(
    Math.abs(
      randomizeSeedBox!.x +
        randomizeSeedBox!.width / 2 -
        (randomizeSeedIconBox!.x + randomizeSeedIconBox!.width / 2),
    ),
  ).toBeLessThanOrEqual(0.5);
  expect(
    Math.abs(
      randomizeSeedBox!.y +
        randomizeSeedBox!.height / 2 -
        (randomizeSeedIconBox!.y + randomizeSeedIconBox!.height / 2),
    ),
  ).toBeLessThanOrEqual(0.5);

  await openPromptExamples(page);
  const promptExample = page.getByRole("combobox", {
    name: "Search example prompts",
    exact: true,
  });
  await expect(promptExample).toHaveAttribute("placeholder", "Search examples");
  const promptExampleContent = page.locator('[data-slot="combobox-content"]');
  await expect(promptExampleContent).toBeVisible();
  const promptExampleContentBox = await promptExampleContent.boundingBox();
  expect(promptExampleContentBox).not.toBeNull();
  expect(promptExampleContentBox!.width).toBeCloseTo(288, 0);
  expect(promptExampleContentBox!.x).toBeGreaterThanOrEqual(0);
  expect(
    promptExampleContentBox!.x + promptExampleContentBox!.width,
  ).toBeLessThanOrEqual(page.viewportSize()!.width);
  const promptOptions = promptExampleContent.getByRole("option");
  await expect(promptOptions).toHaveCount(100);
  await expect(promptOptions.first()).toHaveAttribute(
    "data-slot",
    "combobox-item",
  );

  const promptBeforeSearch = await prompt.inputValue();
  await promptExample.click();
  await expect(promptExample).toBeFocused();
  await page.keyboard.type("Joyful dance");
  await expect(promptExample).toBeFocused();
  await expect(promptExample).toHaveValue("Joyful dance");
  await expect(prompt).toHaveValue(promptBeforeSearch);
  await expect(promptOptions).toHaveCount(1);
  await expect(promptOptions).toHaveText("Joyful dance");
  await promptExample.press("ArrowDown");
  await promptExample.press("Enter");
  await expect(prompt).toHaveValue("A person performs a joyful dance.");

  await page.evaluate(async () => {
    const { generationActionsControl } =
      await import("/src/ui-control-store.ts");
    generationActionsControl.setState({
      menuDisabled: false,
      regenerateDisabled: false,
      newMotionDisabled: false,
    });
  });
  await page.locator("#generation-actions-menu").click();
  await expect(
    page.getByRole("menuitem", {
      name: "Regenerate from current time",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("menuitem", {
      name: "Start new motion",
      exact: true,
    }),
  ).toBeVisible();
  await page.keyboard.press("Escape");

  const validation = await page.evaluate(async () => {
    const { validateGenerationForm } = await import("/src/main.ts");
    return {
      empty: validateGenerationForm("", "2").promptError,
      multilingual: validateGenerationForm("人物が歩く。", "2").values,
      long: validateGenerationForm("a".repeat(281), "2").promptError,
      seed: validateGenerationForm("A person walks.", "-1").seedError,
      valid: validateGenerationForm("A person walks forward.", "4294967295")
        .values,
    };
  });
  expect(validation.empty).toContain("Describe the motion");
  expect(validation.multilingual).toEqual({
    prompt: "人物が歩く。",
    seed: 2,
  });
  expect(validation.long).toContain("280 characters");
  expect(validation.seed).toContain("whole-number seed");
  expect(validation.valid).toEqual({
    prompt: "A person walks forward.",
    seed: 4_294_967_295,
  });
});

test("uses the Select control to update runtime playback speed", async ({
  page,
}) => {
  await gotoReadyApp(page);
  await page.evaluate(async () => {
    const [{ SkeletonViewer }, { playbackSpeedControl }] = await Promise.all([
      import("/src/viewer.ts"),
      import("/src/ui-control-store.ts"),
    ]);
    const probe = globalThis as typeof globalThis & {
      __playbackSpeeds?: number[];
    };
    probe.__playbackSpeeds = [];
    const prototype = SkeletonViewer.prototype;
    const original = prototype.setSpeed;
    prototype.setSpeed = function (speed: number): void {
      probe.__playbackSpeeds?.push(speed);
      original.call(this, speed);
    };
    playbackSpeedControl.setState({ disabled: false });
  });

  const speed = page.getByRole("combobox", {
    name: "Playback speed",
  });
  await expect(speed).toHaveAttribute("data-slot", "select-trigger");
  await expect(page.locator("select#playback-speed")).toHaveCount(0);
  await speed.click();
  const speedContent = page.locator('[data-slot="select-content"]');
  await expect(speedContent).toBeVisible();
  const [speedBox, speedContentBox] = await Promise.all([
    speed.boundingBox(),
    speedContent.boundingBox(),
  ]);
  expect(speedBox).not.toBeNull();
  expect(speedContentBox).not.toBeNull();
  expect(Math.abs(speedContentBox!.width - speedBox!.width)).toBeLessThanOrEqual(
    2,
  );
  expect(
    Math.abs(
      speedContentBox!.x +
        speedContentBox!.width -
        (speedBox!.x + speedBox!.width),
    ),
  ).toBeLessThanOrEqual(2);
  const doubleSpeed = page.getByRole("option", { name: "2×" });
  await expect(doubleSpeed).toHaveAttribute("data-slot", "select-item");
  await doubleSpeed.click();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          (
            globalThis as typeof globalThis & {
              __playbackSpeeds?: number[];
            }
          ).__playbackSpeeds ?? [],
      ),
    )
    .toContain(2);
});

test("fits the example prompt popup within a narrow viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 270, height: 844 });
  await gotoReadyApp(page);
  await openPromptExamples(page);

  const content = page.locator('[data-slot="combobox-content"]');
  const box = await content.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.width).toBeGreaterThanOrEqual(240);
  expect(box!.width).toBeLessThanOrEqual(288);
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(270);
});

test("keeps labels, keyboard focus, and canvas controls accessible", async ({
  page,
}) => {
  await gotoReadyApp(page);

  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "ARDY browser motion workspace",
    }),
  ).toHaveClass(/sr-only/);
  await expect(
    page.getByRole("heading", { level: 2, name: "Motion preview" }),
  ).toHaveClass(/sr-only/);

  await page.locator(".skip-link").focus();
  await expect(page.locator(".skip-link")).toBeFocused();

  const settingsTrigger = page.locator("#settings-trigger");
  const settingsContent = page.locator("#preview-settings");
  const [viewportBefore, triggerBox] = await Promise.all([
    page.locator("#viewport").boundingBox(),
    settingsTrigger.boundingBox(),
  ]);
  expect(viewportBefore).not.toBeNull();
  expect(triggerBox).not.toBeNull();
  expect(triggerBox!.y - viewportBefore!.y).toBeGreaterThanOrEqual(8);
  expect(triggerBox!.y - viewportBefore!.y).toBeLessThanOrEqual(16);
  expect(
    viewportBefore!.x +
      viewportBefore!.width -
      triggerBox!.x -
      triggerBox!.width,
  ).toBeGreaterThanOrEqual(8);
  expect(
    viewportBefore!.x +
      viewportBefore!.width -
      triggerBox!.x -
      triggerBox!.width,
  ).toBeLessThanOrEqual(16);
  await expect(settingsContent).toHaveCount(1);
  await expect(settingsContent).toHaveAttribute("data-slot", "popover-content");
  await expect(settingsContent).toBeHidden();
  await settingsTrigger.click();
  await expect(settingsTrigger).toHaveAttribute("aria-expanded", "true");
  await expect(settingsContent).toBeVisible();
  await expect(
    settingsContent.locator('[data-slot="popover-title"]'),
  ).toHaveClass(/sr-only/);
  await expect(
    settingsContent.locator('[data-slot="popover-header"]'),
  ).toHaveCount(0);
  await expect(
    settingsContent.locator('[data-slot="popover-description"]'),
  ).toHaveCount(0);
  const motionTab = page.getByRole("tab", { name: "Motion" });
  const viewTab = page.getByRole("tab", { name: "View" });
  await expect(motionTab).toHaveAttribute("aria-selected", "true");
  await motionTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(viewTab).toBeFocused();
  await page.keyboard.press("ArrowLeft");
  await expect(motionTab).toBeFocused();
  await viewTab.click();
  await expect(viewTab).toHaveAttribute("aria-selected", "true");
  expect(await page.locator("#viewport").boundingBox()).toEqual(viewportBefore);
  const previewSettingsOverflow = await settingsContent.evaluate((element) => {
    const style = getComputedStyle(element);
    const borderWidth =
      Number.parseFloat(style.borderLeftWidth) +
      Number.parseFloat(style.borderRightWidth);
    return {
      overflowY: style.overflowY,
      scrollbarWidth: element.offsetWidth - element.clientWidth - borderWidth,
    };
  });
  expect(previewSettingsOverflow.overflowY).toBe("visible");
  expect(previewSettingsOverflow.scrollbarWidth).toBeLessThanOrEqual(0.5);
  await page.keyboard.press("Escape");
  await expect(settingsTrigger).toHaveAttribute("aria-expanded", "false");
  await expect(settingsContent).toBeHidden();
  await expect(settingsTrigger).toBeFocused();

  await expect(
    page.getByRole("textbox", {
      name: "Motion description",
      exact: true,
    }),
  ).toHaveCount(1);
  await expect(page.locator("#seed")).toHaveCount(1);
  await expect(page.locator("#target-buffer")).toHaveCount(1);

  await expect(page.locator("#model-runtime-state")).toHaveCount(0);
  const canvas = page.locator("#motion-canvas");
  await expect(canvas).toHaveAttribute("aria-keyshortcuts", /W A S D/);
  await canvas.focus();
  await expect(canvas).toBeFocused();
  await expect
    .poll(() =>
      canvas.evaluate((element) => getComputedStyle(element).outlineStyle),
    )
    .not.toBe("none");
  const wasdHandled = await canvas.evaluate((element) =>
    ["KeyW", "KeyA", "KeyS", "KeyD"].map((code, index) => {
      const event = new KeyboardEvent("keydown", {
        key: code.at(-1)?.toLowerCase(),
        code,
        repeat: index === 0,
        bubbles: true,
        cancelable: true,
      });
      element.dispatchEvent(event);
      return event.defaultPrevented;
    }),
  );
  expect(wasdHandled).toEqual([true, true, true, true]);
  const promptHandled = await page.locator("#prompt").evaluate((element) => {
    const event = new KeyboardEvent("keydown", {
      key: "w",
      code: "KeyW",
      bubbles: true,
      cancelable: true,
    });
    element.dispatchEvent(event);
    return event.defaultPrevented;
  });
  expect(promptHandled).toBe(false);
  await canvas.focus();
  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("=");
  await page.keyboard.press("Home");

  await expect(page.locator("#loop-toggle")).toHaveCount(0);
  const resetCamera = page.locator("#reset-camera");
  await expect(resetCamera).toHaveAttribute("data-variant", "outline");
  await expect(page.locator("#viewport").locator("#reset-camera")).toHaveCount(
    1,
  );
  await expect(
    page.locator("#playback-bar").locator("#reset-camera"),
  ).toHaveCount(0);
  const resetBorder = await resetCamera.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      style: style.borderStyle,
      width: style.borderWidth,
    };
  });
  expect(resetBorder).toEqual({ style: "solid", width: "1px" });
});

test("labels playback and preview icon buttons with tooltips", async ({
  page,
}) => {
  await gotoReadyApp(page);
  await installCameraMovementProbe(page);
  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await page.keyboard.press("w");
  await page.evaluate(async () => {
    const { CORE27_JOINT_COUNT, CORE27_SKELETON } =
      await import("/src/viewer.ts");
    interface MotionViewer {
      setMotion(
        motion: {
          skeleton: typeof CORE27_SKELETON;
          positions: Float32Array;
          positionsShape: [number, number, number];
          frameCount: number;
          fps: number;
        },
        options: { playing: boolean },
      ): void;
    }
    const probe = globalThis as typeof globalThis & {
      __cameraMovementProbe?: { viewer: MotionViewer | null };
    };
    const viewer = probe.__cameraMovementProbe?.viewer;
    if (!viewer) throw new Error("Preview viewer is unavailable.");
    const frameCount = 200;
    viewer.setMotion(
      {
        skeleton: CORE27_SKELETON,
        positions: new Float32Array(frameCount * CORE27_JOINT_COUNT * 3),
        positionsShape: [frameCount, CORE27_JOINT_COUNT, 3],
        frameCount,
        fps: 20,
      },
      { playing: false },
    );
  });
  await expect(page.locator("#play-pause")).toBeEnabled();
  await expect(page.locator("#loop-toggle")).toHaveCount(0);

  for (const [selector, label] of [
    ["#play-pause", "Play motion"],
    ["#reset-camera", "Reset camera"],
    ["#settings-trigger", "Settings"],
  ] as const) {
    const trigger = page.locator(selector);
    const tooltipTrigger = trigger.locator("xpath=..");
    const tooltip = page
      .locator('[data-slot="tooltip-content"]')
      .filter({ hasText: label });
    await expect(tooltipTrigger).toHaveAttribute(
      "data-slot",
      "tooltip-trigger",
    );
    await trigger.focus();
    await expect(tooltip).toBeVisible();

    await canvas.focus();
    await expect(tooltip).toBeHidden();
    await trigger.hover();
    await expect(tooltip).toBeVisible();
    await page.waitForTimeout(750);
    await expect(tooltip).toBeVisible();
    await page.mouse.move(0, 0);
    await expect(tooltip).toBeHidden();
  }

  const playPause = page.locator("#play-pause");
  await playPause.click();
  await expect(playPause).toHaveAccessibleName("Pause motion");
  const pauseTooltip = page
    .locator('[data-slot="tooltip-content"]')
    .filter({ hasText: "Pause motion" });
  await playPause.hover();
  await expect(pauseTooltip).toBeVisible();
  await page.mouse.move(0, 0);
});

test("moves continuously from one held W keydown and stops on keyup", async ({
  page,
}) => {
  await gotoReadyApp(page);
  await installCameraMovementProbe(page);

  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await page.keyboard.down("w");

  await expect
    .poll(async () => {
      const state = await cameraMovementProbeState(page);
      return state.initialPosition
        ? horizontalDistance(state.initialPosition, state.position)
        : 0;
    })
    .toBeGreaterThan(0.001);
  const midway = (await cameraMovementProbeState(page)).position;
  await expect
    .poll(async () =>
      horizontalDistance(
        midway,
        (await cameraMovementProbeState(page)).position,
      ),
    )
    .toBeGreaterThan(0.001);

  const heldState = await cameraMovementProbeState(page);
  expect(
    heldState.inputs.filter(([forward, right]) => forward || right),
  ).toEqual([[1, 0]]);

  await page.keyboard.up("w");
  await expect
    .poll(async () => (await cameraMovementProbeState(page)).inputs.at(-1))
    .toEqual([0, 0]);
  await waitForAnimationFrames(page, 3);
  const stopped = (await cameraMovementProbeState(page)).position;
  await waitForAnimationFrames(page, 4);
  const afterStop = (await cameraMovementProbeState(page)).position;
  expect(horizontalDistance(stopped, afterStop)).toBeLessThan(1e-8);
});

test("clears held camera movement when the window loses focus", async ({
  page,
}) => {
  await gotoReadyApp(page);
  await installCameraMovementProbe(page);

  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await page.keyboard.down("d");
  await expect
    .poll(async () => {
      const state = await cameraMovementProbeState(page);
      return state.initialPosition
        ? horizontalDistance(state.initialPosition, state.position)
        : 0;
    })
    .toBeGreaterThan(0.001);

  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await expect
    .poll(async () => (await cameraMovementProbeState(page)).inputs.at(-1))
    .toEqual([0, 0]);
  await waitForAnimationFrames(page, 3);
  const stopped = (await cameraMovementProbeState(page)).position;
  await waitForAnimationFrames(page, 4);
  const afterStop = (await cameraMovementProbeState(page)).position;
  expect(horizontalDistance(stopped, afterStop)).toBeLessThan(1e-8);

  // Clear Playwright's pressed-key state after the synthetic blur.
  await page.keyboard.up("d");
});

test("reports internal failures to the console without rendering an error panel", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  const internalErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      message.text().includes("[ARDY] Inference failed")
    ) {
      internalErrors.push(message.text());
    }
  });
  await page.addInitScript(() => {
    const NativeWorker = globalThis.Worker;
    const WorkerProxy = new Proxy(NativeWorker, {
      construct(target, argumentsList) {
        const worker = Reflect.construct(target, argumentsList) as Worker;
        (
          globalThis as typeof globalThis & {
            __ardyTestWorker?: Worker;
          }
        ).__ardyTestWorker = worker;
        return worker;
      },
    });
    Object.defineProperty(globalThis, "Worker", {
      configurable: true,
      value: WorkerProxy,
    });
  });

  await page.goto("/");
  await waitForPreviewReady(page);
  await page.evaluate(() => {
    const worker = (
      globalThis as typeof globalThis & {
        __ardyTestWorker?: Worker;
      }
    ).__ardyTestWorker;
    if (!worker) throw new Error("Inference worker was not observed.");
    worker.dispatchEvent(
      new MessageEvent("message", {
        data: {
          type: "error",
          requestId: "synthetic-internal-failure",
          error: {
            name: "Error",
            message: "Synthetic internal inference failure",
          },
        },
      }),
    );
  });

  await expect.poll(() => internalErrors).toHaveLength(1);
  await expect(
    page.locator("#error-banner, #error-title, #error-message, #dismiss-error"),
  ).toHaveCount(0);
  await expect(
    page.getByText("Synthetic internal inference failure", {
      exact: false,
    }),
  ).toHaveCount(0);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  expect(pageErrors).toEqual([]);
});

test("uses a viewport-centered workspace and settings drawer on narrow screens", async ({
  page,
}) => {
  await page.setViewportSize({ width: 800, height: 900 });
  await gotoReadyApp(page);

  const viewportPanel = page.locator("#viewport-panel");
  const viewportPanelBox = await viewportPanel.boundingBox();
  expect(viewportPanelBox).not.toBeNull();
  expect(viewportPanelBox!.x).toBe(0);
  expect(viewportPanelBox!.y).toBe(0);
  expect(viewportPanelBox!.width).toBeCloseTo(800, 1);
  expect(viewportPanelBox!.height).toBeCloseTo(900, 1);
  await expect(page.locator("#generator-panel")).toHaveCount(0);
  await expect(page.locator("#sidebar-toggle")).toHaveCount(0);
  await expect(page.locator(".sidebar-toggle-anchor")).toHaveCount(0);
  await expect(
    page.getByRole("button", {
      name: "Settings",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", {
      name: "Motion description",
      exact: true,
    }),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.documentElement.scrollWidth <= window.innerWidth &&
          document.documentElement.scrollHeight <= window.innerHeight,
      ),
    )
    .toBe(true);

  await openSettings(page);
  await expect(page.locator("#preview-settings")).toHaveAttribute(
    "data-slot",
    "drawer-popup",
  );
  await expect(
    page.locator("#preview-settings [data-slot='drawer-title']"),
  ).toHaveClass(/sr-only/);
  await expect(
    page.locator("#preview-settings [data-slot='drawer-header']"),
  ).toHaveCount(0);
  await expect(
    page.locator("#preview-settings [data-slot='drawer-description']"),
  ).toHaveCount(0);
  await expect(page.getByRole("tab", { name: "Motion" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(
    page.getByRole("slider", { name: "Buffer ahead", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Motion generation", { exact: true }),
  ).toHaveCount(0);
  await expect(page.locator("fieldset:has(#seed)")).toHaveCount(0);
  const motionTabsBox = await page
    .locator('[data-slot="tabs-list"]')
    .boundingBox();
  expect(motionTabsBox).not.toBeNull();
  expect(
    900 - (motionTabsBox!.y + motionTabsBox!.height),
  ).toBeCloseTo(16, 0);
  await page.getByRole("tab", { name: "View" }).click();
  await expect(page.locator("#import-vrm")).toBeVisible();
  await expect
    .poll(async () => {
      const box = await page
        .locator('[data-slot="tabs-list"]')
        .boundingBox();
      return box ? 900 - (box.y + box.height) : Number.NaN;
    })
    .toBeCloseTo(16, 0);
});

test("honors reduced motion and keeps shadcn controls usable on mobile", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await gotoReadyApp(page);

  await expect
    .poll(() =>
      page
        .locator("#generate")
        .evaluate((element) => getComputedStyle(element).transitionDuration),
    )
    .toBe("0s");
  await openPromptExamples(page);
  await expect
    .poll(() =>
      page
        .locator('[data-slot="combobox-content"]')
        .evaluate((element) => getComputedStyle(element).animationName),
    )
    .toBe("none");
  await expect
    .poll(() =>
      page
        .locator("#prompt-example")
        .evaluate((element) => getComputedStyle(element).fontSize),
    )
    .toBe("16px");
  await page.keyboard.press("Escape");

  await expect(page.locator("#generation-progress")).toHaveCount(0);
  await expect(page.locator("#cancel-generation")).toHaveCount(0);
  await expect(page.locator("#generation-stage")).toHaveCount(0);
  await expect(page.locator("#generation-percent")).toHaveCount(0);

  const [viewportPanel, viewport, promptComposer, playbackBar] =
    await Promise.all(
      ["#viewport-panel", "#viewport", ".prompt-composer", "#playback-bar"].map(
        (selector) => page.locator(selector).boundingBox(),
      ),
    );
  expect(viewportPanel).not.toBeNull();
  expect(viewport).not.toBeNull();
  expect(promptComposer).not.toBeNull();
  expect(playbackBar).not.toBeNull();
  expect(viewportPanel!.y).toBe(0);
  expect(viewportPanel!.height).toBeCloseTo(844, 1);
  expect(viewport!.y + viewport!.height).toBeCloseTo(promptComposer!.y, 1);
  expect(promptComposer!.y + promptComposer!.height).toBeCloseTo(
    playbackBar!.y,
    1,
  );

  await expect(page.locator("#generator-panel")).toHaveCount(0);
  await openSettings(page);
  await expect(page.locator("#preview-settings")).toHaveAttribute(
    "data-slot",
    "drawer-popup",
  );

  // Regression: the initial mobile drawer placement must make the slider
  // thumb measurable and interactive.
  const targetBuffer = page.locator("#target-buffer");
  const targetBufferThumb = targetBuffer.locator('[data-slot="slider-thumb"]');
  const targetBufferTrack = targetBuffer.locator('[data-slot="slider-track"]');
  await expect(targetBufferThumb).toBeVisible();
  const [targetBufferThumbBox, targetBufferTrackBox] = await Promise.all([
    targetBufferThumb.boundingBox(),
    targetBufferTrack.boundingBox(),
  ]);
  expect(targetBufferThumbBox).not.toBeNull();
  expect(targetBufferTrackBox).not.toBeNull();
  expect(targetBufferThumbBox!.width).toBeGreaterThan(0);
  expect(targetBufferThumbBox!.height).toBeGreaterThan(0);
  const thumbCenterX =
    targetBufferThumbBox!.x + targetBufferThumbBox!.width / 2;
  expect(thumbCenterX).toBeGreaterThanOrEqual(targetBufferTrackBox!.x - 1);
  expect(thumbCenterX).toBeLessThanOrEqual(
    targetBufferTrackBox!.x + targetBufferTrackBox!.width + 1,
  );

  await expect(page.locator(".inspector-panel")).toHaveCount(0);
  await expect
    .poll(() =>
      page
        .locator("#seed")
        .evaluate((element) => getComputedStyle(element).fontSize),
    )
    .toBe("16px");
  await page.keyboard.press("Escape");
  await expect(page.locator("#settings-trigger")).toBeFocused();

  await openSettings(page, "view");
  const preview = await page.locator("#viewport").boundingBox();
  expect(preview).not.toBeNull();
  expect(preview!.height).toBeGreaterThan(0);
  await expect
    .poll(() =>
      page
        .locator("#prompt")
        .evaluate((element) => getComputedStyle(element).fontSize),
    )
    .toBe("16px");

  const controlSelectors = [
    "#generate",
    "#play-pause",
    "#playback-speed",
    "#reset-camera",
    "#settings-trigger",
    "#import-vrm",
  ];
  const boxes = await Promise.all(
    controlSelectors.map((selector) => page.locator(selector).boundingBox()),
  );
  for (const box of boxes) {
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(28);
    expect(box!.height).toBeGreaterThanOrEqual(28);
  }

  const playbackBoxes = await Promise.all(
    ["#play-pause", "#playback-speed"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(
    Math.max(...playbackBoxes.map((box) => box!.y)) -
      Math.min(...playbackBoxes.map((box) => box!.y)),
  ).toBeLessThanOrEqual(2);
  const overlayBoxes = await Promise.all(
    ["#reset-camera", "#settings-trigger"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(
    Math.max(...overlayBoxes.map((box) => box!.y)) -
      Math.min(...overlayBoxes.map((box) => box!.y)),
  ).toBeLessThanOrEqual(2);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollHeight <= window.innerHeight,
      ),
    )
    .toBe(true);
});

test("keeps coarse-pointer controls at least 44 pixels", async ({
  context,
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const client = await context.newCDPSession(page);
  await client.send("Emulation.setTouchEmulationEnabled", {
    enabled: true,
    maxTouchPoints: 5,
  });

  await gotoReadyApp(page);
  await expect
    .poll(() =>
      page.evaluate(() => matchMedia("(any-pointer: coarse)").matches),
    )
    .toBe(true);

  const alwaysVisibleControls = [
    page.locator('[data-slot="input-group"]:has(#prompt)'),
    page.locator('[data-slot="combobox-trigger"]'),
    page.locator("#generate"),
    page.locator("#generation-actions-menu"),
    page.locator("#settings-trigger"),
    page.locator("#play-pause"),
    page.locator("#playback-speed"),
    page.locator("#reset-camera"),
    page.locator("#timeline"),
  ];
  const boxes = await Promise.all(
    alwaysVisibleControls.map((control) => control.boundingBox()),
  );
  for (const [index, box] of boxes.entries()) {
    expect(box).not.toBeNull();
    expect(
      box!.width,
      `always-visible control ${index} tap-target width`,
    ).toBeGreaterThanOrEqual(43.5);
    expect(
      box!.height,
      `always-visible control ${index} tap-target height`,
    ).toBeGreaterThanOrEqual(43.5);
  }

  await openSettings(page);
  const [tabsListBox, motionSettingsBox] = await Promise.all([
    page.locator('[data-slot="tabs-list"]').boundingBox(),
    page
      .locator('[data-slot="field-group"]:has(#target-buffer)')
      .boundingBox(),
  ]);
  expect(tabsListBox).not.toBeNull();
  expect(motionSettingsBox).not.toBeNull();
  expect(tabsListBox!.y).toBeGreaterThanOrEqual(
    motionSettingsBox!.y + motionSettingsBox!.height,
  );
  const drawerControls = [
    page.getByRole("tab", { name: "Motion" }),
    page.getByRole("tab", { name: "View" }),
    page.locator("#randomize-seed"),
    page.locator("#target-buffer"),
    page.locator("#clear-model-cache"),
  ];
  for (const [index, control] of drawerControls.entries()) {
    await control.scrollIntoViewIfNeeded();
    const box = await control.boundingBox();
    expect(box).not.toBeNull();
    expect(
      box!.width,
      `drawer control ${index} tap-target width`,
    ).toBeGreaterThanOrEqual(43.5);
    expect(
      box!.height,
      `drawer control ${index} tap-target height`,
    ).toBeGreaterThanOrEqual(43.5);
  }
  const targetBufferThumb = page.locator(
    '#target-buffer [data-slot="slider-thumb"]',
  );
  await expect(targetBufferThumb).toBeVisible();
  expect(await targetBufferThumb.boundingBox()).not.toBeNull();
  for (const sliderSelector of ["#target-buffer", "#timeline"]) {
    const slider = page.locator(sliderSelector);
    const track = slider.locator('[data-slot="slider-track"]');
    const [sliderBox, trackBox] = await Promise.all([
      slider.boundingBox(),
      track.boundingBox(),
    ]);
    expect(sliderBox).not.toBeNull();
    expect(trackBox).not.toBeNull();
    expect(
      Math.abs(
        sliderBox!.y +
          sliderBox!.height / 2 -
          (trackBox!.y + trackBox!.height / 2),
      ),
      `${sliderSelector} track vertical alignment`,
    ).toBeLessThanOrEqual(0.5);
  }
  await page.keyboard.press("Escape");

  const promptExamples = page.locator('[data-slot="combobox-item"]');
  await openPromptExamples(page);
  await expect(promptExamples).toHaveCount(100);
  for (const index of [0, 49, 99]) {
    const item = promptExamples.nth(index);
    await item.scrollIntoViewIfNeeded();
    const box = await item.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.height).toBeGreaterThanOrEqual(44);
  }
  await page.keyboard.press("Escape");
  await openSettings(page, "view");
  const importVrm = page.locator("#import-vrm");
  await expect(importVrm).toBeVisible();
  const importVrmBox = await importVrm.boundingBox();
  expect(importVrmBox).not.toBeNull();
  expect(importVrmBox!.width).toBeGreaterThanOrEqual(43.5);
  expect(importVrmBox!.height).toBeGreaterThanOrEqual(43.5);

  const displayControlIds = [
    "show-vrm",
    "show-skeleton",
    "show-contacts",
    "show-orientations",
    "show-trajectory",
  ];
  for (const controlId of displayControlIds) {
    const label = page.locator(`label[for="${controlId}"]`);
    const box = await label.boundingBox();
    expect(box).not.toBeNull();
    expect(
      box!.height,
      `${controlId} label tap-target height`,
    ).toBeGreaterThanOrEqual(43.5);
  }
  await expect(page.locator("#stream-generation")).toHaveCount(0);

  const orientations = page.locator("#show-orientations");
  await expect(orientations).toHaveAttribute("aria-checked", "true");
  const orientationsLabel = page.locator('label[for="show-orientations"]');
  await orientationsLabel.scrollIntoViewIfNeeded();
  const orientationsLabelBox = await orientationsLabel.boundingBox();
  expect(orientationsLabelBox).not.toBeNull();
  await page.mouse.click(
    orientationsLabelBox!.x + orientationsLabelBox!.width - 2,
    orientationsLabelBox!.y + orientationsLabelBox!.height - 2,
  );
  await expect(orientations).toHaveAttribute("aria-checked", "false");
  await page.keyboard.press("Escape");
  await expect(page.locator("#preview-settings")).toBeHidden();
});

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import {
  allowRequiredWebGpuFeatureForPreflight,
  openPreviewSettings,
  setSliderValue,
  waitForPreviewReady,
} from "./control-helpers";
import {
  createMockModelFiles,
  installMockModelWorker,
  missingDevelopmentModelRoute,
  routeMockModelFiles,
} from "./model-files-fixture";

test.beforeEach(async ({ page }) => {
  await allowRequiredWebGpuFeatureForPreflight(page);
  await page.route(
    missingDevelopmentModelRoute,
    async (route) => {
      await route.fulfill({ status: 404, body: "Not found" });
    },
  );
});

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
    if (!viewerModuleUrl) throw new Error("Viewer module URL was not observed.");
    const { SkeletonViewer } = await import(
      /* @vite-ignore */ viewerModuleUrl
    );
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

async function waitForAnimationFrames(page: Page, count: number): Promise<void> {
  await page.evaluate(async (frameCount) => {
    for (let frame = 0; frame < frameCount; frame += 1) {
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    }
  }, count);
}

async function openMotionControls(page: Page): Promise<void> {
  const trigger = page.getByRole("button", {
    name: "Motion controls",
    exact: true,
  });
  await expect(trigger).toBeVisible();
  await trigger.click();
  await expect(
    page.getByRole("dialog", {
      name: "Motion controls",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator("#generator-panel")).toBeVisible();
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
  await expect(page.locator("#model-runtime-state")).toHaveText(
    "Unavailable",
  );
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
  expect(panePositions[0]!.y).toBe(0);
  expect(panePositions[1]!.y).toBe(0);
  expect(panePositions[0]!.height).toBeCloseTo(900, 1);
  expect(panePositions[1]!.height).toBeCloseTo(900, 1);

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
    "Motion generation",
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
    page.getByText(
      "Clear, typo-free English. Apply updates while streaming.",
      { exact: true },
    ),
  ).toHaveCount(0);
  await expect(page.locator("#privacy-badge")).toHaveCount(0);
  await expect(page.locator("#gpu-badge")).toHaveCount(0);
  await expect(page.locator("#isolation-badge")).toHaveCount(0);
  await expect(page.locator("#backend")).toHaveCount(0);
  await expect(page.locator("#model-cache")).toBeVisible();
  await expect(page.locator("#model-cache-state")).toHaveText(
    "Needs attention",
  );
  await expect(page.locator("#download-model")).toHaveText("Retry download");
  await expect(page.getByText("20 FPS", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Core40", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Runtime notes", { exact: true })).toHaveCount(0);
  await expect(page.locator("#runtime-settings")).toHaveCount(0);

  for (const selector of [
    "#generate",
    "#restart-generation",
    "#restart-from-now",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
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
  await openPreviewSettings(page);
  await expect(page.getByText("VRM avatar", { exact: true })).toBeVisible();
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
  await expect(page.getByText("Reference motion", { exact: true })).toHaveCount(0);
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
  await expect(
    page.getByText(unavailableReason, { exact: true }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("alertdialog", {
      name: "Download model files?",
    }),
  ).toHaveCount(0);
  await expect(page.locator("#download-model")).toHaveCount(0);
  await expect(page.locator("#generate")).toBeDisabled();

  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
  await page.mouse.click(2, 2);
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button")).toHaveCount(0);
});

test("confirms model download and manages the browser cache", async ({
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

  const downloadDialog = page.getByRole("alertdialog", {
    name: "Download model files?",
  });
  const postpone = downloadDialog.getByRole("button", {
    name: "Not now",
    exact: true,
  });
  const confirmDownload = downloadDialog.getByRole("button", {
    name: "Download model",
    exact: true,
  });
  await expect(downloadDialog).toBeVisible();
  await expect(postpone).toBeFocused();
  await expect(page.locator("#model-cache-state")).toHaveText(
    "Not cached",
  );
  await expect(page.locator("#model-cache-files")).toHaveText("0 of 5");
  await expect(page.locator("#model-runtime-state")).toHaveText(
    "Not loaded",
  );
  expect(
    modelRequests.filter((path) => !path.endsWith("model.json.gz")),
  ).toEqual([]);

  await postpone.click();
  await expect(downloadDialog).toBeHidden();
  const downloadButton = page.locator("#download-model");
  await expect(downloadButton).toHaveText("Download model");
  await downloadButton.click();
  await expect(downloadDialog).toBeVisible();
  await expect(postpone).toBeFocused();

  await confirmDownload.click();
  await expect(downloadDialog).toBeHidden();
  await expect(page.locator("#model-cache-state")).toHaveText(
    /Downloading|Verifying/,
  );
  await expect(page.locator("#model-download-progress")).toBeVisible();
  await expect(page.locator("#model-cache-state")).toHaveText("Cached");
  await expect(page.locator("#model-cache-files")).toHaveText("5 of 5");
  await expect(page.locator("#model-runtime-state")).toHaveText("Ready");
  await expect(page.locator("#model-download-progress")).toHaveCount(0);
  const payloadRequestCount = modelRequests.filter(
    (path) => !path.endsWith("model.json.gz"),
  ).length;
  expect(payloadRequestCount).toBe(5);

  await page.reload();
  await expect(downloadDialog).toHaveCount(0);
  await expect(page.locator("#model-cache-state")).toHaveText("Cached");
  await expect(page.locator("#model-runtime-state")).toHaveText("Ready");
  expect(
    modelRequests.filter((path) => !path.endsWith("model.json.gz")),
  ).toHaveLength(payloadRequestCount);

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
    .getByRole("button", { name: "Clear cache", exact: true })
    .click();
  await expect(clearDialog).toBeHidden();
  await expect(page.locator("#model-cache-state")).toHaveText(
    "Not cached",
  );
  await expect(page.locator("#model-cache-files")).toHaveText("0 of 5");
  await expect(page.locator("#model-runtime-state")).toHaveText("Ready");
  await expect(downloadButton).toBeVisible();
  await expect(clearCache).toHaveCount(0);
});

test("rejects renderer initialization instead of using a WebGL fallback", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript(() => {
    const requestedContexts: string[] = [];
    type ContextGetter = (
      contextId: string,
      ...options: unknown[]
    ) => unknown;
    const canvasPrototype =
      HTMLCanvasElement.prototype as unknown as {
        getContext: ContextGetter;
      };
    const originalGetContext = canvasPrototype.getContext;
    canvasPrototype.getContext = function (
      contextId,
      ...options
    ): unknown {
      requestedContexts.push(contextId);
      return Reflect.apply(
        originalGetContext,
        this,
        [contextId, ...options],
      );
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
  await page.goto("/");

  const radii = await page.evaluate(() =>
    [
      "#model-cache",
      "#generate",
      "#prompt",
      '[data-slot="combobox-trigger"]',
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

  expect(radii).toEqual([
    "0px",
    "0px",
    "0px",
    "0px",
    "0px",
    "0px",
  ]);
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
    const {
      CORE27_JOINT_COUNT,
      CORE27_SKELETON,
      SkeletonViewer,
    } = viewerModule;
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
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );

      const ground = internal.scene.getObjectByName(
        "camera-relative-ground-grid",
      );
      const rig = internal.scene.getObjectByName("shadow-follow-rig");
      const light = internal.scene.getObjectByName("shadow-key-light");
      const snapshot = () => ({
        ground: ground
          ? { x: ground.position.x, z: ground.position.z }
          : null,
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
        await internal.renderer.compileAsync(
          internal.scene,
          internal.camera,
        );
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
        validationError =
          (await device?.popErrorScope())?.message ?? null;
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
          isWebGPUBackend:
            internal.renderer.backend.isWebGPUBackend === true,
          isWebGLBackend:
            internal.renderer.backend.isWebGLBackend === true,
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
          2 *
          (internal.camera.far +
            internal.controls.maxDistance +
            5),
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

  const snappedTarget = (value: number): number =>
    Math.round(value / 5) * 5;
  for (const state of [
    groundState.initial,
    groundState.afterOrbit,
    groundState.afterMove,
  ]) {
    expect(state.ground?.x).toBeCloseTo(snappedTarget(state.target.x));
    expect(state.ground?.z).toBeCloseTo(snappedTarget(state.target.z));
  }
  expect(groundState.afterOrbit.ground).toEqual(
    groundState.initial.ground,
  );
  expect(groundState.afterMove.ground).not.toEqual(
    groundState.initial.ground,
  );
  expect(groundState.afterMove.camera).not.toEqual(
    groundState.initial.camera,
  );
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
    const {
      CORE27_JOINT_COUNT,
      CORE27_SKELETON,
      SkeletonViewer,
    } = await import("/src/viewer.ts");
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);
    const viewer = await SkeletonViewer.create(canvas);
    try {
      const positions = new Float32Array(
        2 * CORE27_JOINT_COUNT * 3,
      );
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
      viewer.resetCamera();
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

  const delta = (
    after: number[],
    before: number[],
  ): number[] => after.map((value, index) => value - before[index]);
  const relative = (state: {
    camera: number[];
    target: number[];
  }): number[] => delta(state.camera, state.target);
  expect(delta(cameraState.followed.camera, cameraState.initial.camera)).toEqual(
    [6, 0, -4],
  );
  expect(delta(cameraState.followed.target, cameraState.initial.target)).toEqual(
    [6, 0, -4],
  );
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
  expect(Math.hypot(manualCameraDelta[0], manualCameraDelta[2])).toBeGreaterThan(
    0,
  );
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
  await page.goto("/");

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
  await expect(page.locator("#prompt-count")).toHaveText("57 / 280");
  await page.locator("#prompt-count").click();
  await expect(prompt).toBeFocused();

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
  await expect(promptExample).toHaveAttribute(
    "placeholder",
    "Search examples",
  );
  const promptExampleContent = page.locator(
    '[data-slot="combobox-content"]',
  );
  await expect(promptExampleContent).toBeVisible();
  const promptOptions = promptExampleContent.getByRole("option");
  await expect(promptOptions).toHaveCount(100);
  await expect(promptOptions.first()).toHaveAttribute(
    "data-slot",
    "combobox-item",
  );

  await promptExample.fill("Joyful dance");
  await expect(promptOptions).toHaveCount(1);
  await expect(promptOptions).toHaveText("Joyful dance");
  await promptExample.press("ArrowDown");
  await promptExample.press("Enter");
  await expect(prompt).toHaveValue("A person performs a joyful dance.");

  const validation = await page.evaluate(async () => {
    const { validateGenerationForm } = await import("/src/main.ts");
    return {
      empty: validateGenerationForm("", "2").promptError,
      multilingual: validateGenerationForm("人物が歩く。", "2").values,
      long: validateGenerationForm("a".repeat(281), "2").promptError,
      seed: validateGenerationForm("A person walks.", "-1").seedError,
      valid: validateGenerationForm(
        "A person walks forward.",
        "4294967295",
      ).values,
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
  await page.goto("/");
  await waitForPreviewReady(page);
  await page.evaluate(async () => {
    const [{ SkeletonViewer }, { playbackSpeedControl }] =
      await Promise.all([
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

test("keeps labels, keyboard focus, and canvas controls accessible", async ({
  page,
}) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "ARDY browser motion workspace",
    }),
  ).toHaveClass(/sr-only/);
  await expect(
    page.getByRole("heading", { level: 2, name: "Motion controls" }),
  ).toHaveClass(/sr-only/);
  await expect(
    page.getByRole("heading", { level: 2, name: "Motion preview" }),
  ).toHaveClass(/sr-only/);

  await page.keyboard.press("Tab");
  await expect(page.locator(".skip-link")).toBeFocused();

  const previewSettingsTrigger = page.locator("#preview-settings-trigger");
  const previewSettingsContent = page.locator("#preview-settings");
  const [viewportBefore, triggerBox] = await Promise.all([
    page.locator("#viewport").boundingBox(),
    previewSettingsTrigger.boundingBox(),
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
  await expect(previewSettingsContent).toHaveCount(1);
  await expect(previewSettingsContent).toHaveAttribute(
    "data-slot",
    "popover-content",
  );
  await expect(previewSettingsContent).toBeHidden();
  await previewSettingsTrigger.click();
  await expect(previewSettingsTrigger).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expect(previewSettingsContent).toBeVisible();
  expect(await page.locator("#viewport").boundingBox()).toEqual(
    viewportBefore,
  );
  const previewSettingsOverflow = await previewSettingsContent.evaluate(
    (element) => {
      const style = getComputedStyle(element);
      const borderWidth =
        Number.parseFloat(style.borderLeftWidth) +
        Number.parseFloat(style.borderRightWidth);
      return {
        overflowY: style.overflowY,
        scrollbarWidth:
          element.offsetWidth - element.clientWidth - borderWidth,
      };
    },
  );
  expect(previewSettingsOverflow.overflowY).toBe("visible");
  expect(previewSettingsOverflow.scrollbarWidth).toBeLessThanOrEqual(0.5);
  await page.keyboard.press("Escape");
  await expect(previewSettingsTrigger).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  await expect(previewSettingsContent).toBeHidden();
  await expect(previewSettingsTrigger).toBeFocused();

  await expect(
    page.getByRole("textbox", {
      name: "Motion description",
      exact: true,
    }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("spinbutton", { name: "Seed", exact: true }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("slider", { name: "Buffer ahead", exact: true }),
  ).toHaveCount(1);

  await expect(page.locator("#model-runtime-state")).toHaveText(
    "Unavailable",
  );
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
  await expect(resetCamera).toHaveAttribute(
    "data-variant",
    "outline",
  );
  const resetBorder = await resetCamera.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      style: style.borderStyle,
      width: style.borderWidth,
    };
  });
  expect(resetBorder).toEqual({ style: "solid", width: "1px" });
});

test("labels every icon button in the playback bar with a tooltip", async ({
  page,
}) => {
  await page.goto("/");
  await waitForPreviewReady(page);
  await installCameraMovementProbe(page);
  const canvas = page.locator("#motion-canvas");
  await canvas.focus();
  await page.keyboard.press("w");
  await page.evaluate(async () => {
    const {
      CORE27_JOINT_COUNT,
      CORE27_SKELETON,
    } = await import("/src/viewer.ts");
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
    const probe = (
      globalThis as typeof globalThis & {
        __cameraMovementProbe?: { viewer: MotionViewer | null };
      }
    );
    const viewer = probe.__cameraMovementProbe?.viewer;
    if (!viewer) throw new Error("Preview viewer is unavailable.");
    const frameCount = 200;
    viewer.setMotion(
      {
        skeleton: CORE27_SKELETON,
        positions: new Float32Array(
          frameCount * CORE27_JOINT_COUNT * 3,
        ),
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
  await page.goto("/");
  await waitForPreviewReady(page);
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
  await page.goto("/");
  await waitForPreviewReady(page);
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
        const worker = Reflect.construct(
          target,
          argumentsList,
        ) as Worker;
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

  await expect
    .poll(() => internalErrors)
    .toHaveLength(1);
  await expect(
    page.locator(
      "#error-banner, #error-title, #error-message, #dismiss-error",
    ),
  ).toHaveCount(0);
  await expect(
    page.getByText("Synthetic internal inference failure", {
      exact: false,
    }),
  ).toHaveCount(0);
  await expect
    .poll(() =>
      page.locator("#generator-panel").evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
      ),
    )
    .toBe(true);
  expect(pageErrors).toEqual([]);
});

test("keeps model cache actions inside the input panel at minimum width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/");
  await openMotionControls(page);

  const [panel, modelCard, downloadButton] = await Promise.all(
    ["#generator-panel", "#model-cache", "#download-model"].map((selector) =>
      page.locator(selector).boundingBox(),
    ),
  );
  expect(panel).not.toBeNull();
  expect(modelCard).not.toBeNull();
  expect(downloadButton).not.toBeNull();
  expect(modelCard!.x + modelCard!.width).toBeLessThanOrEqual(
    panel!.x + panel!.width,
  );
  expect(downloadButton!.x + downloadButton!.width).toBeLessThanOrEqual(
    panel!.x + panel!.width,
  );
  expect(
    await page
      .locator("#generator-panel")
      .evaluate((element) => element.scrollWidth <= element.clientWidth),
  ).toBe(true);
});

test("toggles the motion-control sidebar only in the side-by-side layout", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");

  const generatorPanel = page.locator("#generator-panel");
  const viewportPanel = page.locator("#viewport-panel");
  const viewport = page.locator("#viewport");
  const sidebarToggle = page.locator("#sidebar-toggle");
  const [generatorBefore, viewportPanelBefore, viewportBox, toggleBox] =
    await Promise.all([
      generatorPanel.boundingBox(),
      viewportPanel.boundingBox(),
      viewport.boundingBox(),
      sidebarToggle.boundingBox(),
    ]);
  expect(generatorBefore).not.toBeNull();
  expect(viewportPanelBefore).not.toBeNull();
  expect(viewportBox).not.toBeNull();
  expect(toggleBox).not.toBeNull();
  expect(toggleBox!.x - viewportBox!.x).toBeGreaterThanOrEqual(8);
  expect(toggleBox!.x - viewportBox!.x).toBeLessThanOrEqual(16);
  expect(toggleBox!.y - viewportBox!.y).toBeGreaterThanOrEqual(8);
  expect(toggleBox!.y - viewportBox!.y).toBeLessThanOrEqual(16);
  await expect(
    sidebarToggle.locator("svg.tabler-icon-layout-sidebar"),
  ).toHaveCount(1);
  await expect(sidebarToggle).toHaveAccessibleName(
    "Hide motion controls",
  );
  await expect(sidebarToggle).toHaveAttribute("aria-expanded", "true");
  const sidebarTooltip = page
    .locator('[data-slot="tooltip-content"]')
    .filter({ hasText: "Hide motion controls" });
  await sidebarToggle.hover();
  await expect(sidebarTooltip).toBeVisible();
  await expect(sidebarTooltip).toHaveAttribute("data-open", "");
  await page.waitForTimeout(750);
  await expect(sidebarTooltip).toBeVisible();
  await expect(sidebarTooltip).toHaveAttribute("data-open", "");
  await page.mouse.move(0, 0);
  await expect(sidebarTooltip).toBeHidden();

  await sidebarToggle.click();
  await expect(generatorPanel).toBeHidden();
  await expect(sidebarToggle).toHaveAccessibleName(
    "Show motion controls",
  );
  await expect(sidebarToggle).toHaveAttribute("aria-expanded", "false");
  const viewportPanelCollapsed = await viewportPanel.boundingBox();
  expect(viewportPanelCollapsed).not.toBeNull();
  expect(viewportPanelCollapsed!.x).toBe(0);
  expect(viewportPanelCollapsed!.width).toBeGreaterThan(
    viewportPanelBefore!.width,
  );

  await sidebarToggle.click();
  await expect(generatorPanel).toBeVisible();
  await expect(sidebarToggle).toHaveAccessibleName(
    "Hide motion controls",
  );
  expect(await generatorPanel.boundingBox()).toEqual(generatorBefore);
  expect(await viewportPanel.boundingBox()).toEqual(viewportPanelBefore);
});

test("uses a viewport-centered workspace and motion drawer on narrow screens", async ({
  page,
}) => {
  await page.setViewportSize({ width: 800, height: 900 });
  await page.goto("/");

  const viewportPanel = page.locator("#viewport-panel");
  const viewportPanelBox = await viewportPanel.boundingBox();
  expect(viewportPanelBox).not.toBeNull();
  expect(viewportPanelBox!.x).toBe(0);
  expect(viewportPanelBox!.y).toBe(0);
  expect(viewportPanelBox!.width).toBeCloseTo(800, 1);
  expect(viewportPanelBox!.height).toBeCloseTo(900, 1);
  await expect(page.locator("#generator-panel")).toBeHidden();
  await expect(page.locator("#sidebar-toggle")).toHaveCount(0);
  await expect(page.locator(".sidebar-toggle-anchor")).toHaveCount(0);
  await expect(
    page.getByRole("button", {
      name: "Motion controls",
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

  await openMotionControls(page);
  await expect(
    page.getByRole("slider", { name: "Buffer ahead", exact: true }),
  ).toBeVisible();
});

test("honors reduced motion and keeps shadcn controls usable on mobile", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect
    .poll(() =>
      page
        .locator("#generate-spinner")
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

  const [viewportPanel, viewport, promptComposer, playbackBar] =
    await Promise.all(
      [
        "#viewport-panel",
        "#viewport",
        ".prompt-composer",
        "#playback-bar",
      ].map((selector) => page.locator(selector).boundingBox()),
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

  await expect(page.locator("#generator-panel")).toBeHidden();
  await openMotionControls(page);

  // Regression: the initial mobile drawer placement must make the slider
  // thumb measurable and interactive.
  const targetBuffer = page.locator("#target-buffer");
  const targetBufferThumb = targetBuffer.locator(
    '[data-slot="slider-thumb"]',
  );
  const targetBufferTrack = targetBuffer.locator(
    '[data-slot="slider-track"]',
  );
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

  const downloadModel = page.locator("#download-model");
  const downloadModelBox = await downloadModel.boundingBox();
  expect(downloadModelBox).not.toBeNull();
  await page.mouse.move(
    downloadModelBox!.x + downloadModelBox!.width / 2,
    downloadModelBox!.y + downloadModelBox!.height / 2,
  );
  await page.mouse.down();
  await expect
    .poll(() =>
      downloadModel.evaluate(
        (element) => getComputedStyle(element).translate,
      ),
    )
    .toBe("none");
  await page.mouse.move(0, 0);
  await page.mouse.up();

  const modelActionBounds = await Promise.all(
    ["#download-model", "#model-cache", "#generator-panel"].map(
      (selector) => page.locator(selector).boundingBox(),
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

  await expect(page.locator(".inspector-panel")).toHaveCount(0);
  await expect
    .poll(() =>
      page
        .locator("#seed")
        .evaluate((element) => getComputedStyle(element).fontSize),
    )
    .toBe("16px");
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", {
      name: "Motion controls",
      exact: true,
    }),
  ).toBeFocused();

  await openPreviewSettings(page);
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
    "#preview-settings-trigger",
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
    ["#play-pause", "#playback-speed", "#reset-camera"].map(
      (selector) => page.locator(selector).boundingBox(),
    ),
  );
  expect(
    Math.max(...playbackBoxes.map((box) => box!.y)) -
      Math.min(...playbackBoxes.map((box) => box!.y)),
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

  await page.goto("/");
  await expect
    .poll(() =>
      page.evaluate(() => matchMedia("(any-pointer: coarse)").matches),
    )
    .toBe(true);

  const alwaysVisibleControls = [
    page.locator('[data-slot="input-group"]:has(#prompt)'),
    page.locator('[data-slot="combobox-trigger"]'),
    page.locator("#generate"),
    page.getByRole("button", {
      name: "Motion controls",
      exact: true,
    }),
    page.locator("#preview-settings-trigger"),
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

  await openMotionControls(page);
  const drawerControls = [
    page.locator("#download-model"),
    page.locator("#randomize-seed"),
    page.locator("#target-buffer"),
    page.locator("#restart-generation"),
    page.locator("#restart-from-now"),
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
  await openPreviewSettings(page);
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
  const orientationsLabel = page.locator(
    'label[for="show-orientations"]',
  );
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

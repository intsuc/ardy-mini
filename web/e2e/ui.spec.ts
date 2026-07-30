// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { writeFile } from "node:fs/promises";

import { expect, test, type Page } from "@playwright/test";

import {
  openPreviewSettings,
  setSliderValue,
} from "./control-helpers";

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
  expect(panePositions[0]!.y).toBe(0);
  expect(panePositions[1]!.y).toBe(0);
  expect(panePositions[0]!.height).toBeCloseTo(900, 1);
  expect(panePositions[1]!.height).toBeCloseTo(900, 1);

  await expect(page.getByLabel("Motion description")).toBeEditable();
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
  await expect(page.locator("#model-setup-help")).toContainText(
    "ardy-minilm-core40-browser-v1.tar.gz",
  );
  await expect(page.locator("#privacy-badge")).toHaveCount(0);
  await expect(page.locator("#gpu-badge")).toHaveCount(0);
  await expect(page.locator("#isolation-badge")).toHaveCount(0);
  await expect(page.locator("#backend")).toHaveCount(0);
  await expect(page.locator("#import-model")).toContainText("Choose model pack");
  await expect(page.locator("#model-file-input")).toHaveAttribute(
    "accept",
    ".tar.gz,application/gzip",
  );
  await expect(page.locator("#model-file-input")).not.toHaveAttribute(
    "multiple",
  );
  await expect(page.getByText("20 FPS", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Core40", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Runtime notes", { exact: true })).toHaveCount(0);
  await expect(page.locator("#runtime-settings")).toHaveCount(0);

  for (const selector of [
    "#generate",
    "#apply-prompt",
    "#restart-generation",
    "#restart-from-now",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  for (const selector of [
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

  await expect(page.locator("#preview-settings")).toBeVisible();
  await expect(page.locator("#preview-settings")).toHaveAttribute(
    "data-slot",
    "accordion",
  );
  await expect(page.locator("#stream-generation")).toHaveAttribute(
    "data-slot",
    "switch",
  );
  await expect(page.locator("#duration")).toHaveAttribute(
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

test("blocks model loading before archive work when WebGPU is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: {
        requestAdapter: async () => null,
      },
    });
  });
  await page.goto("/");

  await expect(page.locator("#model-title")).toHaveText("WebGPU required");
  await expect(page.locator("#model-state")).toHaveText("Unavailable");
  await expect(page.locator("#model-detail")).toContainText(
    "No compatible GPU adapter is available",
  );
  await expect(page.locator("#import-model")).toBeDisabled();
  await expect(page.locator("#model-file-input")).toBeDisabled();
  await expect(page.locator("#model-setup-help")).toBeHidden();
  await expect(page.locator("#generate-help")).toContainText(
    "WebGPU is required",
  );
});

test("retains the square Lyra treatment on standard shadcn surfaces", async ({
  page,
}) => {
  await page.goto("/");

  const radii = await page.evaluate(() =>
    [
      "#model-card",
      "#generate",
      "#prompt",
      "#prompt-example",
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
      /(?:WebGLProgram|shader|GLSL)/i.test(message.text())
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

    const viewer = new SkeletonViewer(canvas);
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
          readonly dithering?: boolean;
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
      interface ShaderDiagnostics {
        readonly runnable?: boolean;
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
          readonly shadowMap: { readonly enabled: boolean };
          readonly debug: {
            onShaderError:
              | ((
                  gl: WebGL2RenderingContext,
                  program: WebGLProgram,
                  vertexShader: WebGLShader,
                  fragmentShader: WebGLShader,
                ) => void)
              | null;
          };
          readonly info: {
            readonly programs:
              | readonly { readonly diagnostics?: ShaderDiagnostics }[]
              | null;
          };
          compileAsync(scene: unknown, camera: unknown): Promise<unknown>;
          render(scene: unknown, camera: unknown): void;
        };
        vrm: unknown;
        vrmRetargetPlan: unknown;
        vrmRoot: {
          visible: boolean;
          readonly position: { readonly x: number; readonly z: number };
        };
      };
      const shaderErrors: string[] = [];
      internal.renderer.debug.onShaderError = (
        gl,
        program,
        vertexShader,
        fragmentShader,
      ) => {
        shaderErrors.push(
          [
            gl.getProgramInfoLog(program),
            gl.getShaderInfoLog(vertexShader),
            gl.getShaderInfoLog(fragmentShader),
          ]
            .filter(Boolean)
            .join("\n"),
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

      await internal.renderer.compileAsync(
        internal.scene,
        internal.camera,
      );
      internal.renderer.render(internal.scene, internal.camera);
      const failedPrograms =
        internal.renderer.info.programs?.filter(
          (program) => program.diagnostics?.runnable === false,
        ).length ?? 0;
      const width =
        (ground?.geometry?.parameters.width ?? 0) *
        Math.abs(ground?.scale.x ?? 0);
      const height =
        (ground?.geometry?.parameters.height ?? 0) *
        Math.abs(ground?.scale.y ?? 0);

      return {
        rendererShadows: internal.renderer.shadowMap.enabled,
        ground: ground
          ? {
              geometry: ground.geometry?.type,
              width,
              height,
              receiveShadow: ground.receiveShadow,
              dithering: ground.material?.dithering,
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
        shaderErrors,
        failedPrograms,
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  expect(groundState.rendererShadows).toBe(true);
  expect(groundState.namedGroundCount).toBe(1);
  expect(groundState.gridHelpers).toEqual([]);
  expect(groundState.ground).toMatchObject({
    geometry: "PlaneGeometry",
    receiveShadow: true,
    dithering: true,
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
  expect(groundState.shaderErrors).toEqual([]);
  expect(groundState.failedPrograms).toBe(0);
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
    const viewer = new SkeletonViewer(canvas);
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

  const prompt = page.getByLabel("Motion description");
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

  await setSliderValue(page, "#duration", 8);
  await expect(page.locator("#duration-output")).toHaveText("8 seconds");
  await setSliderValue(page, "#target-buffer", 120);
  await expect(page.locator("#target-buffer-output")).toHaveText("120 frames");
  await expect(page.getByRole("spinbutton", { name: "Seed" })).toHaveValue("2");

  const promptExample = page.getByRole("combobox", {
    name: "Example prompt",
  });
  await expect(promptExample).toHaveAttribute(
    "placeholder",
    "Search examples",
  );
  await expect(
    page.getByRole("button", { name: "Open example prompts" }),
  ).toBeVisible();
  await promptExample.click();
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
  await expect(promptExample).toHaveValue("Joyful dance");
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
  const previewSettingsContent = page.locator(
    '[data-slot="accordion-content"]',
  );
  await expect(previewSettingsContent).toHaveCount(1);
  await expect(previewSettingsContent).toBeHidden();
  await previewSettingsTrigger.focus();
  await page.keyboard.press("Space");
  await expect(previewSettingsTrigger).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expect(previewSettingsContent).toBeVisible();
  await page.keyboard.press("Space");
  await expect(previewSettingsTrigger).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  await expect(previewSettingsContent).toBeHidden();

  for (const label of [
    "Motion description",
    "Seed",
  ]) {
    await expect(page.getByLabel(label, { exact: true })).toHaveCount(1);
  }
  await expect(
    page.getByRole("slider", { name: "Duration", exact: true }),
  ).toHaveCount(1);

  await expect(page.locator("#model-state")).toHaveText("Not loaded");
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

  const loopToggle = page.locator("#loop-toggle");
  await expect(loopToggle).toBeDisabled();
  await expect(loopToggle).toHaveAttribute("aria-pressed", "true");
});

test("moves continuously from one held W keydown and stops on keyup", async ({
  page,
}) => {
  await page.goto("/");
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

test("keeps invalid model-pack errors beside the model setup action", async ({
  page,
}, testInfo) => {
  await page.goto("/");

  const invalidPack = testInfo.outputPath("invalid-pack.tar.gz");
  await writeFile(invalidPack, "not a gzip archive");
  await page.locator("#model-file-input").setInputFiles(invalidPack);

  const modelError = page.locator("#model-error-banner");
  await expect(modelError).toBeVisible();
  await expect(modelError).toBeFocused();
  await expect(modelError).toContainText(/decompress|gzip|shader-f16/i);
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
  await page.locator("#prompt-example").click();
  await expect
    .poll(() =>
      page
        .locator('[data-slot="combobox-content"]')
        .evaluate((element) => getComputedStyle(element).animationName),
    )
    .toBe("none");
  await page.keyboard.press("Escape");
  const importModel = page.locator("#import-model");
  const importModelBox = await importModel.boundingBox();
  expect(importModelBox).not.toBeNull();
  await page.mouse.move(
    importModelBox!.x + importModelBox!.width / 2,
    importModelBox!.y + importModelBox!.height / 2,
  );
  await page.mouse.down();
  await expect
    .poll(() =>
      importModel.evaluate((element) => getComputedStyle(element).translate),
    )
    .toBe("none");
  await page.mouse.up();

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
  expect(panes[0]!.y).toBe(0);
  expect(panes[0]!.y).toBeLessThan(panes[1]!.y);
  await expect(page.locator(".inspector-panel")).toHaveCount(0);

  await openPreviewSettings(page);
  const preview = await page.locator("#viewport").boundingBox();
  expect(preview).not.toBeNull();
  expect(preview!.height).toBeGreaterThanOrEqual(384);

  const mobileFormFontSizes = await page.evaluate(() =>
    ["#prompt", "#seed", "#prompt-example"].map((selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing mobile form control: ${selector}`);
      }
      return getComputedStyle(element).fontSize;
    }),
  );
  expect(mobileFormFontSizes).toEqual(["16px", "16px", "16px"]);

  const controlSelectors = [
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
    controlSelectors.map((selector) => page.locator(selector).boundingBox()),
  );
  for (const box of boxes) {
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(28);
    expect(box!.height).toBeGreaterThanOrEqual(28);
  }

  const playbackBoxes = await Promise.all(
    ["#play-pause", "#playback-speed", "#loop-toggle", "#reset-camera"].map(
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
});

test("keeps coarse-pointer controls at least 44 pixels", async ({
  browser,
}, testInfo) => {
  const context = await browser.newContext({
    baseURL: testInfo.project.use.baseURL as string,
    hasTouch: true,
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();

  try {
    await page.goto("/");
    await openPreviewSettings(page);
    await expect
      .poll(() =>
        page.evaluate(() => matchMedia("(any-pointer: coarse)").matches),
      )
      .toBe(true);

    const selectors = [
      "#import-model",
      '[data-slot="input-group"]:has(#prompt-example)',
      "[data-combobox-trigger]",
      "#apply-prompt",
      "#randomize-seed",
      "#generate",
      "#preview-settings-trigger",
      "#play-pause",
      "#playback-speed",
      "#loop-toggle",
      "#reset-camera",
      "#import-vrm",
      "#duration",
      "#target-buffer",
      "#timeline",
    ];
    const boxes = await Promise.all(
      selectors.map((selector) => page.locator(selector).boundingBox()),
    );
    for (const box of boxes) {
      expect(box).not.toBeNull();
      expect(box!.width).toBeGreaterThanOrEqual(44);
      expect(box!.height).toBeGreaterThanOrEqual(44);
    }

    const promptExamples = page.locator('[data-slot="combobox-item"]');
    await page.locator("#prompt-example").click();
    await expect(promptExamples).toHaveCount(100);
    for (const index of [0, 49, 99]) {
      const item = promptExamples.nth(index);
      await item.scrollIntoViewIfNeeded();
      const box = await item.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.height).toBeGreaterThanOrEqual(44);
    }
    await page.keyboard.press("Escape");

    for (const controlId of [
      "stream-generation",
      "show-vrm",
      "show-skeleton",
      "show-contacts",
      "show-orientations",
      "show-trajectory",
    ]) {
      const label = page.locator(`label[for="${controlId}"]`);
      const box = await label.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.height).toBeGreaterThanOrEqual(44);
    }

    const continuousGeneration = page.locator("#stream-generation");
    const continuousLabel = page.locator(
      'label[for="stream-generation"]',
    );
    await continuousLabel.scrollIntoViewIfNeeded();
    const continuousLabelBox = await continuousLabel.boundingBox();
    expect(continuousLabelBox).not.toBeNull();
    await page.mouse.click(
      continuousLabelBox!.x + continuousLabelBox!.width - 2,
      continuousLabelBox!.y + 2,
    );
    await expect(continuousGeneration).toHaveAttribute(
      "aria-checked",
      "false",
    );

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
  } finally {
    await context.close();
  }
});

test("returns focus to model selection after saved pack removal", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#model-state")).toHaveText("Not loaded");

  const removeModel = page.locator("#remove-model");
  await removeModel.evaluate((element) => {
    (element as HTMLButtonElement).hidden = false;
  });
  await removeModel.click();
  const dialog = page.getByRole("alertdialog", {
    name: "Remove the saved model pack?",
  });
  const cancel = page.getByRole("button", { name: "Cancel", exact: true });
  const confirm = page.locator("#confirm-remove-model");

  await expect(dialog).toBeVisible();
  await expect(cancel).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(confirm).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(cancel).toBeFocused();

  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(removeModel).toBeFocused();

  await removeModel.click();
  await expect(cancel).toBeFocused();
  await cancel.click();
  await expect(dialog).toBeHidden();
  await expect(removeModel).toBeFocused();

  await removeModel.click();
  await expect(cancel).toBeFocused();
  await page.mouse.click(2, 2);
  await expect(dialog).toBeVisible();

  await confirm.click();
  await expect(dialog).toBeHidden();
  await expect(page.locator("#import-model")).toBeFocused();
  await expect(removeModel).toBeHidden();
});

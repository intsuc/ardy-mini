// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

test("smoothly resets the camera, remains interruptible, and honors reduced motion", async ({
  page,
}) => {
  await page.goto("/");

  const state = await page.evaluate(async () => {
    const { SkeletonViewer } = await import("/src/viewer.ts");
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);
    const viewer = await SkeletonViewer.create(canvas);
    const internal = viewer as unknown as {
      camera: {
        position: {
          x: number;
          y: number;
          z: number;
        };
      };
      controls: {
        target: {
          x: number;
          y: number;
          z: number;
        };
      };
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
    const moveAway = () => {
      for (let step = 0; step < 15; step += 1) {
        viewer.moveCamera(1, 1);
      }
    };
    const waitForAnimationTime = async (milliseconds: number) => {
      const end = performance.now() + milliseconds;
      do {
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve()),
        );
      } while (performance.now() < end);
    };

    try {
      moveAway();
      const start = snapshot();
      viewer.resetCamera();
      const immediate = snapshot();
      await waitForAnimationTime(120);
      const midpoint = snapshot();
      await waitForAnimationTime(700);
      const completed = snapshot();

      moveAway();
      viewer.resetCamera();
      await waitForAnimationTime(100);
      viewer.moveCamera(1, 0);
      const interrupted = snapshot();
      await waitForAnimationTime(750);
      const afterInterrupt = snapshot();

      moveAway();
      viewer.resetCamera();
      await waitForAnimationTime(100);
      viewer.setReducedMotion(true);
      const reducedDuringTransition = snapshot();

      viewer.setReducedMotion(false);
      moveAway();
      viewer.setReducedMotion(true);
      viewer.resetCamera();
      const reduced = snapshot();

      return {
        start,
        immediate,
        midpoint,
        completed,
        interrupted,
        afterInterrupt,
        reducedDuringTransition,
        reduced,
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  const distance = (
    left: { camera: number[]; target: number[] },
    right: { camera: number[]; target: number[] },
  ) =>
    Math.hypot(
      ...left.camera.map((value, index) => value - right.camera[index]),
      ...left.target.map((value, index) => value - right.target[index]),
    );
  const expected = {
    camera: [3.1, 2.15, 3.4],
    target: [0, 0.85, 0],
  };

  expect(distance(state.immediate, state.start)).toBeCloseTo(0);
  expect(distance(state.midpoint, state.start)).toBeGreaterThan(0.01);
  expect(distance(state.midpoint, expected)).toBeGreaterThan(0.01);
  expect(distance(state.completed, expected)).toBeLessThan(1e-5);
  expect(distance(state.afterInterrupt, state.interrupted)).toBeLessThan(1e-5);
  expect(distance(state.reducedDuringTransition, expected)).toBeLessThan(1e-5);
  expect(distance(state.reduced, expected)).toBeLessThan(1e-5);
});

test("keeps a smooth reset anchored to a moving motion root", async ({
  page,
}) => {
  await page.goto("/");

  const state = await page.evaluate(async () => {
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
    const internal = viewer as unknown as {
      camera: { position: { toArray(): number[] } };
      controls: { target: { toArray(): number[] } };
    };

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
      for (let step = 0; step < 15; step += 1) {
        viewer.moveCamera(1, 1);
      }
      viewer.resetCamera();
      await new Promise<void>((resolve) => window.setTimeout(resolve, 100));
      viewer.seek(1);
      await new Promise<void>((resolve) => window.setTimeout(resolve, 750));
      return {
        camera: internal.camera.position.toArray(),
        target: internal.controls.target.toArray(),
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  expect(state.target[0]).toBeCloseTo(6);
  expect(state.target[1]).toBeCloseTo(0.7);
  expect(state.target[2]).toBeCloseTo(-4);
  expect(state.camera[0] - state.target[0]).toBeCloseTo(1.95);
  expect(state.camera[1] - state.target[1]).toBeCloseTo(1.2);
  expect(state.camera[2] - state.target[2]).toBeCloseTo(2.5);
});

test("updates the held heading when orbiting horizontally at the top-down pole", async ({
  page,
}) => {
  await page.goto("/");

  const state = await page.evaluate(async () => {
    const { SkeletonViewer } = await import("/src/viewer.ts");
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);
    const viewer = await SkeletonViewer.create(canvas);
    const internal = viewer as unknown as {
      camera: {
        position: {
          x: number;
          y: number;
          z: number;
        };
      };
      controls: {
        target: {
          x: number;
          y: number;
          z: number;
        };
        getAzimuthalAngle(): number;
        getPolarAngle(): number;
      };
    };
    const snapshot = () => ({
      camera: {
        x: internal.camera.position.x,
        y: internal.camera.position.y,
        z: internal.camera.position.z,
      },
      target: {
        x: internal.controls.target.x,
        y: internal.controls.target.y,
        z: internal.controls.target.z,
      },
      azimuth: internal.controls.getAzimuthalAngle(),
      polar: internal.controls.getPolarAngle(),
    });
    const movement = (
      before: ReturnType<typeof snapshot>,
      after: ReturnType<typeof snapshot>,
    ) => ({
      x: after.camera.x - before.camera.x,
      y: after.camera.y - before.camera.y,
      z: after.camera.z - before.camera.z,
      targetX: after.target.x - before.target.x,
      targetY: after.target.y - before.target.y,
      targetZ: after.target.z - before.target.z,
    });

    try {
      viewer.setReducedMotion(true);
      viewer.orbit(0, 100);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
      const beforeFirstMove = snapshot();
      viewer.moveCamera(1, 0);
      const afterFirstMove = snapshot();
      viewer.moveCamera(-1, 0);

      viewer.orbit(3, 0);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
      const beforeSecondMove = snapshot();
      viewer.moveCamera(1, 0);
      const afterSecondMove = snapshot();

      return {
        first: {
          before: beforeFirstMove,
          delta: movement(beforeFirstMove, afterFirstMove),
        },
        second: {
          before: beforeSecondMove,
          delta: movement(beforeSecondMove, afterSecondMove),
        },
      };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  const expectedDirection = (azimuth: number) => ({
    x: -Math.sin(azimuth),
    z: -Math.cos(azimuth),
  });
  const normalized = (movement: { x: number; z: number }) => {
    const length = Math.hypot(movement.x, movement.z);
    return { x: movement.x / length, z: movement.z / length };
  };
  const firstDirection = normalized(state.first.delta);
  const secondDirection = normalized(state.second.delta);
  const expectedFirst = expectedDirection(state.first.before.azimuth);
  const expectedSecond = expectedDirection(state.second.before.azimuth);

  expect(state.first.before.polar).toBeLessThan(1e-4);
  expect(state.second.before.polar).toBeLessThan(1e-4);
  expect(state.second.before.azimuth).not.toBeCloseTo(
    state.first.before.azimuth,
  );
  expect(firstDirection.x).toBeCloseTo(expectedFirst.x);
  expect(firstDirection.z).toBeCloseTo(expectedFirst.z);
  expect(secondDirection.x).toBeCloseTo(expectedSecond.x);
  expect(secondDirection.z).toBeCloseTo(expectedSecond.z);
  expect(
    firstDirection.x * secondDirection.x +
      firstDirection.z * secondDirection.z,
  ).toBeLessThan(0.98);
  for (const delta of [state.first.delta, state.second.delta]) {
    expect(delta.y).toBeCloseTo(0);
    expect(delta.targetX).toBeCloseTo(delta.x);
    expect(delta.targetY).toBeCloseTo(delta.y);
    expect(delta.targetZ).toBeCloseTo(delta.z);
  }
});

test("orbits past horizontal so the camera can look upward", async ({
  page,
}) => {
  await page.goto("/");

  const state = await page.evaluate(async () => {
    const { SkeletonViewer } = await import("/src/viewer.ts");
    const host = document.createElement("div");
    host.style.width = "320px";
    host.style.height = "320px";
    const canvas = document.createElement("canvas");
    host.append(canvas);
    document.body.append(host);
    const viewer = await SkeletonViewer.create(canvas);
    const internal = viewer as unknown as {
      camera: {
        position: {
          x: number;
          y: number;
          z: number;
        };
      };
      controls: {
        target: {
          x: number;
          y: number;
          z: number;
        };
        getDistance(): number;
        getPolarAngle(): number;
      };
    };

    const snapshot = () => ({
      cameraY: internal.camera.position.y,
      targetY: internal.controls.target.y,
      distance: internal.controls.getDistance(),
      polar: internal.controls.getPolarAngle(),
    });

    try {
      viewer.setReducedMotion(true);
      const before = snapshot();
      viewer.orbit(0, -4);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
      return { before, after: snapshot() };
    } finally {
      viewer.dispose();
      host.remove();
    }
  });

  expect(state.before.polar).toBeLessThan(Math.PI / 2);
  expect(state.before.cameraY).toBeGreaterThan(state.before.targetY);
  expect(state.after.polar).toBeGreaterThan(Math.PI / 2);
  expect(state.after.cameraY).toBeLessThan(state.after.targetY);
  expect(state.after.distance).toBeCloseTo(state.before.distance);
});

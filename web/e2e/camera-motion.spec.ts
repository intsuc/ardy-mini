// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

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

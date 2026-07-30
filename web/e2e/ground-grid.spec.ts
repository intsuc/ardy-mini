// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

import { waitForPreviewReady } from "./control-helpers";

test("aligns each major line with a minor line in the GPU shader", async ({
  page,
}) => {
  await page.goto("/");

  const result = await page.evaluate(async () => {
    const { renderGroundGridPhaseSamples } = await import(
      "/e2e/ground-grid-harness.ts"
    );
    return renderGroundGridPhaseSamples();
  });

  expect(result.origin[0]).toBeGreaterThan(100);
  expect(result.origin[1]).toBeGreaterThan(100);
  expect(result.origin[2]).toBeLessThan(10);
  expect(result.origin[3]).toBe(255);
  expect(result.betweenLines.slice(0, 3)).toEqual([0, 0, 0]);
  expect(result.nextMinorLine[0]).toBeLessThan(10);
  expect(result.nextMinorLine[1]).toBeGreaterThan(240);
  expect(result.nextMinorLine[2]).toBeLessThan(10);
  expect(result.nextMinorLine[3]).toBe(255);
  for (const minorLine of [
    result.beforeNextMajorLine,
    result.afterNextMajorLine,
  ]) {
    expect(minorLine[0]).toBeLessThan(10);
    expect(minorLine[1]).toBeGreaterThan(240);
    expect(minorLine[2]).toBeLessThan(10);
    expect(minorLine[3]).toBe(255);
  }
  // At x=5 both integer phases must coincide, with the major color replacing
  // the minor color exactly as it does at the origin.
  expect(result.nextMajorLine).toEqual(result.origin);
});

test("keeps the distant minor grid stable under a small camera rotation", async ({
  page,
}) => {
  await page.goto("/");

  const result = await page.evaluate(async () => {
    const { renderGroundGridObliqueDiagnostics } = await import(
      "/e2e/ground-grid-harness.ts"
    );
    return renderGroundGridObliqueDiagnostics();
  });

  expect(result.meanAbsoluteDelta).toBeLessThan(0.2);
  expect(result.maximumRowMeanStep).toBeLessThan(0.35);
});

test("dithers the production fog gradient without adding noise to the sky", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1_600, height: 1_000 });
  await page.goto("/");
  await waitForPreviewReady(page);
  await page.evaluate(async () => {
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() =>
        requestAnimationFrame(() => resolve()),
      ),
    );
  });

  const canvas = page.locator("#motion-canvas");
  const screenshot = await canvas.screenshot();
  const diagnostics = await page.evaluate(async (imageDataUrl) => {
    const image = new Image();
    image.src = imageDataUrl;
    await image.decode();
    const analysisCanvas = document.createElement("canvas");
    analysisCanvas.width = image.naturalWidth;
    analysisCanvas.height = image.naturalHeight;
    const context = analysisCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (!context) throw new Error("2D analysis context is unavailable.");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(
      0,
      0,
      analysisCanvas.width,
      analysisCanvas.height,
    ).data;

    const analyzeBand = (
      startYRatio: number,
      endYRatio: number,
    ): {
      horizontalEqualRatio: number;
      firstColor: number;
      maximumHorizontalRun: number;
      uniqueColors: number;
    } => {
      const startX = Math.floor(analysisCanvas.width * 0.1);
      const endX = Math.floor(analysisCanvas.width * 0.9);
      const startY = Math.floor(analysisCanvas.height * startYRatio);
      const endY = Math.floor(analysisCanvas.height * endYRatio);
      const colors = new Set<number>();
      let equalNeighbors = 0;
      let firstColor = -1;
      let neighborCount = 0;
      let maximumHorizontalRun = 0;

      for (let y = startY; y < endY; y += 1) {
        let previousColor = -1;
        let runLength = 0;
        for (let x = startX; x < endX; x += 1) {
          const offset = (y * analysisCanvas.width + x) * 4;
          const color =
            (pixels[offset] << 16) |
            (pixels[offset + 1] << 8) |
            pixels[offset + 2];
          colors.add(color);
          if (firstColor === -1) firstColor = color;
          if (color === previousColor) {
            runLength += 1;
            equalNeighbors += 1;
          } else {
            maximumHorizontalRun = Math.max(
              maximumHorizontalRun,
              runLength,
            );
            runLength = 1;
            previousColor = color;
          }
          if (x > startX) neighborCount += 1;
        }
        maximumHorizontalRun = Math.max(
          maximumHorizontalRun,
          runLength,
        );
      }

      return {
        horizontalEqualRatio: equalNeighbors / neighborCount,
        firstColor,
        maximumHorizontalRun,
        uniqueColors: colors.size,
      };
    };

    return {
      width: analysisCanvas.width,
      sky: analyzeBand(0.02, 0.1),
      distantGround: analyzeBand(0.16, 0.3),
    };
  }, `data:image/png;base64,${screenshot.toString("base64")}`);

  // The scene-pass depth mask leaves the near-black background uniform and
  // avoids turning the dither into visible grain in the sky.
  expect(diagnostics.sky.uniqueColors).toBe(1);
  expect(diagnostics.sky.firstColor).toBeLessThanOrEqual(0x010101);
  expect(diagnostics.sky.horizontalEqualRatio).toBe(1);
  // Before the post-transform dither, quantized fog bands occupied almost a
  // full row (985 px) and over 90% of horizontal neighbors were identical.
  expect(diagnostics.distantGround.maximumHorizontalRun).toBeLessThan(
    diagnostics.width * 0.15,
  );
  expect(
    diagnostics.distantGround.horizontalEqualRatio,
  ).toBeLessThan(0.7);
});

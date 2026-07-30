// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from "@playwright/test";

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
  expect(result.shaderErrors).toEqual([]);
});

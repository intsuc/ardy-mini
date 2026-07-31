// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import {
  openPreviewSettings,
  setSliderValue,
} from "./control-helpers";
import { MODEL_CACHE_PREFIX } from "../src/runtime/model-assets";
import { RESUMABLE_TRANSPORT_BLOCK_BYTES } from "../src/runtime/resumable-transport";

const configuredModelDirectory = process.env.ARDY_BROWSER_MODEL_DIR;
const reducedMotion = process.env.ARDY_BROWSER_REDUCED_MOTION === "1";
const operationTimeout = 20 * 60 * 1000;
const developmentModelFamilyPath =
  "/models/ardy-minilm-core40-browser-v1/";

async function runGeneration(
  page: Page,
  trigger: () => Promise<void>,
  expectedFrames: number,
): Promise<number> {
  const started = performance.now();
  const generate = page.locator("#generate");
  const diagnostics = page.locator("#preview-diagnostics");
  await trigger();
  await expect(generate).toHaveAttribute("aria-busy", "true");
  await expect(generate).not.toContainText("%");
  await expect(generate).toHaveAttribute("aria-busy", "false", {
    timeout: operationTimeout,
  });
  await expect(diagnostics).toHaveText(
    new RegExp(`^${expectedFrames} frames · \\d+ ms$`),
  );
  await expect(page.locator("#app-status")).toContainText(
    `session contains ${expectedFrames} frames`,
  );
  await expect(page.locator("#error-banner")).toHaveCount(0);
  return performance.now() - started;
}

test.describe("real browser model files", () => {
  test.skip(
    !configuredModelDirectory,
    "Set ARDY_BROWSER_MODEL_DIR to the exported model-family directory to opt into the real-model test.",
  );

  test("downloads individual files and exercises browser session generation", async ({
    page,
  }, testInfo) => {
    testInfo.setTimeout(45 * 60 * 1000);

    const consoleMessages: string[] = [];
    const pageErrors: string[] = [];
    const modelRequests: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "warning" || message.type() === "error") {
        consoleMessages.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("request", (request) => {
      const path = new URL(request.url()).pathname;
      if (path.includes(developmentModelFamilyPath)) {
        modelRequests.push(path);
      }
    });

    if (reducedMotion) await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");

    const environment = await page.evaluate(async () => {
      const gpu = (
        navigator as Navigator & {
          gpu?: {
            requestAdapter(): Promise<{
              info?: {
                vendor?: string;
                architecture?: string;
                device?: string;
                description?: string;
              };
            } | null>;
          };
        }
      ).gpu;
      const adapter = gpu ? await gpu.requestAdapter() : null;
      return {
        userAgent: navigator.userAgent,
        crossOriginIsolated: globalThis.crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency,
        deviceMemory:
          (navigator as Navigator & { deviceMemory?: number }).deviceMemory ??
          null,
        webgpu: gpu !== undefined,
        adapterAvailable: adapter !== null,
        features: adapter ? [...adapter.features] : [],
        adapter:
          adapter?.info === undefined
            ? null
            : {
                vendor: adapter.info.vendor ?? "",
                architecture: adapter.info.architecture ?? "",
                device: adapter.info.device ?? "",
                description: adapter.info.description ?? "",
              },
      };
    });
    expect(environment.crossOriginIsolated).toBe(true);
    expect(environment.webgpu).toBe(true);
    expect(environment.adapterAvailable).toBe(true);

    const startupDialog = page.getByRole("alertdialog");
    const confirmDownload = page.locator("#confirm-model-download");
    await expect(startupDialog).toBeVisible({
      timeout: operationTimeout,
    });
    await expect(startupDialog).toHaveAccessibleName(
      "Download model files?",
    );
    await expect(confirmDownload).toHaveText("Download model");
    const manifestRequests = modelRequests.filter((requestPath) =>
      requestPath.endsWith("/model.json.gz"),
    );
    expect(manifestRequests).toHaveLength(1);
    const selectedVariant = /^\/models\/ardy-minilm-core40-browser-v1\/(fp16|fp32)\/model\.json\.gz$/.exec(
      manifestRequests[0],
    )?.[1];
    expect(selectedVariant).toMatch(/^(?:fp16|fp32)$/);
    const expectedModelPath =
      `${developmentModelFamilyPath}${selectedVariant}/`;
    expect(
      modelRequests.some(
        (requestPath) =>
          requestPath.startsWith(developmentModelFamilyPath) &&
          !requestPath.startsWith(expectedModelPath),
      ),
    ).toBe(false);

    const loadStart = performance.now();
    await confirmDownload.click();
    await expect(page.locator(".workspace")).toHaveAttribute(
      "data-ready",
      "true",
      { timeout: operationTimeout },
    );
    await expect(startupDialog).toBeHidden();
    const loadWallMs = performance.now() - loadStart;
    const cacheEntries = await page.evaluate(async (cachePrefix) => {
      let count = 0;
      let maximumBytes = 0;
      let missingContentLength = 0;
      for (const cacheName of await caches.keys()) {
        if (!cacheName.startsWith(cachePrefix)) continue;
        const cache = await caches.open(cacheName);
        for (const request of await cache.keys()) {
          const response = await cache.match(request);
          const contentLength = response?.headers.get("content-length");
          if (
            contentLength === null ||
            contentLength === undefined ||
            !/^\d+$/.test(contentLength)
          ) {
            missingContentLength += 1;
            continue;
          }
          count += 1;
          maximumBytes = Math.max(maximumBytes, Number(contentLength));
        }
      }
      return { count, maximumBytes, missingContentLength };
    }, MODEL_CACHE_PREFIX);
    expect(cacheEntries.count).toBeGreaterThan(5);
    expect(cacheEntries.missingContentLength).toBe(0);
    expect(cacheEntries.maximumBytes).toBeLessThanOrEqual(
      RESUMABLE_TRANSPORT_BLOCK_BYTES,
    );
    for (const file of [
      "model.json.gz",
      "text_encoder.onnx",
      "denoiser.onnx",
      "decoder.onnx",
    ]) {
      expect(
        modelRequests.some((requestPath) =>
          requestPath ===
          `${expectedModelPath}${
            file.endsWith(".gz") ? file : `${file}.gz`
          }`,
        ),
      ).toBe(true);
    }
    const prompt = page.getByLabel("Motion description");
    await openPreviewSettings(page);
    const motionTab = page.getByRole("tab", {
      name: "Motion",
      exact: true,
    });
    await motionTab.click();
    await expect(motionTab).toHaveAttribute("aria-selected", "true");
    const seed = page.getByRole("spinbutton", { name: "Seed" });
    await prompt.fill("人物が歩く。");
    await seed.fill("-1");
    await page.locator("#generation-form").evaluate((element) => {
      (element as HTMLFormElement).requestSubmit();
    });
    await expect(page.locator("#prompt-error")).toBeEmpty();
    await expect(prompt).toHaveValue("人物が歩く。");
    await expect(prompt).not.toHaveAttribute("aria-invalid", "true");
    await expect(page.locator("#seed-error")).toContainText(
      "whole-number seed",
    );
    await expect(seed).toHaveAttribute("aria-invalid", "true");
    await expect(seed.locator("xpath=ancestor::*[@data-slot='field']")).toHaveAttribute(
      "data-invalid",
      "true",
    );
    await expect(seed).toBeFocused();

    await prompt.fill("A person walks forward confidently.");
    await seed.fill("2");
    await expect(page.locator("#seed-error")).toBeEmpty();
    await expect(seed).not.toHaveAttribute("aria-invalid", "true");
    await expect(seed.locator("xpath=ancestor::*[@data-slot='field']")).not.toHaveAttribute(
      "data-invalid",
      "true",
    );
    await seed.fill("-1");
    await expect(page.locator("#seed-error")).toContainText(
      "whole-number seed",
    );
    await seed.fill("2");
    await expect(page.locator("#seed-error")).toBeEmpty();

    await setSliderValue(page, "#target-buffer", 40);

    const timings: Record<string, number> = {};
    timings.initialGenerationWallMs = await runGeneration(
      page,
      () => page.locator("#generate").click(),
      40,
    );
    const firstGenerationUi = {
      diagnostics:
        (await page.locator("#preview-diagnostics").textContent()) ?? "",
    };
    expect(firstGenerationUi.diagnostics).toMatch(
      /^40 frames · \d+ ms$/,
    );
    await expect(page.locator("#error-banner")).toHaveCount(0);
    expect(pageErrors).toEqual([]);
    expect(
      consoleMessages.filter((message) => message.startsWith("error:")),
    ).toEqual([]);

    const ui = await page.evaluate(() => ({
      diagnostics:
        document.querySelector("#preview-diagnostics")?.textContent ?? "",
      status: document.querySelector("#app-status")?.textContent ?? "",
    }));
    await testInfo.attach("real-model-metrics.json", {
      body: Buffer.from(
        JSON.stringify(
          {
            reducedMotion,
            loadWallMs,
            timings,
            environment,
            selectedVariant,
            cacheEntries,
            modelRequests,
            firstGenerationUi,
            ui,
            consoleMessages,
            pageErrors,
          },
          null,
          2,
        ),
      ),
      contentType: "application/json",
    });
  });
});

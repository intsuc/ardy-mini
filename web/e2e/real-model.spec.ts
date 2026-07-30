// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import { setSliderValue } from "./control-helpers";

const configuredModelDirectory = process.env.ARDY_BROWSER_MODEL_DIR;
const reducedMotion = process.env.ARDY_BROWSER_REDUCED_MOTION === "1";
const operationTimeout = 20 * 60 * 1000;

async function runGeneration(
  page: Page,
  trigger: () => Promise<void>,
  expectedFrames: number,
): Promise<number> {
  const started = performance.now();
  await trigger();
  await expect(page.locator("#generate")).toHaveAttribute("aria-busy", "true");
  await expect(page.locator("#generate-label")).not.toContainText("%");
  await expect(page.locator("#generate")).toHaveAttribute("aria-busy", "false", {
    timeout: operationTimeout,
  });
  await expect(page.locator("#generation-stage")).toHaveText(
    `${expectedFrames} frames`,
  );
  await expect(page.locator("#generation-percent")).toHaveText(/^\d+ ms$/);
  await expect(page.locator("#app-status")).toContainText(
    `session contains ${expectedFrames} frames`,
  );
  await expect(page.locator("#error-banner")).toHaveCount(0);
  return performance.now() - started;
}

test.describe("real browser model files", () => {
  test.skip(
    !configuredModelDirectory,
    "Set ARDY_BROWSER_MODEL_DIR to the exported model directory to opt into the real-model test.",
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
      if (path.includes("/models/ardy-minilm-core40-browser-v1/")) {
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
    test.skip(
      !environment.features.includes("shader-f16"),
      "The real mixed-precision model requires a WebGPU adapter with shader-f16.",
    );

    const downloadDialog = page.getByRole("alertdialog", {
      name: "Download model files?",
    });
    await expect(downloadDialog).toBeVisible({
      timeout: operationTimeout,
    });
    await expect(page.locator("#model-cache-state")).toHaveText(
      "Not cached",
    );
    await expect(page.locator("#model-runtime-state")).toHaveText(
      "Not loaded",
    );

    const loadStart = performance.now();
    await downloadDialog
      .getByRole("button", { name: "Download model", exact: true })
      .click();
    await expect(page.locator("#model-runtime-state")).toHaveText("Ready", {
      timeout: operationTimeout,
    });
    await expect(page.locator("#model-cache-state")).toHaveText("Cached");
    const loadWallMs = performance.now() - loadStart;
    for (const file of [
      "model.json.gz",
      "text_encoder.onnx",
      "denoiser.onnx",
      "decoder.onnx",
    ]) {
      expect(
        modelRequests.some((requestPath) =>
          requestPath.endsWith(
            file.endsWith(".gz") ? file : `${file}.gz`,
          ),
        ),
      ).toBe(true);
    }
    const prompt = page.getByLabel("Motion description");
    const seed = page.getByRole("spinbutton", { name: "Seed" });
    await prompt.fill("人物が歩く。");
    await seed.fill("-1");
    await page.locator("#generate").click();
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
    await page.evaluate(() => {
      const stage = document.querySelector("#generation-stage");
      const progress = document.querySelector("#generation-progress");
      if (!stage || !progress) {
        throw new Error("Missing generation UI");
      }
      const stages = [stage.textContent ?? ""];
      new MutationObserver(() => stages.push(stage.textContent ?? "")).observe(
        stage,
        { childList: true, characterData: true, subtree: true },
      );
      let hiddenMutations = 0;
      new MutationObserver((records) => {
        hiddenMutations += records.length;
        (
          window as typeof window & {
            __ardyGenerationProgressHiddenMutations?: number;
          }
        ).__ardyGenerationProgressHiddenMutations = hiddenMutations;
      }).observe(progress, {
        attributes: true,
        attributeFilter: ["hidden"],
      });
      (
        window as typeof window & {
          __ardyGenerationStages?: string[];
          __ardyGenerationProgressHiddenMutations?: number;
        }
      ).__ardyGenerationStages = stages;
      (
        window as typeof window & {
          __ardyGenerationProgressHiddenMutations?: number;
        }
      ).__ardyGenerationProgressHiddenMutations = 0;
    });

    const timings: Record<string, number> = {};
    timings.initialGenerationWallMs = await runGeneration(
      page,
      () => page.locator("#generate").click(),
      40,
    );
    await expect(page.locator("#generate")).toBeFocused();
    const firstGenerationUi = await page.evaluate(() => ({
      stages:
        (
          window as typeof window & { __ardyGenerationStages?: string[] }
        ).__ardyGenerationStages ?? [],
    }));
    expect(firstGenerationUi.stages).toContain("Received 40 frames");
    await expect(page.locator("#generation-stage")).toHaveText("40 frames");
    await expect(page.locator("#generation-percent")).toHaveText(/^\d+ ms$/);
    await expect(page.locator("#generation-progress")).toHaveAttribute(
      "data-state",
      "complete",
    );
    await expect(page.locator("#generation-progress")).toBeVisible();
    await expect(page.locator("#error-banner")).toHaveCount(0);
    expect(
      await page.evaluate(
        () =>
          (
            window as typeof window & {
              __ardyGenerationProgressHiddenMutations?: number;
            }
          ).__ardyGenerationProgressHiddenMutations ?? 0,
      ),
    ).toBe(0);
    expect(pageErrors).toEqual([]);
    expect(
      consoleMessages.filter((message) => message.startsWith("error:")),
    ).toEqual([]);

    const ui = await page.evaluate(() => ({
      cache:
        document.querySelector("#model-cache-state")?.textContent ?? "",
      modelRuntime:
        document.querySelector("#model-runtime-state")?.textContent ?? "",
      frames: document.querySelector("#generation-stage")?.textContent ?? "",
      runtime:
        document.querySelector("#generation-percent")?.textContent ?? "",
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

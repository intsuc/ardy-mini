// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

import {
  openPreviewSettings,
  setCheckedState,
  setSliderValue,
} from "./control-helpers";

const configuredPack = process.env.ARDY_BROWSER_MODEL_PACK;
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
  await expect(page.locator("#model-error-banner")).toBeHidden();
  await expect(page.locator("#error-banner")).toBeHidden();
  return performance.now() - started;
}

test.describe("real browser model-pack", () => {
  test.skip(
    !configuredPack,
    "Set ARDY_BROWSER_MODEL_PACK to the exported .tar.gz archive to opt into the real-model test.",
  );

  test("loads the archive and exercises browser session generation", async ({
    page,
  }, testInfo) => {
    testInfo.setTimeout(45 * 60 * 1000);

    // This test covers inference rather than persistence. Avoid a second
    // OPFS copy while running in CI or on a developer workstation.
    await page.addInitScript(() => {
      try {
        Object.defineProperty(navigator, "storage", {
          configurable: true,
          value: {},
        });
      } catch {
        // A browser may expose a non-configurable StorageManager.
      }
    });

    const consoleMessages: string[] = [];
    const pageErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "warning" || message.type() === "error") {
        consoleMessages.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => pageErrors.push(error.message));

    if (reducedMotion) await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    await expect(page.locator("#model-state")).toHaveText("Not loaded");
    await expect(page.locator("#model-setup-help")).toContainText(
      "ardy-minilm-core40-browser-v1.tar.gz",
    );

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

    await page.evaluate(() => {
      const title = document.querySelector("#model-title");
      if (!title) throw new Error("Missing model title");
      const stages = [title.textContent ?? ""];
      new MutationObserver(() => stages.push(title.textContent ?? "")).observe(
        title,
        { childList: true, characterData: true, subtree: true },
      );
      (
        window as typeof window & { __ardyModelLoadStages?: string[] }
      ).__ardyModelLoadStages = stages;
    });
    const loadStart = performance.now();
    await page
      .locator("#model-file-input")
      .setInputFiles(path.resolve(configuredPack!));
    await expect(page.locator("#model-state")).toHaveText("Ready", {
      timeout: operationTimeout,
    });
    await expect(page.locator("#model-setup-help")).toBeHidden();
    await expect(page.locator("#import-model-label")).toHaveText(
      "Replace model pack",
    );
    const loadWallMs = performance.now() - loadStart;
    const modelLoadStages = await page.evaluate(
      () =>
        (
          window as typeof window & { __ardyModelLoadStages?: string[] }
        ).__ardyModelLoadStages ?? [],
    );
    const joinedLoadStages = modelLoadStages.join("\n");
    for (const graph of [
      "text_encoder.onnx",
      "denoiser.onnx",
      "decoder.onnx",
    ]) {
      expect(joinedLoadStages).toContain(graph);
    }
    await expect(page.locator("#model-detail")).toHaveText(
      "ardy-minilm-core40-browser-v1",
    );
    await expect(page.locator("#model-detail")).not.toContainText(
      /WebGPU|FPS/,
    );
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

    await setSliderValue(page, "#duration", 2);
    await setCheckedState(page, "#stream-generation", false);
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
    timings.replace40WallMs = await runGeneration(
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
    await openPreviewSettings(page);
    await expect(page.locator("#show-contacts")).toBeChecked();
    await expect(page.locator("#show-orientations")).toBeChecked();

    const playPause = page.locator("#play-pause");
    if ((await playPause.getAttribute("aria-label")) === "Pause motion") {
      await playPause.click();
    }
    await expect(playPause).toHaveAttribute("aria-label", "Play motion");

    await setSliderValue(page, "#target-buffer", 40);
    await setSliderValue(page, "#timeline", 39);
    timings.appendWallMs = await runGeneration(
      page,
      () => setCheckedState(page, "#stream-generation", true),
      80,
    );
    await setCheckedState(page, "#stream-generation", false);
    if ((await playPause.getAttribute("aria-label")) === "Pause motion") {
      await playPause.click();
    }
    await expect(playPause).toHaveAttribute("aria-label", "Play motion");

    await setSliderValue(page, "#timeline", 18);
    await expect(page.locator("#timeline").getByRole("slider")).toHaveAttribute(
      "aria-valuenow",
      "18",
    );
    timings.branchWallMs = await runGeneration(
      page,
      () => page.locator("#restart-from-now").click(),
      56,
    );

    await prompt.fill("A person turns left and waves.");
    timings.livePromptWallMs = await runGeneration(
      page,
      () => page.locator("#apply-prompt").click(),
      76,
    );
    await expect(prompt).toHaveValue("A person turns left and waves.");

    timings.finalReplaceWallMs = await runGeneration(
      page,
      () => page.locator("#restart-generation").click(),
      40,
    );
    await expect(page.locator("#generation-stage")).toHaveText("40 frames");
    if (reducedMotion) {
      await expect(playPause).toHaveAttribute("aria-label", "Play motion");
      await expect(page.locator("#loop-toggle")).toHaveAttribute(
        "aria-pressed",
        "false",
      );
      await expect(page.locator("#timeline").getByRole("slider")).toHaveAttribute(
        "aria-valuenow",
        "0",
      );
    } else {
      await expect(playPause).toHaveAttribute("aria-label", "Pause motion");
    }

    await expect(page.locator("#model-error-banner")).toBeHidden();
    await expect(page.locator("#error-banner")).toBeHidden();
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
      model: document.querySelector("#model-detail")?.textContent ?? "",
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
            modelLoadStages,
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

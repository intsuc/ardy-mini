// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

const configuredPack = process.env.ARDY_BROWSER_MODEL_PACK;
const configuredBackend = process.env.ARDY_BROWSER_BACKEND ?? "wasm";
const reducedMotion = process.env.ARDY_BROWSER_REDUCED_MOTION === "1";
const operationTimeout = 20 * 60 * 1000;
const webGpuLaunchArgs =
  configuredBackend === "webgpu"
    ? [
        "--enable-unsafe-webgpu",
        ...(process.platform === "linux"
          ? ["--use-angle=vulkan", "--enable-features=Vulkan"]
          : []),
      ]
    : [];

test.use({ launchOptions: { args: webGpuLaunchArgs } });

async function setRange(
  page: Page,
  selector: string,
  value: number,
): Promise<void> {
  await page.locator(selector).evaluate((element, nextValue) => {
    const input = element as HTMLInputElement;
    input.value = String(nextValue);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
}

async function runGeneration(
  page: Page,
  trigger: () => Promise<void>,
  expectedFrames: number,
): Promise<number> {
  const started = performance.now();
  await trigger();
  await expect(page.locator("#generate")).toHaveAttribute("aria-busy", "true");
  await expect(page.locator("#generate")).toHaveAttribute("aria-busy", "false", {
    timeout: operationTimeout,
  });
  await expect(page.locator("#motion-badge")).toHaveText(
    `${expectedFrames} frames · 20 FPS`,
  );
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
    "Set ARDY_BROWSER_MODEL_PACK to opt into the ~1.4 GiB four-graph real-model test.",
  );

  test("loads four sessions and exercises protocol-v2 session generation", async ({
    page,
  }, testInfo) => {
    testInfo.setTimeout(45 * 60 * 1000);
    expect(["auto", "webgpu", "wasm"]).toContain(configuredBackend);

    // This test covers inference rather than persistence. Avoid a second
    // ~1.4 GiB OPFS copy while running in CI or on a developer workstation.
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
    await page.waitForFunction(() => document.querySelector("#backend") !== null);
    await expect(page.locator(".setup-note")).toContainText(
      "about 1.4 GiB, four ONNX graphs",
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
    await page.evaluate((backend) => {
      const select = document.querySelector<HTMLSelectElement>("#backend");
      if (!select) throw new Error("Missing backend selector");
      select.value = backend;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }, configuredBackend);

    const loadStart = performance.now();
    await page
      .locator("#model-file-input")
      .setInputFiles(path.resolve(configuredPack!));
    await expect(page.locator("#model-state")).toHaveText("Ready", {
      timeout: operationTimeout,
    });
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
      "denoiser_constraints.onnx",
      "decoder.onnx",
    ]) {
      expect(joinedLoadStages).toContain(graph);
    }
    if (configuredBackend === "wasm") {
      await expect(page.locator("#model-detail")).toContainText("WebAssembly");
    } else if (configuredBackend === "webgpu") {
      await expect(page.locator("#model-detail")).toContainText("WebGPU");
    }
    await expect(page.locator("#constraint-track-root")).toBeEnabled();
    await expect(page.locator("#postprocess-enabled")).toBeEnabled();

    const prompt = page.getByLabel("Motion description");
    await prompt.fill("人物が歩く。");
    await page.locator("#generate").click();
    await expect(page.locator("#prompt-error")).toContainText(
      "typo-free English",
    );
    await expect(prompt).toBeFocused();

    await prompt.fill("A person walks forward confidently.");
    const seed = page.getByRole("spinbutton", { name: "Seed" });
    await seed.fill("-1");
    await page.locator("#generate").click();
    await expect(page.locator("#seed-error")).toContainText(
      "whole-number seed",
    );
    await expect(seed).toBeFocused();
    await seed.fill("2");

    await setRange(page, "#duration", 2);
    await page.locator("#stream-generation").uncheck();
    await page.getByText("Postprocess", { exact: true }).click();
    await page.locator("#postprocess-enabled").check();
    await page.evaluate(() => {
      const stage = document.querySelector("#generation-stage");
      const badge = document.querySelector("#motion-badge");
      if (!stage || !badge) throw new Error("Missing generation UI");
      const stages = [stage.textContent ?? ""];
      const badges = [badge.textContent ?? ""];
      new MutationObserver(() => stages.push(stage.textContent ?? "")).observe(
        stage,
        { childList: true, characterData: true, subtree: true },
      );
      new MutationObserver(() => badges.push(badge.textContent ?? "")).observe(
        badge,
        { childList: true, characterData: true, subtree: true },
      );
      (
        window as typeof window & {
          __ardyGenerationStages?: string[];
          __ardyMotionBadges?: string[];
        }
      ).__ardyGenerationStages = stages;
      (
        window as typeof window & {
          __ardyGenerationStages?: string[];
          __ardyMotionBadges?: string[];
        }
      ).__ardyMotionBadges = badges;
    });

    const timings: Record<string, number> = {};
    timings.replace40WallMs = await runGeneration(
      page,
      () => page.locator("#generate").click(),
      40,
    );
    const firstGenerationUi = await page.evaluate(() => ({
      stages:
        (
          window as typeof window & { __ardyGenerationStages?: string[] }
        ).__ardyGenerationStages ?? [],
      badges:
        (
          window as typeof window & { __ardyMotionBadges?: string[] }
        ).__ardyMotionBadges ?? [],
    }));
    expect(firstGenerationUi.stages).toContain("Received 40 frames");
    expect(firstGenerationUi.badges).toContain("40 frames · 20 FPS");
    await expect(page.locator("#generation-stage")).toHaveText(
      /40 session frames|Received 40 frames/,
    );
    await expect(page.locator("#runtime-metric")).toBeVisible();
    await expect(page.locator("#runtime-value")).toHaveAttribute(
      "title",
      /Foot slide/,
    );
    await expect(page.locator("#show-contacts")).toBeChecked();
    await page.locator("#show-orientations").check();
    await expect(page.locator("#show-orientations")).toBeChecked();
    await expect(page.locator("#export-session")).toBeEnabled();
    await expect(page.locator("#export-motion")).toBeEnabled();

    const playPause = page.locator("#play-pause");
    if ((await playPause.getAttribute("aria-label")) === "Pause motion") {
      await playPause.click();
    }
    await expect(playPause).toHaveAttribute("aria-label", "Play motion");

    await page.locator("#constraint-track-root").click();
    await page.locator("#constraint-type").selectOption("position");
    await page.locator("#constraint-frame").fill("60");
    await page.locator("#constraint-end-frame").fill("60");
    await page.locator("#add-constraint").click();
    await expect(page.locator("#constraint-track-root")).toHaveAttribute(
      "data-has-constraint",
      "true",
    );
    await expect(page.locator("#constraint-track-root")).toHaveAttribute(
      "title",
      /frame 60/,
    );
    await expect(page.locator("#app-status")).toContainText(
      "root position constraint at frames 60–60",
    );

    await setRange(page, "#target-buffer", 40);
    await setRange(page, "#timeline", 39);
    timings.appendWallMs = await runGeneration(
      page,
      () => page.locator("#stream-generation").check(),
      80,
    );
    const rootErrorAfter = Number(
      await page
        .locator("#correction-metric")
        .getAttribute("data-root-error-after"),
    );
    expect(Number.isFinite(rootErrorAfter)).toBe(true);
    expect(rootErrorAfter).toBeLessThanOrEqual(0.041);
    await expect(page.locator("#correction-metric")).toBeVisible();
    await expect(page.locator("#root-error-value")).toContainText("→");
    await expect(page.locator("#foot-slide-value")).toContainText("→");
    await page.locator("#stream-generation").uncheck();
    if ((await playPause.getAttribute("aria-label")) === "Pause motion") {
      await playPause.click();
    }
    await expect(playPause).toHaveAttribute("aria-label", "Play motion");

    await setRange(page, "#timeline", 18);
    await expect(page.locator("#timeline")).toHaveValue("18");
    timings.branchWallMs = await runGeneration(
      page,
      () => page.locator("#restart-from-now").click(),
      56,
    );

    await page.locator("#replan-buffer").fill("4");
    await prompt.fill("A person turns left and waves.");
    timings.livePromptWallMs = await runGeneration(
      page,
      () => page.locator("#apply-prompt").click(),
      60,
    );
    await expect(prompt).toHaveValue("A person turns left and waves.");

    timings.finalReplaceWallMs = await runGeneration(
      page,
      () => page.locator("#restart-generation").click(),
      40,
    );
    await expect(page.locator("#generation-stage")).toHaveText(
      /40 session frames|Received 40 frames/,
    );
    if (reducedMotion) {
      await expect(playPause).toHaveAttribute("aria-label", "Play motion");
      await expect(page.locator("#loop-toggle")).toHaveAttribute(
        "aria-pressed",
        "false",
      );
      await expect(page.locator("#timeline")).toHaveValue("0");
    } else {
      await expect(playPause).toHaveAttribute("aria-label", "Pause motion");
    }

    await expect(page.locator("#model-error-banner")).toBeHidden();
    await expect(page.locator("#error-banner")).toBeHidden();
    expect(pageErrors).toEqual([]);
    expect(
      consoleMessages.filter((message) => message.startsWith("error:")),
    ).toEqual([]);

    const ui = await page.evaluate(() => ({
      model: document.querySelector("#model-detail")?.textContent ?? "",
      motion: document.querySelector("#motion-badge")?.textContent ?? "",
      runtime: document.querySelector("#runtime-value")?.textContent ?? "",
      runtimeTitle:
        document.querySelector("#runtime-value")?.getAttribute("title") ?? "",
      status: document.querySelector("#app-status")?.textContent ?? "",
      rootConstraint:
        document
          .querySelector("#constraint-track-root")
          ?.getAttribute("title") ?? "",
      correction:
        document.querySelector("#correction-metric")?.textContent ?? "",
      rootErrorAfter:
        document
          .querySelector("#correction-metric")
          ?.getAttribute("data-root-error-after") ?? "",
      footSlideAfter:
        document
          .querySelector("#correction-metric")
          ?.getAttribute("data-foot-slide-after") ?? "",
    }));
    await testInfo.attach("real-model-metrics.json", {
      body: Buffer.from(
        JSON.stringify(
          {
            configuredBackend,
            reducedMotion,
            launchArgs: webGpuLaunchArgs,
            loadWallMs,
            timings,
            constraintRootErrorAfter: rootErrorAfter,
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

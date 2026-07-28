// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";

import { expect, test } from "@playwright/test";

const configuredPack = process.env.ARDY_BROWSER_MODEL_PACK;
const configuredBackend = process.env.ARDY_BROWSER_BACKEND ?? "wasm";
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

test.describe("real browser model-pack", () => {
  test.skip(
    !configuredPack,
    "Set ARDY_BROWSER_MODEL_PACK to opt into the ~800 MB real-model test.",
  );

  test("loads all sessions and generates a 40-frame clip", async ({
    page,
  }, testInfo) => {
    testInfo.setTimeout(20 * 60 * 1000);
    expect(["auto", "webgpu", "wasm"]).toContain(configuredBackend);

    // This test covers inference, not OPFS persistence, and avoids an additional
    // ~800 MB model-pack copy while running in CI or a developer workstation.
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
    page.on("console", (message) => {
      if (message.type() === "warning" || message.type() === "error") {
        consoleMessages.push(`${message.type()}: ${message.text()}`);
      }
    });
    await page.goto("/");
    await page.waitForFunction(() => document.querySelector("#backend") !== null);
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
    await page.evaluate((backend) => {
      const select = document.querySelector<HTMLSelectElement>("#backend");
      if (select === null) throw new Error("Missing backend selector");
      select.value = backend;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }, configuredBackend);

    const loadStart = performance.now();
    await page
      .locator("#model-file-input")
      .setInputFiles(path.resolve(configuredPack!));
    await expect(page.locator("#model-state")).toHaveText("Ready", {
      timeout: 20 * 60 * 1000,
    });
    const loadWallMs = performance.now() - loadStart;
    if (configuredBackend === "wasm") {
      await expect(page.locator("#model-detail")).toContainText("WebAssembly");
    } else if (configuredBackend === "webgpu") {
      await expect(page.locator("#model-detail")).toContainText("WebGPU");
    }

    await page.getByLabel("Motion prompt").fill(
      "A person walks forward confidently.",
    );
    await page.getByRole("spinbutton", { name: "Seed" }).fill("2");
    await page.locator("#duration").evaluate((element) => {
      const input = element as HTMLInputElement;
      input.value = "2";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const generationStart = performance.now();
    await page.getByRole("button", { name: "Generate motion" }).click();
    await expect(page.locator("#motion-badge")).toHaveAttribute(
      "data-state",
      "ready",
      { timeout: 20 * 60 * 1000 },
    );
    const generationWallMs = performance.now() - generationStart;
    await expect(page.locator("#motion-badge")).toContainText(
      "40 frames · 20 FPS · seed 2",
    );
    await expect(page.locator("#error-banner")).toBeHidden();

    const ui = await page.evaluate(() => ({
      model: document.querySelector("#model-detail")?.textContent ?? "",
      motion: document.querySelector("#motion-badge")?.textContent ?? "",
      runtime: document.querySelector("#runtime-value")?.textContent ?? "",
    }));
    await testInfo.attach("real-model-metrics.json", {
      body: Buffer.from(
        JSON.stringify(
          {
            configuredBackend,
            launchArgs: webGpuLaunchArgs,
            loadWallMs,
            generationWallMs,
            environment,
            ui,
            consoleMessages,
          },
          null,
          2,
        ),
      ),
      contentType: "application/json",
    });
  });
});

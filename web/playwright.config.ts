// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from "@playwright/test";

const webGpuLaunchArgs = [
  "--enable-unsafe-webgpu",
  ...(process.platform === "linux"
    ? ["--use-angle=vulkan", "--enable-features=Vulkan"]
    : []),
];

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  // Every page owns a native WebGPU device. Running many Chromium pages in
  // parallel can exhaust Vulkan device/queue resources and leave adapter
  // initialization pending, which is not representative of the single-page
  // application runtime.
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
    video: "off",
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: { args: webGpuLaunchArgs },
      },
    },
  ],
});

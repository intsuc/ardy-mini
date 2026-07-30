// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, type Page } from "@playwright/test"

export async function allowRequiredWebGpuFeatureForPreflight(
  page: Page
): Promise<void> {
  await page.addInitScript(() => {
    const gpu = (
      navigator as Navigator & {
        gpu?: {
          requestAdapter(...options: unknown[]): Promise<GPUAdapter | null>
        }
      }
    ).gpu
    if (!gpu) return

    const requestAdapter = gpu.requestAdapter.bind(gpu)
    let preflightPending = true
    Object.defineProperty(gpu, "requestAdapter", {
      configurable: true,
      value: async (...options: unknown[]) => {
        const adapter = await requestAdapter(...options)
        if (!adapter || !preflightPending) return adapter
        preflightPending = false

        const features = new Proxy(adapter.features, {
          get(target, property) {
            if (property === "has") {
              return (feature: GPUFeatureName) =>
                feature === "shader-f16" || target.has(feature)
            }
            const value = Reflect.get(target, property, target)
            return typeof value === "function" ? value.bind(target) : value
          },
        })
        return new Proxy(adapter, {
          get(target, property) {
            if (property === "features") return features
            const value = Reflect.get(target, property, target)
            return typeof value === "function" ? value.bind(target) : value
          },
        })
      },
    })
  })
}

export async function waitForPreviewReady(page: Page): Promise<void> {
  await expect(page.locator("#model-cache-state")).not.toHaveText(
    "Checking",
    { timeout: 15_000 }
  )
  const downloadDialog = page.getByRole("alertdialog", {
    name: "Download model files?",
  })
  if (await downloadDialog.isVisible()) {
    await downloadDialog
      .getByRole("button", { name: "Not now", exact: true })
      .click()
  }
  await expect(page.locator("#model-runtime-state")).toHaveText(
    /^(?:Not loaded|Unavailable)$/,
    { timeout: 15_000 }
  )
  await expect(page.locator("#import-vrm")).toBeEnabled()
}

export async function setSliderValue(
  page: Page,
  selector: string,
  value: number
): Promise<void> {
  const thumb = page.locator(selector).getByRole("slider")
  await expect(thumb).toBeEnabled()
  await thumb.focus()

  const minimum = Number(
    (await thumb.getAttribute("aria-valuemin")) ??
      (await thumb.getAttribute("min"))
  )
  const maximum = Number(
    (await thumb.getAttribute("aria-valuemax")) ??
      (await thumb.getAttribute("max"))
  )
  if (
    !Number.isFinite(minimum) ||
    !Number.isFinite(maximum) ||
    value < minimum ||
    value > maximum
  ) {
    throw new RangeError(
      `Slider target ${value} is outside ${minimum}–${maximum}.`
    )
  }

  await page.keyboard.press("Home")
  for (let attempt = 0; attempt < 1000; attempt += 1) {
    const current = Number(await thumb.getAttribute("aria-valuenow"))
    if (current === value) break
    const key = current < value ? "ArrowRight" : "ArrowLeft"
    await page.keyboard.press(key)
    const next = Number(await thumb.getAttribute("aria-valuenow"))
    if (next === current) {
      throw new RangeError(`Slider cannot reach ${value} from ${current}.`)
    }
  }

  await expect(thumb).toHaveAttribute("aria-valuenow", String(value))
}

export async function setCheckedState(
  page: Page,
  selector: string,
  checked: boolean
): Promise<void> {
  const control = page.locator(selector)
  const expected = String(checked)
  if ((await control.getAttribute("aria-checked")) !== expected) {
    await control.click()
  }
  await expect(control).toHaveAttribute("aria-checked", expected)
}

export async function openPreviewSettings(page: Page): Promise<void> {
  const trigger = page.locator("#preview-settings-trigger")
  await expect(trigger).toBeVisible()
  if ((await trigger.getAttribute("aria-expanded")) !== "true") {
    await trigger.click()
  }
  await expect(trigger).toHaveAttribute("aria-expanded", "true")
}

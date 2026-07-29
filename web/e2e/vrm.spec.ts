// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

interface TestVrmMetadata {
  readonly name: string;
  readonly version: string;
  readonly author: string;
}

const requiredHumanBones = {
  hips: { node: 0 },
  spine: { node: 1 },
  head: { node: 2 },
  leftUpperLeg: { node: 3 },
  leftLowerLeg: { node: 4 },
  leftFoot: { node: 5 },
  rightUpperLeg: { node: 6 },
  rightLowerLeg: { node: 7 },
  rightFoot: { node: 8 },
  leftUpperArm: { node: 9 },
  leftLowerArm: { node: 10 },
  leftHand: { node: 11 },
  rightUpperArm: { node: 12 },
  rightLowerArm: { node: 13 },
  rightHand: { node: 14 },
} as const;

/**
 * Builds a tiny, self-contained VRM 1.0 GLB. The fixture intentionally has no
 * mesh: this test covers local avatar loading and humanoid lifecycle without
 * checking a large binary model into the repository or fetching one at runtime.
 */
function createTestVrm(metadata: TestVrmMetadata): Buffer {
  const json = {
    asset: {
      version: "2.0",
      generator: "ARDY Playwright VRM fixture",
    },
    extensionsUsed: ["VRMC_vrm"],
    extensions: {
      VRMC_vrm: {
        specVersion: "1.0",
        meta: {
          name: metadata.name,
          version: metadata.version,
          authors: [metadata.author],
          copyrightInformation: "Copyright ARDY Playwright",
          contactInformation: "",
          references: [],
          thirdPartyLicenses: "",
          licenseUrl: "https://vrm.dev/licenses/1.0/",
          avatarPermission: "onlyAuthor",
          allowExcessivelyViolentUsage: false,
          allowExcessivelySexualUsage: false,
          commercialUsage: "personalNonProfit",
          allowPoliticalOrReligiousUsage: false,
          allowAntisocialOrHateUsage: false,
          creditNotation: "required",
          allowRedistribution: false,
          modification: "prohibited",
        },
        humanoid: {
          humanBones: requiredHumanBones,
        },
      },
    },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [
      {
        name: "Hips",
        translation: [0, 1, 0],
        children: [1, 3, 6],
      },
      {
        name: "Spine",
        translation: [0, 0.25, 0],
        children: [2, 9, 12],
      },
      { name: "Head", translation: [0, 0.45, 0] },
      {
        name: "LeftUpperLeg",
        translation: [0.1, -0.05, 0],
        children: [4],
      },
      {
        name: "LeftLowerLeg",
        translation: [0, -0.45, 0],
        children: [5],
      },
      { name: "LeftFoot", translation: [0, -0.45, 0.08] },
      {
        name: "RightUpperLeg",
        translation: [-0.1, -0.05, 0],
        children: [7],
      },
      {
        name: "RightLowerLeg",
        translation: [0, -0.45, 0],
        children: [8],
      },
      { name: "RightFoot", translation: [0, -0.45, 0.08] },
      {
        name: "LeftUpperArm",
        translation: [0.2, 0.3, 0],
        children: [10],
      },
      {
        name: "LeftLowerArm",
        translation: [0.3, 0, 0],
        children: [11],
      },
      { name: "LeftHand", translation: [0.25, 0, 0] },
      {
        name: "RightUpperArm",
        translation: [-0.2, 0.3, 0],
        children: [13],
      },
      {
        name: "RightLowerArm",
        translation: [-0.3, 0, 0],
        children: [14],
      },
      { name: "RightHand", translation: [-0.25, 0, 0] },
    ],
  };

  const jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  const jsonChunkLength = Math.ceil(jsonBytes.length / 4) * 4;
  const glb = Buffer.alloc(12 + 8 + jsonChunkLength, 0x20);

  glb.writeUInt32LE(0x46546c67, 0);
  glb.writeUInt32LE(2, 4);
  glb.writeUInt32LE(glb.length, 8);
  glb.writeUInt32LE(jsonChunkLength, 12);
  glb.writeUInt32LE(0x4e4f534a, 16);
  jsonBytes.copy(glb, 20);
  return glb;
}

async function openPreviewSettings(page: Page): Promise<void> {
  const settings = page.locator("#preview-settings");
  await expect(settings).toBeVisible();
  if (!(await settings.evaluate((element) => element.hasAttribute("open")))) {
    await settings.locator(":scope > summary").click();
  }
}

test("loads, hides, replaces, and removes a local VRM avatar", async ({
  page,
}) => {
  await page.goto("/");
  await openPreviewSettings(page);

  const state = page.locator("#vrm-state");
  const card = page.locator("#vrm-card");
  const showAvatar = page.getByRole("checkbox", {
    name: "Show VRM avatar",
  });

  await expect(state).toHaveText("Optional");
  await expect(card).toHaveAttribute("data-state", "missing");
  await expect(page.locator("#vrm-name")).toHaveText("No avatar loaded");
  await expect(page.locator("#remove-vrm")).toBeDisabled();
  await expect(showAvatar).toBeDisabled();

  await page.evaluate(() => {
    const stateElement = document.querySelector("#vrm-state");
    const cardElement = document.querySelector("#vrm-card");
    if (!stateElement || !cardElement) {
      throw new Error("Missing VRM status elements");
    }
    const states = [
      {
        label: stateElement.textContent ?? "",
        busy: cardElement.hasAttribute("aria-busy"),
      },
    ];
    new MutationObserver(() => {
      states.push({
        label: stateElement.textContent ?? "",
        busy: cardElement.hasAttribute("aria-busy"),
      });
    }).observe(cardElement.parentElement ?? cardElement, {
      attributes: true,
      attributeFilter: ["aria-busy", "data-state"],
      childList: true,
      characterData: true,
      subtree: true,
    });
    (
      window as typeof window & {
        __ardyVrmStates?: Array<{ label: string; busy: boolean }>;
      }
    ).__ardyVrmStates = states;
  });

  await page.locator("#vrm-file-input").setInputFiles({
    name: "synthetic-avatar.vrm",
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Synthetic Avatar",
      version: "1.2",
      author: "ARDY Test",
    }),
  });

  await expect(state).toHaveText("VRM 1.0");
  await expect(card).toHaveAttribute("data-state", "ready");
  await expect(card).not.toHaveAttribute("aria-busy", "");
  await expect(page.locator("#vrm-name")).toHaveText("Synthetic Avatar");
  await expect(page.locator("#vrm-detail")).toHaveText(
    "VRM 1.0 · model 1.2 · by ARDY Test · local preview",
  );
  await expect(page.locator("#import-vrm-label")).toHaveText("Replace VRM");
  await expect(page.locator("#remove-vrm")).toBeEnabled();
  await expect(showAvatar).toBeEnabled();
  await expect(showAvatar).toBeChecked();
  await expect(page.locator("#vrm-error-banner")).toBeHidden();

  const observedStates = await page.evaluate(
    () =>
      (
        window as typeof window & {
          __ardyVrmStates?: Array<{ label: string; busy: boolean }>;
        }
      ).__ardyVrmStates ?? [],
  );
  expect(observedStates).toContainEqual({ label: "Loading", busy: true });

  await showAvatar.uncheck();
  await expect(showAvatar).not.toBeChecked();
  await expect(page.locator("#app-status")).toContainText("VRM avatar hidden.");
  await showAvatar.check();
  await expect(showAvatar).toBeChecked();
  await expect(page.locator("#app-status")).toContainText("VRM avatar shown.");

  await page.locator("#vrm-file-input").setInputFiles({
    name: "replacement.vrm",
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Replacement Avatar",
      version: "2.0",
      author: "Second Author",
    }),
  });
  await expect(page.locator("#vrm-name")).toHaveText("Replacement Avatar");
  await expect(page.locator("#vrm-detail")).toContainText("model 2.0");
  await expect(page.locator("#vrm-detail")).toContainText("by Second Author");
  await expect(showAvatar).toBeChecked();

  await page.locator("#remove-vrm").click();
  await expect(state).toHaveText("Optional");
  await expect(card).toHaveAttribute("data-state", "missing");
  await expect(page.locator("#vrm-name")).toHaveText("No avatar loaded");
  await expect(page.locator("#import-vrm-label")).toHaveText("Load VRM");
  await expect(page.locator("#remove-vrm")).toBeDisabled();
  await expect(showAvatar).toBeDisabled();
  await expect(page.locator("#app-status")).toContainText(
    "Removed the VRM avatar.",
  );
});

test("focuses a dismissible error when the selected VRM is invalid", async ({
  page,
}) => {
  await page.goto("/");
  await openPreviewSettings(page);

  const fileChooserPromise = page.waitForEvent("filechooser");
  await page.locator("#import-vrm").click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles({
    name: "empty.vrm",
    mimeType: "model/gltf-binary",
    buffer: Buffer.alloc(0),
  });

  const error = page.locator("#vrm-error-banner");
  await expect(error).toBeVisible();
  await expect(error).toBeFocused();
  await expect(page.locator("#vrm-error-message")).toHaveText(
    "The selected VRM file is empty.",
  );
  await expect(page.locator("#vrm-state")).toHaveText("Optional");
  await expect(page.locator("#vrm-name")).toHaveText("No avatar loaded");

  await page.locator("#dismiss-vrm-error").click();
  await expect(error).toBeHidden();
});

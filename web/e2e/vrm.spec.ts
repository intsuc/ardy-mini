// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import {
  openPreviewSettings,
  setCheckedState,
} from "./control-helpers";

interface TestVrmMetadata {
  readonly name: string;
  readonly version: string;
  readonly author: string;
}

interface DragFile {
  readonly name: string;
  readonly mimeType: string;
  readonly buffer: Buffer;
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

async function beginPageFileDrag(
  page: Page,
  files: readonly DragFile[],
) {
  const dataTransfer = await page.evaluateHandle(
    (serializedFiles) => {
      const transfer = new DataTransfer();
      for (const file of serializedFiles) {
        transfer.items.add(
          new File(
            [Uint8Array.from(file.bytes)],
            file.name,
            { type: file.mimeType },
          ),
        );
      }
      return transfer;
    },
    files.map((file) => ({
      name: file.name,
      mimeType: file.mimeType,
      bytes: Array.from(file.buffer),
    })),
  );
  await page.dispatchEvent("#app", "dragenter", { dataTransfer });
  await page.dispatchEvent("#app", "dragover", { dataTransfer });
  return dataTransfer;
}

async function dropPageFiles(
  page: Page,
  dataTransfer: Awaited<ReturnType<typeof beginPageFileDrag>>,
) {
  await page.dispatchEvent("#app", "drop", { dataTransfer });
}

test("loads, hides, replaces, and removes a local VRM avatar", async ({
  page,
}) => {
  await page.goto("/");
  await openPreviewSettings(page);

  const card = page.locator("#vrm-card");
  const name = page.locator("#vrm-name");
  const detail = page.locator("#vrm-detail");
  const dropTarget = page.locator("#vrm-drop-target");
  const showAvatar = page.getByRole("switch", {
    name: "Show VRM avatar",
  });

  await expect(page.locator("#vrm-state")).toHaveCount(0);
  await expect(card).toHaveAttribute("data-state", "missing");
  await expect(name).toHaveText("No avatar loaded");
  await expect(detail).toHaveText("Load a VRM 0.x or 1.0 file.");
  await expect(page.locator("#remove-vrm")).toBeDisabled();
  await expect(showAvatar).toBeDisabled();
  await expect(dropTarget).toBeHidden();

  await page.locator("#vrm-file-input").setInputFiles({
    name: "synthetic-avatar.vrm",
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Synthetic Avatar",
      version: "1.2",
      author: "ARDY Test",
    }),
  });

  await expect(card).toHaveAttribute("data-state", "ready");
  await expect(card).not.toHaveAttribute("aria-busy", "");
  await expect(name).toHaveText("Synthetic Avatar");
  await expect(detail).toHaveText(
    "VRM 1.0 · model 1.2 · by ARDY Test",
  );
  await expect(page.locator("#import-vrm-label")).toHaveText("Replace VRM");
  await expect(page.locator("#remove-vrm")).toBeEnabled();
  await expect(showAvatar).toBeEnabled();
  await expect(showAvatar).toBeChecked();
  await expect(page.locator("#vrm-error-banner")).toBeHidden();

  await setCheckedState(page, "#show-vrm", false);
  await expect(showAvatar).not.toBeChecked();
  await expect(page.locator("#app-status")).toContainText("VRM avatar hidden.");
  await setCheckedState(page, "#show-vrm", true);
  await expect(showAvatar).toBeChecked();
  await expect(page.locator("#app-status")).toContainText("VRM avatar shown.");

  const replacementTransfer = await beginPageFileDrag(page, [
    {
      name: "replacement.vrm",
      mimeType: "model/gltf-binary",
      buffer: createTestVrm({
        name: "Replacement Avatar",
        version: "2.0",
        author: "Second Author",
      }),
    },
  ]);
  try {
    await expect(dropTarget).toBeVisible();
    await dropPageFiles(page, replacementTransfer);
  } finally {
    await replacementTransfer.dispose();
  }

  await expect(dropTarget).toBeHidden();
  await expect(name).toHaveText("Replacement Avatar");
  await expect(detail).toHaveText(
    "VRM 1.0 · model 2.0 · by Second Author",
  );
  await expect(showAvatar).toBeChecked();

  await page.locator("#remove-vrm").click();
  await expect(card).toHaveAttribute("data-state", "missing");
  await expect(name).toHaveText("No avatar loaded");
  await expect(detail).toHaveText("Load a VRM 0.x or 1.0 file.");
  await expect(page.locator("#import-vrm-label")).toHaveText("Load VRM");
  await expect(page.locator("#remove-vrm")).toBeDisabled();
  await expect(showAvatar).toBeDisabled();
  await expect(page.locator("#app-status")).toContainText(
    "Removed the VRM avatar.",
  );
});

test("preserves the current avatar and focuses an error for an invalid VRM drop", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#preview-settings-trigger")).toHaveAttribute(
    "aria-expanded",
    "false",
  );

  await page.locator("#vrm-file-input").setInputFiles({
    name: "stable-avatar.vrm",
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Stable Avatar",
      version: "1.0",
      author: "ARDY Test",
    }),
  });
  await expect(page.locator("#vrm-name")).toHaveText("Stable Avatar");

  const dropTarget = page.locator("#vrm-drop-target");
  const invalidTransfer = await beginPageFileDrag(page, [
    {
      name: "empty.vrm",
      mimeType: "model/gltf-binary",
      buffer: Buffer.alloc(0),
    },
  ]);
  try {
    await expect(dropTarget).toBeVisible();
    await dropPageFiles(page, invalidTransfer);
  } finally {
    await invalidTransfer.dispose();
  }

  const error = page.locator("#vrm-error-banner");
  await expect(dropTarget).toBeHidden();
  await expect(page.locator("#preview-settings-trigger")).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expect(error).toBeVisible();
  await expect(error).toBeFocused();
  await expect(page.locator("#vrm-error-message")).toHaveText(
    "The selected VRM file is empty.",
  );
  await expect(page.locator("#vrm-name")).toHaveText("Stable Avatar");
  await expect(page.locator("#vrm-detail")).toHaveText(
    "VRM 1.0 · model 1.0 · by ARDY Test",
  );

  await page.locator("#dismiss-vrm-error").click();
  await expect(error).toBeHidden();
});

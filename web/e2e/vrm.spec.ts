// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page } from "@playwright/test";

import {
  allowRequiredWebGpuFeatureForPreflight,
  openViewSettings,
  setCheckedState,
  waitForPreviewReady,
} from "./control-helpers";

test.beforeEach(async ({ page }) => {
  await allowRequiredWebGpuFeatureForPreflight(page);
});

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

/** Builds a tiny, self-contained VRM 1.0 GLB with one MToon triangle. */
function createTestVrm(metadata: TestVrmMetadata): Buffer {
  const binaryBytes = Buffer.alloc(42);
  [
    -0.15, 0, 0,
    0.15, 0, 0,
    0, 0.3, 0,
  ].forEach((value, index) => {
    binaryBytes.writeFloatLE(value, index * 4);
  });
  [0, 1, 2].forEach((value, index) => {
    binaryBytes.writeUInt16LE(value, 36 + index * 2);
  });
  const json = {
    asset: {
      version: "2.0",
      generator: "ARDY Playwright VRM fixture",
    },
    extensionsUsed: ["VRMC_vrm", "VRMC_materials_mtoon"],
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
    buffers: [{ byteLength: binaryBytes.length }],
    bufferViews: [
      {
        buffer: 0,
        byteOffset: 0,
        byteLength: 36,
        target: 34962,
      },
      {
        buffer: 0,
        byteOffset: 36,
        byteLength: 6,
        target: 34963,
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 3,
        type: "VEC3",
        min: [-0.15, 0, 0],
        max: [0.15, 0.3, 0],
      },
      {
        bufferView: 1,
        componentType: 5123,
        count: 3,
        type: "SCALAR",
        min: [0],
        max: [2],
      },
    ],
    materials: [
      {
        name: "MToon test material",
        pbrMetallicRoughness: {
          baseColorFactor: [0.8, 0.4, 0.2, 1],
          metallicFactor: 0,
          roughnessFactor: 1,
        },
        extensions: {
          VRMC_materials_mtoon: {
            specVersion: "1.0",
          },
        },
      },
    ],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0 },
            indices: 1,
            material: 0,
          },
        ],
      },
    ],
    nodes: [
      {
        name: "Hips",
        translation: [0, 1, 0],
        children: [1, 3, 6, 15],
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
      {
        name: "MToonMesh",
        mesh: 0,
        translation: [0, 0.1, 0],
      },
    ],
  };

  const jsonBytes = Buffer.from(JSON.stringify(json), "utf8");
  const jsonChunkLength = Math.ceil(jsonBytes.length / 4) * 4;
  const binaryChunkLength = Math.ceil(binaryBytes.length / 4) * 4;
  const binaryChunkOffset = 20 + jsonChunkLength;
  const glb = Buffer.alloc(
    12 + 8 + jsonChunkLength + 8 + binaryChunkLength,
    0,
  );

  glb.writeUInt32LE(0x46546c67, 0);
  glb.writeUInt32LE(2, 4);
  glb.writeUInt32LE(glb.length, 8);
  glb.writeUInt32LE(jsonChunkLength, 12);
  glb.writeUInt32LE(0x4e4f534a, 16);
  glb.fill(0x20, 20, binaryChunkOffset);
  jsonBytes.copy(glb, 20);
  glb.writeUInt32LE(binaryChunkLength, binaryChunkOffset);
  glb.writeUInt32LE(0x004e4942, binaryChunkOffset + 4);
  binaryBytes.copy(glb, binaryChunkOffset + 8);
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

test("loads MToon VRM materials as WebGPU-compatible node materials", async ({
  page,
}) => {
  await page.goto("/");
  const fixture = createTestVrm({
    name: "MToon Contract",
    version: "1.0",
    author: "ARDY Test",
  });

  const materials = await page.evaluate(async (bytes) => {
    const { loadVrmAvatar } = await import("/src/vrm-loader.ts");
    const loaded = await loadVrmAvatar(
      new File(
        [Uint8Array.from(bytes)],
        "mtoon-contract.vrm",
        { type: "model/gltf-binary" },
      ),
    );
    const result: Array<{
      isMToonNodeMaterial: boolean;
      isNodeMaterial: boolean;
      isShaderMaterial: boolean;
    }> = [];
    try {
      loaded.vrm.scene.traverse((object) => {
        const mesh = object as unknown as {
          readonly isMesh?: boolean;
          readonly material?:
            | {
                readonly isMToonNodeMaterial?: boolean;
                readonly isNodeMaterial?: boolean;
                readonly isShaderMaterial?: boolean;
              }
            | readonly {
                readonly isMToonNodeMaterial?: boolean;
                readonly isNodeMaterial?: boolean;
                readonly isShaderMaterial?: boolean;
              }[];
        };
        if (!mesh.isMesh || !mesh.material) return;
        const meshMaterials = Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material];
        for (const material of meshMaterials) {
          result.push({
            isMToonNodeMaterial:
              material.isMToonNodeMaterial === true,
            isNodeMaterial: material.isNodeMaterial === true,
            isShaderMaterial: material.isShaderMaterial === true,
          });
        }
      });
      return result;
    } finally {
      loaded.utils.deepDispose(loaded.vrm.scene);
    }
  }, [...fixture]);

  expect(materials).toEqual([
    {
      isMToonNodeMaterial: true,
      isNodeMaterial: true,
      isShaderMaterial: false,
    },
  ]);
});

test("shows the selected VRM loading state until the avatar is ready", async ({
  page,
}) => {
  await page.goto("/");
  await waitForPreviewReady(page);
  await openViewSettings(page);

  await page.evaluate(async () => {
    const { SkeletonViewer } = await import("/src/viewer.ts");
    const prototype = SkeletonViewer.prototype;
    const originalLoadVrm = prototype.loadVrm;
    let releaseLoad!: () => void;
    const loadGate = new Promise<void>((resolve) => {
      releaseLoad = resolve;
    });
    (
      globalThis as typeof globalThis & {
        __releaseVrmLoad?: () => void;
      }
    ).__releaseVrmLoad = releaseLoad;
    prototype.loadVrm = async function (file: File) {
      await loadGate;
      prototype.loadVrm = originalLoadVrm;
      return originalLoadVrm.call(this, file);
    };
  });

  const fileName = "loading-state-avatar.vrm";
  await page.locator("#vrm-file-input").setInputFiles({
    name: fileName,
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Loading State Avatar",
      version: "1.0",
      author: "ARDY Test",
    }),
  });

  const loadingStatus = page.locator("#vrm-loading-status");
  const viewport = page.locator("#viewport");
  const card = page.locator("#vrm-card");
  const importVrm = page.locator("#import-vrm");
  const removeVrm = page.locator("#remove-vrm");

  await expect(loadingStatus).toBeVisible();
  await expect(loadingStatus).toContainText("Loading VRM avatar");
  await expect(page.locator("#vrm-loading-file")).toHaveText(fileName);
  await expect(viewport).toHaveAttribute("aria-busy", "true");
  await expect(card).toHaveAttribute("aria-busy", "true");
  await expect(importVrm).toBeDisabled();
  await expect(removeVrm).toBeDisabled();

  await page.evaluate(() => {
    const testWindow = globalThis as typeof globalThis & {
      __releaseVrmLoad?: () => void;
    };
    if (!testWindow.__releaseVrmLoad) {
      throw new Error("VRM load gate is unavailable.");
    }
    testWindow.__releaseVrmLoad();
    delete testWindow.__releaseVrmLoad;
  });

  await expect(card).toHaveAttribute("data-state", "ready");
  await expect(page.locator("#vrm-name")).toHaveText(
    "Loading State Avatar",
  );
  await expect(loadingStatus).toBeHidden();
  await expect(page.locator("#vrm-loading-file")).toHaveText("");
  await expect(viewport).not.toHaveAttribute("aria-busy", "true");
  await expect(card).not.toHaveAttribute("aria-busy", "true");
  await expect(importVrm).toBeEnabled();
  await expect(removeVrm).toBeEnabled();
  await expect(page.locator("#import-vrm-label")).toHaveText("Replace VRM");
});

test("loads, hides, replaces, and removes a local VRM avatar", async ({
  page,
}) => {
  const renderErrors: string[] = [];
  page.on("pageerror", (error) => renderErrors.push(error.message));
  page.on("console", (message) => {
    if (
      (message.type() === "error" &&
        /(?:WebGPU|WGSL|shader|validation)/i.test(message.text())) ||
      (message.type() === "warning" &&
        message.text().includes('"transformedNormalView" is deprecated'))
    ) {
      renderErrors.push(message.text());
    }
  });
  await page.goto("/");
  await waitForPreviewReady(page);
  await openViewSettings(page);

  const settings = page.locator("#preview-settings");
  const card = page.locator("#vrm-card");
  const name = page.locator("#vrm-name");
  const detail = page.locator("#vrm-detail");
  const dropTarget = page.locator("#vrm-drop-target");
  const showAvatar = page.getByRole("checkbox", {
    name: "Show VRM avatar",
  });

  await expect(settings).toHaveAttribute("data-slot", "popover-content");
  await expect(settings).toBeVisible();
  await expect(page.locator("#vrm-state")).toHaveCount(0);
  await expect(card).toHaveAttribute("data-state", "missing");
  await expect(name).toHaveText("No avatar loaded");
  await expect(detail).toHaveText("Load a VRM 0.x or 1.0 file.");
  await expect(page.locator("#import-vrm svg")).toHaveCount(0);
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
  expect(renderErrors).toEqual([]);
});

test("wraps a long unbroken VRM author without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await waitForPreviewReady(page);
  await openViewSettings(page);

  const author = "ExtremelyLongUnbrokenVrmAuthorName".repeat(20);
  await page.locator("#vrm-file-input").setInputFiles({
    name: "long-author.vrm",
    mimeType: "model/gltf-binary",
    buffer: createTestVrm({
      name: "Long Author Avatar",
      version: "1.0",
      author,
    }),
  });

  const detail = page.locator("#vrm-detail");
  await expect(detail).toHaveText(`VRM 1.0 · model 1.0 · by ${author}`);
  const geometry = await detail.evaluate((element) => {
    const card = element.closest<HTMLElement>("#vrm-card");
    if (!card) throw new Error("VRM card is missing.");
    const detailRect = element.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();
    return {
      cardClientWidth: card.clientWidth,
      cardScrollWidth: card.scrollWidth,
      detailClientWidth: element.clientWidth,
      detailScrollWidth: element.scrollWidth,
      detailLeft: detailRect.left,
      detailRight: detailRect.right,
      cardLeft: cardRect.left,
      cardRight: cardRect.right,
      documentScrollWidth: document.documentElement.scrollWidth,
      viewportWidth: window.innerWidth,
    };
  });

  expect(geometry.cardScrollWidth).toBeLessThanOrEqual(
    geometry.cardClientWidth + 1,
  );
  expect(geometry.detailScrollWidth).toBeLessThanOrEqual(
    geometry.detailClientWidth + 1,
  );
  expect(geometry.detailLeft).toBeGreaterThanOrEqual(geometry.cardLeft - 1);
  expect(geometry.detailRight).toBeLessThanOrEqual(geometry.cardRight + 1);
  expect(geometry.documentScrollWidth).toBeLessThanOrEqual(
    geometry.viewportWidth,
  );
});

test("preserves the current avatar and focuses an error for an invalid VRM drop", async ({
  page,
}) => {
  await page.goto("/");
  await waitForPreviewReady(page);
  const settings = page.locator("#preview-settings");
  const settingsTrigger = page.locator("#settings-trigger");
  await openViewSettings(page);
  await settingsTrigger.click();
  await expect(settingsTrigger).toHaveAttribute(
    "aria-expanded",
    "false",
  );
  await expect(settings).toHaveAttribute("data-slot", "popover-content");
  await expect(settings).toBeHidden();

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
  await expect(settingsTrigger).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expect(settings).toBeVisible();
  await expect(
    page.getByRole("tab", { name: "View", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
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

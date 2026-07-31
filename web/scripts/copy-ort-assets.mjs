// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { access, copyFile, mkdir, readFile, readdir, unlink } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const expectedVersion = "1.27.0";
const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const monorepoRoot = fileURLToPath(new URL("../../", import.meta.url));
const ortRoot = fileURLToPath(new URL("../node_modules/onnxruntime-web/", import.meta.url));
const ortDist = fileURLToPath(new URL("../node_modules/onnxruntime-web/dist/", import.meta.url));
const tokenizerRoot = fileURLToPath(
  new URL("../node_modules/@huggingface/tokenizers/", import.meta.url),
);
const notoSansRoot = fileURLToPath(
  new URL("../node_modules/@fontsource-variable/noto-sans/", import.meta.url),
);
const threeRoot = fileURLToPath(new URL("../node_modules/three/", import.meta.url));
const destination = fileURLToPath(new URL("../public/ort/", import.meta.url));
const noticeDestination = fileURLToPath(new URL("../public/notices/", import.meta.url));
const packageMetadata = JSON.parse(await readFile(`${ortRoot}package.json`, "utf8"));

async function sourceDistributionRoot() {
  for (const candidate of [packageRoot, monorepoRoot]) {
    try {
      await access(`${candidate}LICENSE`);
      await access(`${candidate}NOTICE`);
      await access(`${candidate}ATTRIBUTIONS.MD`);
      await access(`${candidate}THIRD_PARTY_MODELS_AND_DATA.md`);
      return candidate;
    } catch {
      // Try the monorepo layout after the standalone Static Space layout.
    }
  }
  throw new Error(
    "Source distribution notices are missing. Expected them beside package.json or in the repository root.",
  );
}

const repositoryRoot = await sourceDistributionRoot();

if (packageMetadata.version !== expectedVersion) {
  throw new Error(
    `Expected onnxruntime-web ${expectedVersion}, found ${String(packageMetadata.version)}. ` +
      "Keep the runtime JavaScript and copied WASM assets on exactly the same version.",
  );
}

// `onnxruntime-web/webgpu` uses this Asyncify WebAssembly core to host its
// WebGPU execution provider. Shipping it does not enable a selectable
// WebAssembly execution provider or a CPU fallback session.
const assets = [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
];
await Promise.all(
  assets.map(async (name) => {
    try {
      await access(`${ortDist}${name}`);
    } catch {
      throw new Error(
        `ONNX Runtime Web ${expectedVersion} is missing the required asset ${name}.`,
      );
    }
  }),
);

await mkdir(destination, { recursive: true });
for (const existing of await readdir(destination)) {
  if (/^ort-wasm.*\.(?:mjs|wasm)$/.test(existing)) {
    await unlink(`${destination}${existing}`);
  }
}
await Promise.all(assets.map((name) => copyFile(`${ortDist}${name}`, `${destination}${name}`)));

await mkdir(noticeDestination, { recursive: true });
const notices = [
  [`${repositoryRoot}LICENSE`, `${noticeDestination}LICENSE`],
  [`${repositoryRoot}NOTICE`, `${noticeDestination}NOTICE`],
  [`${repositoryRoot}ATTRIBUTIONS.MD`, `${noticeDestination}ATTRIBUTIONS.MD`],
  [
    `${repositoryRoot}THIRD_PARTY_MODELS_AND_DATA.md`,
    `${noticeDestination}THIRD_PARTY_MODELS_AND_DATA.md`,
  ],
  [`${packageRoot}ONNXRUNTIME_LICENSE.txt`, `${noticeDestination}ONNXRUNTIME_LICENSE.txt`],
  [`${notoSansRoot}LICENSE`, `${noticeDestination}NOTO_SANS_LICENSE.txt`],
  [`${tokenizerRoot}LICENSE`, `${noticeDestination}TOKENIZERS_LICENSE.txt`],
  [`${threeRoot}LICENSE`, `${noticeDestination}THREE_LICENSE.txt`],
];
await Promise.all(notices.map(([source, target]) => copyFile(source, target)));

console.log(`Copied ${assets.length} ONNX Runtime Web ${expectedVersion} assets into ${destination}`);
console.log(`Copied ${notices.length} source and dependency notices into ${noticeDestination}`);
console.log(`Web app root: ${packageRoot}`);

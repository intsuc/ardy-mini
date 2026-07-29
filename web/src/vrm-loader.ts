// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type { VRM } from "@pixiv/three-vrm";
import type { Mesh } from "three";

const MAX_VRM_FILE_BYTES = 512 * 1024 * 1024;

type ThreeVrmModule = typeof import("@pixiv/three-vrm");

export interface VrmModelInfo {
  readonly name: string;
  readonly version?: string;
  readonly authors: readonly string[];
  readonly metaVersion: "0" | "1";
}

export interface LoadedVrmAvatar {
  readonly vrm: VRM;
  readonly info: VrmModelInfo;
  readonly utils: ThreeVrmModule["VRMUtils"];
}

function modelInfo(vrm: VRM, fallbackName: string): VrmModelInfo {
  const meta = vrm.meta;
  if (meta.metaVersion === "1") {
    return Object.freeze({
      name: meta.name.trim() || fallbackName,
      ...(meta.version?.trim() ? { version: meta.version.trim() } : {}),
      authors: Object.freeze(
        meta.authors.flatMap((author) => {
          const normalized = author.trim();
          return normalized ? [normalized] : [];
        }),
      ),
      metaVersion: "1",
    });
  }
  return Object.freeze({
    name: meta.title?.trim() || fallbackName,
    ...(meta.version?.trim() ? { version: meta.version.trim() } : {}),
    authors: Object.freeze(
      meta.author?.trim() ? [meta.author.trim()] : [],
    ),
    metaVersion: "0",
  });
}

/**
 * Load and optimize a user-selected VRM without adding three-vrm to the
 * application's initial JavaScript chunk.
 */
export async function loadVrmAvatar(file: File): Promise<LoadedVrmAvatar> {
  if (file.size === 0) {
    throw new RangeError("The selected VRM file is empty.");
  }
  if (file.size > MAX_VRM_FILE_BYTES) {
    throw new RangeError("VRM files must be 512 MiB or smaller.");
  }

  const [{ GLTFLoader }, vrmModule] = await Promise.all([
    import("three/examples/jsm/loaders/GLTFLoader.js"),
    import("@pixiv/three-vrm"),
  ]);
  const loader = new GLTFLoader();
  loader.register((parser) => new vrmModule.VRMLoaderPlugin(parser));

  const objectUrl = URL.createObjectURL(file);
  try {
    const gltf = await loader.loadAsync(objectUrl);
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) {
      vrmModule.VRMUtils.deepDispose(gltf.scene);
      throw new TypeError("The selected file does not contain VRM humanoid data.");
    }
    try {
      vrmModule.VRMUtils.removeUnnecessaryVertices(gltf.scene);
      vrmModule.VRMUtils.combineSkeletons(gltf.scene);
      vrmModule.VRMUtils.combineMorphs(vrm);
      vrmModule.VRMUtils.rotateVRM0(vrm);
      vrm.scene.traverse((object) => {
        object.frustumCulled = false;
        const mesh = object as Mesh;
        if (mesh.isMesh) {
          mesh.castShadow = true;
          mesh.receiveShadow = true;
        }
      });
      return Object.freeze({
        vrm,
        info: modelInfo(vrm, file.name.replace(/\.vrm$/iu, "") || "VRM avatar"),
        utils: vrmModule.VRMUtils,
      });
    } catch (error) {
      vrmModule.VRMUtils.deepDispose(vrm.scene);
      throw error;
    }
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

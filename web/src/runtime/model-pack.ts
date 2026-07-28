// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  MODEL_PACK_MANIFEST_FILE,
  type BrowserModelPackManifest,
  normalizePackPath,
  validateModelPackManifest,
} from "./manifest";
import { sha256Hex } from "./hash";

export interface ModelPackProgress {
  stage: "reading-pack" | "hashing-pack";
  completed: number;
  total: number;
  message?: string;
}

export type ModelPackProgressCallback = (progress: ModelPackProgress) => void;

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException("Operation cancelled", "AbortError");
  }
}

function decodeJson(bytes: Uint8Array, path: string): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    throw new Error(
      `Could not parse ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function filePath(file: File): string {
  return normalizePackPath(file.webkitRelativePath || file.name);
}

function locateManifest(paths: readonly string[]): {
  manifestPath: string;
  rootPrefix: string;
} {
  const matches = paths.filter(
    (path) => path === MODEL_PACK_MANIFEST_FILE || path.endsWith(`/${MODEL_PACK_MANIFEST_FILE}`),
  );
  if (matches.length !== 1) {
    throw new Error(
      `Model-pack selection must contain exactly one ${MODEL_PACK_MANIFEST_FILE}`,
    );
  }
  const manifestPath = matches[0];
  return {
    manifestPath,
    rootPrefix: manifestPath.slice(0, -MODEL_PACK_MANIFEST_FILE.length),
  };
}

function stripRoot(path: string, rootPrefix: string): string | undefined {
  if (rootPrefix === "") {
    return path;
  }
  return path.startsWith(rootPrefix) ? path.slice(rootPrefix.length) : undefined;
}

export class ModelPack {
  readonly manifest: BrowserModelPackManifest;
  readonly #files: Map<string, () => Promise<Uint8Array>>;

  constructor(
    manifest: BrowserModelPackManifest,
    files: ReadonlyMap<string, () => Promise<Uint8Array>>,
  ) {
    this.manifest = manifest;
    this.#files = new Map(files);
  }

  has(path: string): boolean {
    return this.#files.has(normalizePackPath(path));
  }

  async read(path: string): Promise<Uint8Array> {
    const canonicalPath = normalizePackPath(path);
    const loader = this.#files.get(canonicalPath);
    if (loader === undefined) {
      throw new Error(`Model pack does not contain ${JSON.stringify(canonicalPath)}`);
    }
    return loader();
  }

  /**
   * Forget an asset after its consumer has initialized. This is particularly
   * important for memory-backed packs, whose ONNX files can be hundreds of MB.
   */
  release(path: string): void {
    this.#files.delete(normalizePackPath(path));
  }
}

async function validateAssets(
  manifest: BrowserModelPackManifest,
  source: ReadonlyMap<string, Uint8Array>,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<ModelPack> {
  const declared = Object.entries(manifest.files);
  const validated = new Map<string, () => Promise<Uint8Array>>();
  let completed = 0;

  for (const [path, description] of declared) {
    throwIfAborted(signal);
    const bytes = source.get(path);
    if (bytes === undefined) {
      throw new Error(`Model pack is missing declared file ${JSON.stringify(path)}`);
    }
    if (bytes.byteLength !== description.size_bytes) {
      throw new Error(
        `Size mismatch for ${path}: expected ${description.size_bytes}, got ${bytes.byteLength}`,
      );
    }
    const actualHash = await sha256Hex(bytes);
    throwIfAborted(signal);
    if (actualHash !== description.sha256) {
      throw new Error(
        `SHA-256 mismatch for ${path}: expected ${description.sha256}, got ${actualHash}`,
      );
    }
    validated.set(path, async () => bytes);
    completed += 1;
    onProgress?.({
      stage: "hashing-pack",
      completed,
      total: declared.length,
      message: path,
    });
  }
  return new ModelPack(manifest, validated);
}

async function validateFileAssets(
  manifest: BrowserModelPackManifest,
  source: ReadonlyMap<string, File>,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<ModelPack> {
  const declared = Object.entries(manifest.files);
  const validated = new Map<string, () => Promise<Uint8Array>>();
  let completed = 0;
  for (const [path, description] of declared) {
    throwIfAborted(signal);
    const file = source.get(path);
    if (file === undefined) {
      throw new Error(`Model pack is missing declared file ${JSON.stringify(path)}`);
    }
    if (file.size !== description.size_bytes) {
      throw new Error(
        `Size mismatch for ${path}: expected ${description.size_bytes}, got ${file.size}`,
      );
    }
    // Hash one file at a time and release its temporary ArrayBuffer before
    // moving on. The immutable File is re-read lazily when its consumer starts.
    let bytes: Uint8Array | undefined = new Uint8Array(await file.arrayBuffer());
    const actualHash = await sha256Hex(bytes);
    bytes = undefined;
    throwIfAborted(signal);
    if (actualHash !== description.sha256) {
      throw new Error(
        `SHA-256 mismatch for ${path}: expected ${description.sha256}, got ${actualHash}`,
      );
    }
    validated.set(
      path,
      async () => new Uint8Array(await file.arrayBuffer()),
    );
    completed += 1;
    onProgress?.({
      stage: "hashing-pack",
      completed,
      total: declared.length,
      message: path,
    });
  }
  return new ModelPack(manifest, validated);
}

/**
 * Load a pack selected using either `webkitdirectory` or the File System Access API.
 *
 * File System Access callers may put the canonical relative path in `File.name`.
 * A single enclosing directory selected with `webkitdirectory` is stripped.
 */
export async function loadModelPackFromFiles(
  files: readonly File[],
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<ModelPack> {
  throwIfAborted(signal);
  if (files.length === 0) {
    throw new Error("No model-pack files were selected");
  }

  const paths = files.map(filePath);
  if (new Set(paths).size !== paths.length) {
    throw new Error("Model-pack selection contains duplicate paths");
  }
  const { manifestPath, rootPrefix } = locateManifest(paths);
  const manifestFile = files[paths.indexOf(manifestPath)];
  const manifestBytes = new Uint8Array(await manifestFile.arrayBuffer());
  throwIfAborted(signal);
  onProgress?.({
    stage: "reading-pack",
    completed: 1,
    total: files.length,
    message: MODEL_PACK_MANIFEST_FILE,
  });
  const manifest = validateModelPackManifest(
    decodeJson(manifestBytes, MODEL_PACK_MANIFEST_FILE),
  );

  const declaredPaths = new Set(Object.keys(manifest.files));
  const source = new Map<string, File>();
  for (let index = 0; index < files.length; index += 1) {
    if (index === paths.indexOf(manifestPath)) {
      continue;
    }
    throwIfAborted(signal);
    const relativePath = stripRoot(paths[index], rootPrefix);
    if (relativePath === undefined || !declaredPaths.has(relativePath)) {
      continue;
    }
    source.set(relativePath, files[index]);
  }
  return validateFileAssets(manifest, source, onProgress, signal);
}

/**
 * Test- and embed-friendly loader. Keys may either be rooted beside manifest.json,
 * or all share one enclosing directory.
 */
export async function loadModelPackFromMemory(
  entries: ReadonlyMap<string, Uint8Array>,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<ModelPack> {
  throwIfAborted(signal);
  const canonical = new Map<string, Uint8Array>();
  for (const [rawPath, bytes] of entries) {
    const path = normalizePackPath(rawPath);
    if (canonical.has(path)) {
      throw new Error(`Duplicate model-pack path ${JSON.stringify(path)}`);
    }
    canonical.set(path, bytes);
  }
  const { manifestPath, rootPrefix } = locateManifest([...canonical.keys()]);
  const manifestBytes = canonical.get(manifestPath);
  if (manifestBytes === undefined) {
    throw new Error(`Missing ${MODEL_PACK_MANIFEST_FILE}`);
  }
  const manifest = validateModelPackManifest(
    decodeJson(manifestBytes, MODEL_PACK_MANIFEST_FILE),
  );
  const source = new Map<string, Uint8Array>();
  for (const path of Object.keys(manifest.files)) {
    const bytes = canonical.get(`${rootPrefix}${path}`);
    if (bytes !== undefined) {
      source.set(path, bytes);
    }
  }
  onProgress?.({
    stage: "reading-pack",
    completed: canonical.size,
    total: canonical.size,
  });
  return validateAssets(manifest, source, onProgress, signal);
}

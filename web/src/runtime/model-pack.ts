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

const TAR_BLOCK_SIZE = 512;
const TAR_END_BLOCKS = 2;
const TAR_NAME_BYTES = 100;
const TAR_PREFIX_OFFSET = 345;
const TAR_PREFIX_BYTES = 155;
const TAR_SIZE_OFFSET = 124;
const TAR_SIZE_BYTES = 12;
const TAR_CHECKSUM_OFFSET = 148;
const TAR_CHECKSUM_BYTES = 8;
const TAR_TYPE_OFFSET = 156;
const TAR_MAGIC_OFFSET = 257;
const TAR_MAGIC = "ustar\0";
const TAR_VERSION_OFFSET = 263;
const TAR_VERSION = "00";
const MAX_TAR_ENTRIES = 10_000;
const MAX_TAR_ENTRY_BYTES = 0x7fff_ffff;
const MAX_TAR_BYTES = 8 * 1024 * 1024 * 1024;
const MAX_TAR_PATH_BYTES = 4_096;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException("Operation cancelled", "AbortError");
  }
}

function decodeTarString(
  field: Uint8Array,
  label: string,
): string {
  const nul = field.indexOf(0);
  const bytes = nul === -1 ? field : field.subarray(0, nul);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(
      `Invalid UTF-8 in tar ${label}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function parseTarOctal(field: Uint8Array, label: string): number {
  if ((field[0] ?? 0) & 0x80) {
    throw new Error(`Base-256 tar ${label} values are not supported`);
  }
  const text = decodeTarString(field, label).trim();
  if (text === "") return 0;
  if (!/^[0-7]+$/.test(text)) {
    throw new Error(`Invalid octal tar ${label}`);
  }
  const value = Number.parseInt(text, 8);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`Tar ${label} exceeds the safe integer range`);
  }
  return value;
}

function isZeroBlock(block: Uint8Array): boolean {
  return block.every((value) => value === 0);
}

function validateTarChecksum(header: Uint8Array): void {
  const expected = parseTarOctal(
    header.subarray(
      TAR_CHECKSUM_OFFSET,
      TAR_CHECKSUM_OFFSET + TAR_CHECKSUM_BYTES,
    ),
    "checksum",
  );
  let actual = 0;
  for (let index = 0; index < header.byteLength; index += 1) {
    actual +=
      index >= TAR_CHECKSUM_OFFSET &&
      index < TAR_CHECKSUM_OFFSET + TAR_CHECKSUM_BYTES
        ? 0x20
        : header[index];
  }
  if (actual !== expected) {
    throw new Error(
      `Tar header checksum mismatch: expected ${expected}, got ${actual}`,
    );
  }
}

function tarEntryPath(header: Uint8Array): string {
  const name = decodeTarString(
    header.subarray(0, TAR_NAME_BYTES),
    "entry name",
  );
  const prefix = decodeTarString(
    header.subarray(
      TAR_PREFIX_OFFSET,
      TAR_PREFIX_OFFSET + TAR_PREFIX_BYTES,
    ),
    "entry prefix",
  );
  const rawPath = prefix ? `${prefix}/${name}` : name;
  if (
    rawPath.length === 0 ||
    new TextEncoder().encode(rawPath).byteLength > MAX_TAR_PATH_BYTES ||
    rawPath.includes("\\") ||
    /[\u0000-\u001f\u007f]/.test(rawPath) ||
    /^[A-Za-z]:/.test(rawPath)
  ) {
    throw new Error(`Unsafe tar entry path ${JSON.stringify(rawPath)}`);
  }
  try {
    const canonicalPath = normalizePackPath(rawPath);
    if (canonicalPath !== rawPath) {
      throw new Error("Tar entry paths must already be canonical");
    }
    return canonicalPath;
  } catch (error) {
    throw new Error(
      `Unsafe tar entry path ${JSON.stringify(rawPath)}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

class TarStreamReader {
  readonly #reader: ReadableStreamDefaultReader<Uint8Array>;
  readonly #signal?: AbortSignal;
  #chunk: Uint8Array<ArrayBufferLike> = new Uint8Array();
  #offset = 0;
  #streamBytes = 0;
  #ended = false;

  constructor(stream: ReadableStream<Uint8Array>, signal?: AbortSignal) {
    this.#reader = stream.getReader();
    this.#signal = signal;
  }

  async #nextChunk(): Promise<boolean> {
    throwIfAborted(this.#signal);
    if (this.#ended) return false;
    let result: ReadableStreamReadResult<Uint8Array>;
    try {
      result = await this.#reader.read();
    } catch (error) {
      throw new Error(
        `Could not decompress the model-pack archive: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    throwIfAborted(this.#signal);
    if (result.done) {
      this.#ended = true;
      this.#chunk = new Uint8Array();
      this.#offset = 0;
      return false;
    }
    if (result.value.byteLength === 0) return this.#nextChunk();
    this.#streamBytes += result.value.byteLength;
    if (this.#streamBytes > MAX_TAR_BYTES) {
      throw new Error(
        `Decompressed model pack exceeds ${MAX_TAR_BYTES} bytes`,
      );
    }
    this.#chunk = result.value;
    this.#offset = 0;
    return true;
  }

  async readExactly(
    length: number,
    options: { allowEof?: boolean } = {},
  ): Promise<Uint8Array | null> {
    if (!Number.isSafeInteger(length) || length < 0) {
      throw new RangeError("Tar read length must be a non-negative safe integer");
    }
    const output = new Uint8Array(length);
    let written = 0;
    while (written < length) {
      if (
        this.#offset >= this.#chunk.byteLength &&
        !(await this.#nextChunk())
      ) {
        if (written === 0 && options.allowEof) return null;
        throw new Error(
          `Truncated tar archive: expected ${length} bytes, received ${written}`,
        );
      }
      const available = this.#chunk.byteLength - this.#offset;
      const count = Math.min(available, length - written);
      output.set(
        this.#chunk.subarray(this.#offset, this.#offset + count),
        written,
      );
      this.#offset += count;
      written += count;
    }
    return output;
  }

  async assertZeroUntilEof(): Promise<void> {
    while (true) {
      if (this.#offset >= this.#chunk.byteLength) {
        if (!(await this.#nextChunk())) return;
      }
      for (
        let index = this.#offset;
        index < this.#chunk.byteLength;
        index += 1
      ) {
        if (this.#chunk[index] !== 0) {
          throw new Error("Tar archive contains data after its end marker");
        }
      }
      this.#offset = this.#chunk.byteLength;
    }
  }

  async cancel(reason?: unknown): Promise<void> {
    try {
      await this.#reader.cancel(reason);
    } catch {
      // Preserve the validation/decompression error that caused cancellation.
    }
  }
}

function monitoredGzipStream(
  archive: File,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): ReadableStream<Uint8Array> {
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser does not support gzip decompression");
  }
  let completed = 0;
  const monitor = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      throwIfAborted(signal);
      completed += chunk.byteLength;
      controller.enqueue(chunk);
      onProgress?.({
        stage: "reading-pack",
        completed,
        total: archive.size,
        message: archive.name,
      });
    },
  });
  const decompressor = new DecompressionStream("gzip");
  return archive.stream().pipeThrough(monitor).pipeThrough({
    readable: decompressor.readable as ReadableStream<Uint8Array>,
    writable: decompressor.writable as WritableStream<Uint8Array>,
  });
}

async function readTarEntries(
  archive: File,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<{
  manifest: BrowserModelPackManifest;
  entries: Map<string, Uint8Array>;
}> {
  const reader = new TarStreamReader(
    monitoredGzipStream(archive, onProgress, signal),
    signal,
  );
  const entries = new Map<string, Uint8Array>();
  const seenPaths = new Set<string>();
  let manifest: BrowserModelPackManifest | null = null;
  let zeroBlocks = 0;
  try {
    while (zeroBlocks < TAR_END_BLOCKS) {
      const header = await reader.readExactly(TAR_BLOCK_SIZE, {
        allowEof: zeroBlocks === 0,
      });
      if (header === null) {
        throw new Error("Tar archive is empty and has no end marker");
      }
      if (isZeroBlock(header)) {
        zeroBlocks += 1;
        continue;
      }
      if (zeroBlocks !== 0) {
        throw new Error("Tar archive has only one zero block before another entry");
      }
      validateTarChecksum(header);
      const magic = decodeTarString(
        header.subarray(TAR_MAGIC_OFFSET, TAR_MAGIC_OFFSET + TAR_MAGIC.length),
        "magic",
      );
      const version = decodeTarString(
        header.subarray(TAR_VERSION_OFFSET, TAR_VERSION_OFFSET + TAR_VERSION.length),
        "version",
      );
      if (`${magic}\0` !== TAR_MAGIC || version !== TAR_VERSION) {
        throw new Error("Model-pack tar must use the POSIX ustar format");
      }
      const type = header[TAR_TYPE_OFFSET];
      if (type !== 0 && type !== 0x30) {
        throw new Error(
          `Model-pack tar entries must be regular files (type ${String.fromCharCode(type) || "\\0"})`,
        );
      }
      const path = tarEntryPath(header);
      if (seenPaths.has(path)) {
        throw new Error(`Duplicate tar entry path ${JSON.stringify(path)}`);
      }
      if (seenPaths.size >= MAX_TAR_ENTRIES) {
        throw new Error(
          `Model-pack tar contains more than ${MAX_TAR_ENTRIES} entries`,
        );
      }
      if (seenPaths.size === 0 && path !== MODEL_PACK_MANIFEST_FILE) {
        throw new Error(
          `The first tar entry must be ${MODEL_PACK_MANIFEST_FILE}`,
        );
      }
      const size = parseTarOctal(
        header.subarray(TAR_SIZE_OFFSET, TAR_SIZE_OFFSET + TAR_SIZE_BYTES),
        "entry size",
      );
      if (size > MAX_TAR_ENTRY_BYTES) {
        throw new Error(
          `Tar entry ${JSON.stringify(path)} exceeds ${MAX_TAR_ENTRY_BYTES} bytes`,
        );
      }
      if (path === MODEL_PACK_MANIFEST_FILE && size > MAX_MANIFEST_BYTES) {
        throw new Error(
          `${MODEL_PACK_MANIFEST_FILE} exceeds ${MAX_MANIFEST_BYTES} bytes`,
        );
      }
      if (manifest !== null) {
        if (!Object.hasOwn(manifest.files, path)) {
          throw new Error(
            `Model-pack archive contains undeclared file ${JSON.stringify(path)}`,
          );
        }
        const description = manifest.files[path];
        if (size !== description.size_bytes) {
          throw new Error(
            `Size mismatch for ${path}: expected ${description.size_bytes}, got ${size}`,
          );
        }
      }
      const bytes = await reader.readExactly(size);
      if (bytes === null) throw new Error("Unreachable tar entry read");
      seenPaths.add(path);
      if (path === MODEL_PACK_MANIFEST_FILE) {
        manifest = validateModelPackManifest(
          decodeJson(bytes, MODEL_PACK_MANIFEST_FILE),
        );
        if (
          Object.hasOwn(manifest.files, MODEL_PACK_MANIFEST_FILE)
        ) {
          throw new Error(
            `${MODEL_PACK_MANIFEST_FILE} must not declare itself as an asset`,
          );
        }
        if (Object.keys(manifest.files).length + 1 > MAX_TAR_ENTRIES) {
          throw new Error(
            `Model pack declares more than ${MAX_TAR_ENTRIES - 1} assets`,
          );
        }
        let declaredBytes = 0;
        for (const description of Object.values(manifest.files)) {
          declaredBytes += description.size_bytes;
          if (
            !Number.isSafeInteger(declaredBytes) ||
            declaredBytes > MAX_TAR_BYTES
          ) {
            throw new Error(
              `Model pack declares more than ${MAX_TAR_BYTES} bytes`,
            );
          }
        }
      } else {
        entries.set(path, bytes);
      }
      const padding = (TAR_BLOCK_SIZE - (size % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE;
      if (padding > 0) {
        const paddingBytes = await reader.readExactly(padding);
        if (paddingBytes === null || !isZeroBlock(paddingBytes)) {
          throw new Error(`Tar entry ${JSON.stringify(path)} has non-zero padding`);
        }
      }
    }
    await reader.assertZeroUntilEof();
    if (manifest === null) {
      throw new Error(
        `Model-pack archive must contain ${MODEL_PACK_MANIFEST_FILE} at its root`,
      );
    }
    for (const path of Object.keys(manifest.files)) {
      if (!entries.has(path)) {
        throw new Error(
          `Model pack is missing declared file ${JSON.stringify(path)}`,
        );
      }
    }
    return { manifest, entries };
  } catch (error) {
    await reader.cancel(error);
    throw error;
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

/**
 * Load the browser model pack from its sole distribution format: a POSIX
 * ustar archive compressed with gzip.
 */
export async function loadModelPackFromTarGzip(
  archive: File,
  onProgress?: ModelPackProgressCallback,
  signal?: AbortSignal,
): Promise<ModelPack> {
  throwIfAborted(signal);
  if (!archive.name.toLowerCase().endsWith(".tar.gz")) {
    throw new Error("Model pack must be a .tar.gz file");
  }
  if (archive.size === 0) {
    throw new Error("The selected model-pack archive is empty");
  }
  const { manifest, entries } = await readTarEntries(
    archive,
    onProgress,
    signal,
  );
  return validateAssets(manifest, entries, onProgress, signal);
}

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { sha256Hex } from "./hash";
import {
  MODEL_MANIFEST_FILE,
  type BrowserModelManifest,
  type ManifestFile,
  normalizeModelPath,
  validateModelManifest,
} from "./manifest";

export const MODEL_CACHE_PREFIX = "ardy-mini-model-files-v1-";

const MODEL_CACHE_MARKER_QUERY = "ardy-model-cache-complete";
const MODEL_CACHE_MARKER_VERSION = 1;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_CACHE_MARKER_BYTES = 64 * 1024;
const MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
const MAX_MODEL_FILES = 10_000;
const SHA256_RE = /^[0-9a-f]{64}$/;

export interface ModelAssetProgress {
  stage: "downloading-model" | "verifying-model";
  completed: number;
  total: number;
  message?: string;
}

export type ModelAssetProgressCallback = (
  progress: ModelAssetProgress,
) => void;

export interface FetchModelManifestOptions {
  signal?: AbortSignal;
}

export interface ModelManifestSource {
  baseUrl: string;
  manifestUrl: string;
  manifest: BrowserModelManifest;
  manifestSha256: string;
  cacheName: string;
  /** Decompressed JSON bytes used for manifest identity. */
  manifestBytes: Uint8Array;
  /** Exact gzip response retained in Cache Storage for offline startup. */
  manifestTransportBytes: Uint8Array;
  manifestTransportSha256: string;
}

export interface ModelCacheStatus {
  supported: boolean;
  cached: boolean;
  complete: boolean;
  cacheName: string;
  modelId: string;
  revision: string;
  manifestSha256: string;
  fileCount: number;
  cachedFileCount: number;
  transportSizeBytes: number;
  cachedTransportSizeBytes: number;
  rawSizeBytes: number;
}

interface CacheCompletionMarker {
  schema_version: typeof MODEL_CACHE_MARKER_VERSION;
  base_url: string;
  manifest_sha256: string;
  manifest_transport_sha256: string;
  manifest_transport_size_bytes: number;
  model_id: string;
  revision: string;
  file_count: number;
  transport_size_bytes: number;
  raw_size_bytes: number;
  completed_at: string;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException("Operation cancelled", "AbortError");
  }
}

function locationHref(): string | undefined {
  return typeof globalThis.location === "undefined"
    ? undefined
    : globalThis.location.href;
}

export function normalizeModelBaseUrl(value: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError("Model base URL must be a non-empty string");
  }
  let url: URL;
  try {
    url = new URL(value, locationHref());
  } catch (error) {
    throw new TypeError("Model base URL must be a valid HTTP(S) URL", {
      cause: error,
    });
  }
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username !== "" ||
    url.password !== "" ||
    url.search !== "" ||
    url.hash !== ""
  ) {
    throw new TypeError(
      "Model base URL must be an HTTP(S) URL without credentials, query, or fragment",
    );
  }
  if (!url.pathname.endsWith("/")) {
    url.pathname += "/";
  }
  return url.href;
}

function manifestUrl(baseUrl: string): string {
  return new URL(MODEL_MANIFEST_FILE, baseUrl).href;
}

function cacheMarkerUrl(baseUrl: string): string {
  const url = new URL(MODEL_MANIFEST_FILE, baseUrl);
  url.searchParams.set(MODEL_CACHE_MARKER_QUERY, "1");
  return url.href;
}

function cacheStorage(): CacheStorage | null {
  return typeof globalThis.caches === "undefined"
    ? null
    : globalThis.caches;
}

function exactArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  if (
    bytes.byteOffset === 0 &&
    bytes.byteLength === bytes.buffer.byteLength &&
    bytes.buffer instanceof ArrayBuffer
  ) {
    return bytes.buffer;
  }
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function finiteSafeTotal(
  values: Iterable<number>,
  label: string,
): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total) || total > MAX_MODEL_BYTES) {
      throw new RangeError(`${label} exceeds ${MAX_MODEL_BYTES} bytes`);
    }
  }
  return total;
}

export function modelTransportSize(manifest: BrowserModelManifest): number {
  return finiteSafeTotal(
    Object.values(manifest.files).map(
      (description) => description.transport.size_bytes,
    ),
    "Compressed model",
  );
}

export function modelRawSize(manifest: BrowserModelManifest): number {
  return finiteSafeTotal(
    Object.values(manifest.files).map(
      (description) => description.size_bytes,
    ),
    "Decompressed model",
  );
}

export function formatModelBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

function decodeJson(bytes: Uint8Array, path: string): unknown {
  try {
    return JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
  } catch (error) {
    throw new Error(
      `Could not parse ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

async function readKnownLength(
  stream: ReadableStream<Uint8Array>,
  expectedBytes: number,
  label: string,
  options: {
    signal?: AbortSignal;
    onChunk?: (completed: number) => void;
  } = {},
): Promise<Uint8Array> {
  if (
    !Number.isSafeInteger(expectedBytes) ||
    expectedBytes < 0 ||
    expectedBytes > MAX_MODEL_BYTES
  ) {
    throw new RangeError(`${label} declares an invalid byte length`);
  }
  const output = new Uint8Array(expectedBytes);
  const reader = stream.getReader();
  let completed = 0;
  try {
    while (true) {
      throwIfAborted(options.signal);
      const result = await reader.read();
      throwIfAborted(options.signal);
      if (result.done) break;
      if (completed + result.value.byteLength > expectedBytes) {
        throw new Error(
          `${label} exceeds its declared size of ${expectedBytes} bytes`,
        );
      }
      output.set(result.value, completed);
      completed += result.value.byteLength;
      options.onChunk?.(completed);
    }
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    throw error;
  }
  if (completed !== expectedBytes) {
    throw new Error(
      `${label} size mismatch: expected ${expectedBytes}, got ${completed}`,
    );
  }
  return output;
}

async function responseBytes(
  response: Response,
  expectedBytes: number,
  label: string,
  options: {
    signal?: AbortSignal;
    onChunk?: (completed: number) => void;
  } = {},
): Promise<Uint8Array> {
  const declaredLength = response.headers.get("content-length");
  if (
    declaredLength !== null &&
    (!/^\d+$/.test(declaredLength) ||
      Number(declaredLength) !== expectedBytes)
  ) {
    throw new Error(
      `${label} Content-Length does not match the expected ${expectedBytes} bytes`,
    );
  }
  if (response.body === null) {
    if (expectedBytes !== 0) {
      throw new Error(`${label} has no response body`);
    }
    return new Uint8Array();
  }
  return readKnownLength(response.body, expectedBytes, label, options);
}

async function readBounded(
  stream: ReadableStream<Uint8Array>,
  maximumBytes: number,
  label: string,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  try {
    while (true) {
      throwIfAborted(signal);
      const result = await reader.read();
      throwIfAborted(signal);
      if (result.done) break;
      byteLength += result.value.byteLength;
      if (byteLength > maximumBytes) {
        throw new RangeError(`${label} exceeds ${maximumBytes} bytes`);
      }
      chunks.push(result.value);
    }
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    throw error;
  }
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function gzipDecompressionStream(
  bytes: Uint8Array,
  label: string,
): ReadableStream<Uint8Array> {
  if (typeof DecompressionStream !== "function") {
    throw new Error("This browser does not support gzip decompression");
  }
  try {
    const decompressor = new DecompressionStream("gzip");
    const input = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes);
        controller.close();
      },
    });
    return input.pipeThrough(
      decompressor as unknown as TransformStream<
        Uint8Array,
        Uint8Array
      >,
    );
  } catch (error) {
    throw new Error(`Could not initialize gzip decompression for ${label}`, {
      cause: error,
    });
  }
}

async function decompressManifest(
  transport: Uint8Array,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  try {
    return await readBounded(
      gzipDecompressionStream(transport, MODEL_MANIFEST_FILE),
      MAX_MANIFEST_BYTES,
      `Decompressed ${MODEL_MANIFEST_FILE}`,
      signal,
    );
  } catch (error) {
    throw new Error(
      `Could not decompress ${MODEL_MANIFEST_FILE}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
}

async function fetchResponse(
  url: string,
  signal?: AbortSignal,
): Promise<Response> {
  throwIfAborted(signal);
  const response = await fetch(url, {
    cache: "no-cache",
    credentials: "omit",
    mode: "cors",
    redirect: "follow",
    signal,
  });
  throwIfAborted(signal);
  if (!response.ok) {
    throw new Error(
      `Could not download ${url}: HTTP ${response.status} ${response.statusText}`.trim(),
    );
  }
  return response;
}

function markerFor(source: ModelManifestSource): CacheCompletionMarker {
  return {
    schema_version: MODEL_CACHE_MARKER_VERSION,
    base_url: source.baseUrl,
    manifest_sha256: source.manifestSha256,
    manifest_transport_sha256: source.manifestTransportSha256,
    manifest_transport_size_bytes:
      source.manifestTransportBytes.byteLength,
    model_id: source.manifest.model.id,
    revision: source.manifest.model.revision,
    file_count: Object.keys(source.manifest.files).length,
    transport_size_bytes: modelTransportSize(source.manifest),
    raw_size_bytes: modelRawSize(source.manifest),
    completed_at: new Date().toISOString(),
  };
}

function validateMarker(
  value: unknown,
  source: ModelManifestSource,
): value is CacheCompletionMarker {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const marker = value as Partial<CacheCompletionMarker>;
  return (
    marker.schema_version === MODEL_CACHE_MARKER_VERSION &&
    marker.base_url === source.baseUrl &&
    marker.manifest_sha256 === source.manifestSha256 &&
    marker.manifest_transport_sha256 ===
      source.manifestTransportSha256 &&
    marker.manifest_transport_size_bytes ===
      source.manifestTransportBytes.byteLength &&
    marker.model_id === source.manifest.model.id &&
    marker.revision === source.manifest.model.revision &&
    marker.file_count === Object.keys(source.manifest.files).length &&
    marker.transport_size_bytes === modelTransportSize(source.manifest) &&
    marker.raw_size_bytes === modelRawSize(source.manifest) &&
    typeof marker.completed_at === "string" &&
    marker.completed_at.length > 0
  );
}

function markerManifestTransport(
  value: unknown,
): { sha256: string; sizeBytes: number } | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const marker = value as Partial<CacheCompletionMarker>;
  if (
    marker.schema_version !== MODEL_CACHE_MARKER_VERSION ||
    typeof marker.manifest_transport_sha256 !== "string" ||
    !SHA256_RE.test(marker.manifest_transport_sha256) ||
    typeof marker.manifest_transport_size_bytes !== "number" ||
    !Number.isSafeInteger(marker.manifest_transport_size_bytes) ||
    marker.manifest_transport_size_bytes <= 0 ||
    marker.manifest_transport_size_bytes > MAX_MANIFEST_BYTES
  ) {
    return null;
  }
  return {
    sha256: marker.manifest_transport_sha256,
    sizeBytes: marker.manifest_transport_size_bytes,
  };
}

async function cacheNameFor(
  baseUrl: string,
  manifest: BrowserModelManifest,
  manifestSha256: string,
): Promise<string> {
  const identity = new TextEncoder().encode(
    `${baseUrl}\0${manifest.model.id}\0${manifest.model.revision}`,
  );
  const identityHash = await sha256Hex(identity);
  return (
    `${MODEL_CACHE_PREFIX}${identityHash.slice(0, 20)}-` +
    manifestSha256
  );
}

async function sourceFromBytes(
  baseUrl: string,
  rawBytes: Uint8Array,
  transportBytes: Uint8Array,
): Promise<ModelManifestSource> {
  const manifest = validateModelManifest(
    decodeJson(rawBytes, MODEL_MANIFEST_FILE),
  );
  const fileCount = Object.keys(manifest.files).length;
  if (fileCount === 0 || fileCount > MAX_MODEL_FILES) {
    throw new RangeError(
      `${MODEL_MANIFEST_FILE} must declare 1–${MAX_MODEL_FILES} files`,
    );
  }
  modelTransportSize(manifest);
  modelRawSize(manifest);
  const manifestSha256 = await sha256Hex(rawBytes);
  return {
    baseUrl,
    manifestUrl: manifestUrl(baseUrl),
    manifest,
    manifestSha256,
    cacheName: await cacheNameFor(baseUrl, manifest, manifestSha256),
    manifestBytes: rawBytes,
    manifestTransportBytes: transportBytes,
    manifestTransportSha256: await sha256Hex(transportBytes),
  };
}

async function cachedSourceForBaseUrl(
  baseUrl: string,
): Promise<ModelManifestSource | null> {
  const storage = cacheStorage();
  if (storage === null) return null;
  const names = (await storage.keys())
    .filter((name) => name.startsWith(MODEL_CACHE_PREFIX))
    .sort()
    .reverse();
  for (const name of names) {
    const cache = await storage.open(name);
    const markerResponse = await cache.match(cacheMarkerUrl(baseUrl));
    const manifestResponse = await cache.match(manifestUrl(baseUrl));
    if (markerResponse === undefined || manifestResponse === undefined) {
      continue;
    }
    try {
      if (markerResponse.body === null) continue;
      const markerBytes = await readBounded(
        markerResponse.body,
        MAX_CACHE_MARKER_BYTES,
        "model cache completion marker",
      );
      const marker = decodeJson(
        markerBytes,
        "model cache completion marker",
      );
      const transportIdentity = markerManifestTransport(marker);
      if (transportIdentity === null) continue;
      const transportBytes = await responseBytes(
        manifestResponse,
        transportIdentity.sizeBytes,
        MODEL_MANIFEST_FILE,
      );
      if (
        (await sha256Hex(transportBytes)) !== transportIdentity.sha256
      ) {
        continue;
      }
      const rawBytes = await decompressManifest(transportBytes);
      const source = await sourceFromBytes(
        baseUrl,
        rawBytes,
        transportBytes,
      );
      if (source.cacheName !== name || !validateMarker(marker, source)) {
        continue;
      }
      return source;
    } catch {
      // Ignore malformed/incomplete caches and continue to another revision.
    }
  }
  return null;
}

export async function fetchModelManifest(
  value: string,
  options: FetchModelManifestOptions = {},
): Promise<ModelManifestSource> {
  const baseUrl = normalizeModelBaseUrl(value);
  const url = manifestUrl(baseUrl);
  try {
    const response = await fetchResponse(url, options.signal);
    const declaredLength = response.headers.get("content-length");
    const size =
      declaredLength === null ? -1 : Number.parseInt(declaredLength, 10);
    let transportBytes: Uint8Array;
    if (declaredLength === null) {
      if (response.body === null) {
        throw new Error(`${MODEL_MANIFEST_FILE} has no response body`);
      }
      transportBytes = await readBounded(
        response.body,
        MAX_MANIFEST_BYTES,
        MODEL_MANIFEST_FILE,
        options.signal,
      );
    } else {
      if (
        size <= 0 ||
        !Number.isSafeInteger(size) ||
        size > MAX_MANIFEST_BYTES
      ) {
        throw new RangeError(
          `${MODEL_MANIFEST_FILE} exceeds ${MAX_MANIFEST_BYTES} bytes`,
        );
      }
      transportBytes = await responseBytes(
        response,
        size,
        MODEL_MANIFEST_FILE,
        { signal: options.signal },
      );
    }
    const rawBytes = await decompressManifest(
      transportBytes,
      options.signal,
    );
    return sourceFromBytes(baseUrl, rawBytes, transportBytes);
  } catch (networkError) {
    throwIfAborted(options.signal);
    const cached = await cachedSourceForBaseUrl(baseUrl);
    if (cached !== null) return cached;
    throw networkError;
  }
}

function emptyCacheStatus(
  source: ModelManifestSource,
  supported: boolean,
): ModelCacheStatus {
  return {
    supported,
    cached: false,
    complete: false,
    cacheName: source.cacheName,
    modelId: source.manifest.model.id,
    revision: source.manifest.model.revision,
    manifestSha256: source.manifestSha256,
    fileCount: Object.keys(source.manifest.files).length,
    cachedFileCount: 0,
    transportSizeBytes: modelTransportSize(source.manifest),
    cachedTransportSizeBytes: 0,
    rawSizeBytes: modelRawSize(source.manifest),
  };
}

async function cachedInventory(
  cache: Cache,
  source: ModelManifestSource,
): Promise<{
  cachedFileCount: number;
  cachedTransportSizeBytes: number;
}> {
  let cachedFileCount = 0;
  let cachedTransportSizeBytes = 0;
  const descriptions = Object.values(source.manifest.files);
  const responses = await Promise.all(
    descriptions.map((description) =>
      cache.match(
        new URL(description.transport.path, source.baseUrl).href,
      ),
    ),
  );
  for (let index = 0; index < descriptions.length; index += 1) {
    const description = descriptions[index];
    const response = responses[index];
    if (response === undefined) continue;
    cachedFileCount += 1;
    cachedTransportSizeBytes += description.transport.size_bytes;
  }
  return { cachedFileCount, cachedTransportSizeBytes };
}

export async function inspectModelCache(
  source: ModelManifestSource,
): Promise<ModelCacheStatus> {
  const storage = cacheStorage();
  if (storage === null) return emptyCacheStatus(source, false);
  if (!(await storage.has(source.cacheName))) {
    return emptyCacheStatus(source, true);
  }
  const cache = await storage.open(source.cacheName);
  const inventory = await cachedInventory(cache, source);
  const [markerResponse, manifestResponse] = await Promise.all([
    cache.match(cacheMarkerUrl(source.baseUrl)),
    cache.match(source.manifestUrl),
  ]);
  if (markerResponse === undefined || manifestResponse === undefined) {
    return {
      ...emptyCacheStatus(source, true),
      cached: true,
      ...inventory,
    };
  }
  try {
    if (markerResponse.body === null) {
      throw new Error("Model cache completion marker has no body");
    }
    const marker = decodeJson(
      await readBounded(
        markerResponse.body,
        MAX_CACHE_MARKER_BYTES,
        "model cache completion marker",
      ),
      "model cache completion marker",
    );
    if (!validateMarker(marker, source)) {
      return {
        ...emptyCacheStatus(source, true),
        cached: true,
        ...inventory,
      };
    }
    const cachedManifestTransport = await responseBytes(
      manifestResponse,
      source.manifestTransportBytes.byteLength,
      MODEL_MANIFEST_FILE,
    );
    if (
      (await sha256Hex(cachedManifestTransport)) !==
      source.manifestTransportSha256
    ) {
      return {
        ...emptyCacheStatus(source, true),
        cached: true,
        ...inventory,
      };
    }
    const cachedManifest = await decompressManifest(
      cachedManifestTransport,
    );
    if (
      cachedManifest.byteLength !== source.manifestBytes.byteLength ||
      (await sha256Hex(cachedManifest)) !== source.manifestSha256
    ) {
      return {
        ...emptyCacheStatus(source, true),
        cached: true,
        ...inventory,
      };
    }
    for (const description of Object.values(source.manifest.files)) {
      const response = await cache.match(
        new URL(description.transport.path, source.baseUrl).href,
      );
      if (response === undefined) {
        return {
          ...emptyCacheStatus(source, true),
          cached: true,
          ...inventory,
        };
      }
    }
    return {
      ...emptyCacheStatus(source, true),
      cached: true,
      complete: true,
      ...inventory,
    };
  } catch {
    return {
      ...emptyCacheStatus(source, true),
      cached: true,
      ...inventory,
    };
  }
}

export async function clearModelCache(
  source?: ModelManifestSource,
): Promise<void> {
  const storage = cacheStorage();
  if (storage === null) return;
  if (source !== undefined) {
    await storage.delete(source.cacheName);
    return;
  }
  const deletions: Array<Promise<boolean>> = [];
  for (const name of await storage.keys()) {
    if (name.startsWith(MODEL_CACHE_PREFIX)) {
      deletions.push(storage.delete(name));
    }
  }
  await Promise.all(deletions);
}

function cachedResponse(
  bytes: Uint8Array,
  mediaType: string,
): Response {
  return new Response(exactArrayBuffer(bytes), {
    headers: {
      "content-length": String(bytes.byteLength),
      "content-type": mediaType,
    },
  });
}

async function verifyTransport(
  bytes: Uint8Array,
  description: ManifestFile,
  path: string,
): Promise<void> {
  if (bytes.byteLength !== description.transport.size_bytes) {
    throw new Error(
      `Compressed size mismatch for ${path}: expected ${description.transport.size_bytes}, got ${bytes.byteLength}`,
    );
  }
  const actualHash = await sha256Hex(bytes);
  if (actualHash !== description.transport.sha256) {
    throw new Error(
      `Compressed SHA-256 mismatch for ${path}: expected ${description.transport.sha256}, got ${actualHash}`,
    );
  }
}

async function decompressAndVerify(
  bytes: Uint8Array,
  description: ManifestFile,
  path: string,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  throwIfAborted(signal);
  let raw: Uint8Array;
  try {
    raw = await readKnownLength(
      gzipDecompressionStream(bytes, path),
      description.size_bytes,
      path,
      { signal },
    );
  } catch (error) {
    throw new Error(
      `Could not decompress ${path}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
  const actualHash = await sha256Hex(raw);
  throwIfAborted(signal);
  if (actualHash !== description.sha256) {
    throw new Error(
      `SHA-256 mismatch for ${path}: expected ${description.sha256}, got ${actualHash}`,
    );
  }
  return raw;
}

async function readTransportResponse(
  response: Response,
  description: ManifestFile,
  path: string,
  signal?: AbortSignal,
  onChunk?: (completed: number) => void,
): Promise<Uint8Array> {
  const bytes = await responseBytes(
    response,
    description.transport.size_bytes,
    `${path} gzip transport`,
    { signal, onChunk },
  );
  await verifyTransport(bytes, description, path);
  throwIfAborted(signal);
  return bytes;
}

async function removeOlderSourceCaches(
  source: ModelManifestSource,
): Promise<void> {
  const storage = cacheStorage();
  if (storage === null) return;
  const removals: Array<Promise<boolean>> = [];
  for (const name of await storage.keys()) {
    if (
      name === source.cacheName ||
      !name.startsWith(MODEL_CACHE_PREFIX)
    ) {
      continue;
    }
    removals.push(storage.delete(name));
  }
  await Promise.all(removals);
}

export class ModelAssets {
  readonly manifest: BrowserModelManifest;
  readonly source: ModelManifestSource;
  readonly #cache: Cache | null;
  readonly #memoryTransports: Map<string, Uint8Array>;
  readonly #loaded = new Map<string, Uint8Array>();
  readonly #inFlight = new Map<string, Promise<Uint8Array>>();
  readonly #verifiedTransports: Set<string>;
  readonly #signal?: AbortSignal;
  readonly #onProgress?: ModelAssetProgressCallback;
  readonly #verifiedRawFiles = new Set<string>();

  constructor(
    source: ModelManifestSource,
    options: {
      cache?: Cache | null;
      memoryTransports?: ReadonlyMap<string, Uint8Array>;
      verifiedTransports?: ReadonlySet<string>;
      signal?: AbortSignal;
      onProgress?: ModelAssetProgressCallback;
    } = {},
  ) {
    this.source = source;
    this.manifest = source.manifest;
    this.#cache = options.cache ?? null;
    this.#memoryTransports = new Map(options.memoryTransports);
    this.#verifiedTransports = new Set(options.verifiedTransports);
    this.#signal = options.signal;
    this.#onProgress = options.onProgress;
  }

  has(path: string): boolean {
    return Object.hasOwn(this.manifest.files, normalizeModelPath(path));
  }

  async #load(path: string): Promise<Uint8Array> {
    throwIfAborted(this.#signal);
    const description = this.manifest.files[path];
    if (description === undefined) {
      throw new Error(`Model does not contain ${JSON.stringify(path)}`);
    }
    const transportUrl = new URL(
      description.transport.path,
      this.source.baseUrl,
    ).href;
    let transport = this.#memoryTransports.get(path);
    let transportVerified =
      transport !== undefined && this.#verifiedTransports.has(path);
    if (transport === undefined && this.#cache !== null) {
      const response = await this.#cache.match(transportUrl);
      if (response === undefined) {
        throw new Error(`Cached model file is missing ${JSON.stringify(path)}`);
      }
      if (this.#verifiedTransports.has(path)) {
        transport = await responseBytes(
          response,
          description.transport.size_bytes,
          `${path} gzip transport`,
          { signal: this.#signal },
        );
        transportVerified = true;
      } else {
        transport = await readTransportResponse(
          response,
          description,
          path,
          this.#signal,
        );
        transportVerified = true;
      }
    }
    if (transport === undefined) {
      throw new Error(`Model file is unavailable ${JSON.stringify(path)}`);
    }
    if (!transportVerified) {
      await verifyTransport(transport, description, path);
    }
    const raw = await decompressAndVerify(
      transport,
      description,
      path,
      this.#signal,
    );
    this.#verifiedRawFiles.add(path);
    this.#onProgress?.({
      stage: "verifying-model",
      completed: this.#verifiedRawFiles.size,
      total: Object.keys(this.manifest.files).length,
      message: path,
    });
    this.#loaded.set(path, raw);
    return raw;
  }

  async read(path: string): Promise<Uint8Array> {
    const canonicalPath = normalizeModelPath(path);
    const loaded = this.#loaded.get(canonicalPath);
    if (loaded !== undefined) return loaded;
    const inFlight = this.#inFlight.get(canonicalPath);
    if (inFlight !== undefined) return inFlight;
    const promise = this.#load(canonicalPath).finally(() => {
      this.#inFlight.delete(canonicalPath);
    });
    this.#inFlight.set(canonicalPath, promise);
    return promise;
  }

  /**
   * Drop decompressed and memory-only transport bytes once their consumer has
   * initialized. Persistent Cache Storage data remains available for reloads.
   */
  release(path: string): void {
    const canonicalPath = normalizeModelPath(path);
    this.#loaded.delete(canonicalPath);
    this.#memoryTransports.delete(canonicalPath);
  }

  hasVerifiedAllFiles(): boolean {
    return (
      this.#verifiedRawFiles.size ===
      Object.keys(this.manifest.files).length
    );
  }
}

export async function loadModelAssets(
  baseUrl: string,
  onProgress?: ModelAssetProgressCallback,
  signal?: AbortSignal,
): Promise<ModelAssets> {
  const source = await fetchModelManifest(baseUrl, { signal });
  throwIfAborted(signal);
  const status = await inspectModelCache(source);
  const storage = cacheStorage();
  if (status.complete && storage !== null) {
    onProgress?.({
      stage: "downloading-model",
      completed: status.transportSizeBytes,
      total: status.transportSizeBytes,
      message: "Cached model",
    });
    return new ModelAssets(source, {
      cache: await storage.open(source.cacheName),
      signal,
      onProgress,
    });
  }

  const entries = Object.entries(source.manifest.files);
  const totalBytes = modelTransportSize(source.manifest);
  const memoryTransports = new Map<string, Uint8Array>();
  let cache: Cache | null = null;
  let completedBytes = 0;
  const verifiedTransports = new Set<string>();

  if (storage !== null) {
    cache = await storage.open(source.cacheName);
    await cache.put(
      source.manifestUrl,
      cachedResponse(
        source.manifestTransportBytes,
        "application/gzip",
      ),
    );
  }

  for (const [path, description] of entries) {
    throwIfAborted(signal);
    const transportUrl = new URL(
      description.transport.path,
      source.baseUrl,
    ).href;
    let transport: Uint8Array | undefined;
    if (cache !== null) {
      const cached = await cache.match(transportUrl);
      if (cached !== undefined) {
        try {
          transport = await readTransportResponse(
            cached,
            description,
            path,
            signal,
          );
        } catch {
          await cache.delete(transportUrl);
        }
      }
    }
    if (transport === undefined) {
      const response = await fetchResponse(transportUrl, signal);
      const fileStart = completedBytes;
      transport = await readTransportResponse(
        response,
        description,
        path,
        signal,
        (fileBytes) =>
          onProgress?.({
            stage: "downloading-model",
            completed: fileStart + fileBytes,
            total: totalBytes,
            message: path,
          }),
      );
      if (cache !== null) {
        await cache.put(
          transportUrl,
          cachedResponse(transport, "application/gzip"),
        );
      }
    } else {
      onProgress?.({
        stage: "downloading-model",
        completed: completedBytes + transport.byteLength,
        total: totalBytes,
        message: path,
      });
    }
    completedBytes += transport.byteLength;
    verifiedTransports.add(path);
    if (cache === null) {
      memoryTransports.set(path, transport);
    }
  }

  return new ModelAssets(source, {
    cache,
    memoryTransports,
    verifiedTransports,
    signal,
    onProgress,
  });
}

/**
 * Mark a revision reusable only after tokenizer and all ONNX sessions have
 * initialized successfully. A missing marker intentionally leaves a resumable
 * partial download.
 */
export async function markModelCacheComplete(
  assets: ModelAssets,
): Promise<void> {
  const source = assets.source;
  const storage = cacheStorage();
  if (storage === null) return;
  if (!assets.hasVerifiedAllFiles()) {
    throw new Error(
      "Cannot complete model cache before every model file is verified",
    );
  }
  const cache = await storage.open(source.cacheName);
  const status = await inspectModelCache(source);
  if (status.complete) return;
  const descriptions = Object.values(source.manifest.files);
  const cachedTransports = await Promise.all(
    descriptions.map((description) =>
      cache.match(
        new URL(description.transport.path, source.baseUrl).href,
      ),
    ),
  );
  const missingIndex = cachedTransports.findIndex(
    (response) => response === undefined,
  );
  if (missingIndex !== -1) {
    throw new Error(
      `Cannot complete model cache with missing ${descriptions[missingIndex].transport.path}`,
    );
  }
  const marker = new TextEncoder().encode(
    JSON.stringify(markerFor(source)),
  );
  await cache.put(
    cacheMarkerUrl(source.baseUrl),
    cachedResponse(marker, "application/json"),
  );
  await removeOlderSourceCaches(source);
}

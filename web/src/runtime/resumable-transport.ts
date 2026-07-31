// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { sha256Hex } from "./hash";
import type { ManifestFile } from "./manifest";

export const RESUMABLE_TRANSPORT_BLOCK_BYTES = 8 * 1024 * 1024;

const PARTIAL_MARKER_SCHEMA_VERSION = 2;
const PARTIAL_KIND_QUERY = "ardy-model-partial";
const PARTIAL_OFFSET_QUERY = "ardy-model-offset";
const PARTIAL_SHA256_QUERY = "ardy-model-sha256";
const PARTIAL_MARKER_KIND = "marker";
const PARTIAL_BLOCK_KIND = "block";
const MAX_MARKER_BYTES = 512 * 1024;
const SHA256_RE = /^[0-9a-f]{64}$/;

interface PartialTransportBlock {
  offset: number;
  size_bytes: number;
  sha256: string;
}

interface PartialTransportMarker {
  schema_version: typeof PARTIAL_MARKER_SCHEMA_VERSION;
  transport_url: string;
  transport_size_bytes: number;
  transport_sha256: string;
  block_size_bytes: typeof RESUMABLE_TRANSPORT_BLOCK_BYTES;
  transport_sha256_verified: boolean;
  blocks: PartialTransportBlock[];
}

interface RestoredPartialTransport {
  marker: PartialTransportMarker;
  completed: number;
}

export interface LoadResumableTransportOptions {
  cache: Cache;
  transportUrl: string;
  description: ManifestFile;
  /**
   * Fetch the complete transport when `offset` is zero, or request
   * `Range: bytes=<offset>-` otherwise. Non-2xx responses must be returned so
   * the loader can apply its range fallback policy.
   */
  fetchRange: (offset: number, signal?: AbortSignal) => Promise<Response>;
  signal?: AbortSignal;
  onProgress?: (completed: number) => void;
  label?: string;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error
    ? signal.reason
    : new DOMException("Operation cancelled", "AbortError");
}

function canonicalTransportUrl(value: string): string {
  const url = new URL(value);
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username !== "" ||
    url.password !== "" ||
    url.hash !== "" ||
    url.searchParams.has(PARTIAL_KIND_QUERY) ||
    url.searchParams.has(PARTIAL_OFFSET_QUERY) ||
    url.searchParams.has(PARTIAL_SHA256_QUERY)
  ) {
    throw new TypeError(
      "Transport URL must be an HTTP(S) URL without credentials, fragment, or reserved partial-cache parameters",
    );
  }
  return url.href;
}

function partialMarkerUrl(transportUrl: string): string {
  const url = new URL(transportUrl);
  url.searchParams.set(PARTIAL_KIND_QUERY, PARTIAL_MARKER_KIND);
  return url.href;
}

function partialBlockUrl(
  transportUrl: string,
  block: PartialTransportBlock,
): string {
  const url = new URL(transportUrl);
  url.searchParams.set(PARTIAL_KIND_QUERY, PARTIAL_BLOCK_KIND);
  url.searchParams.set(PARTIAL_OFFSET_QUERY, String(block.offset));
  url.searchParams.set(PARTIAL_SHA256_QUERY, block.sha256);
  return url.href;
}

function isPartialEntryFor(
  candidateValue: string,
  transportUrl: string,
): boolean {
  let candidate: URL;
  try {
    candidate = new URL(candidateValue);
  } catch {
    return false;
  }
  const kind = candidate.searchParams.get(PARTIAL_KIND_QUERY);
  if (kind !== PARTIAL_MARKER_KIND && kind !== PARTIAL_BLOCK_KIND) {
    return false;
  }
  candidate.searchParams.delete(PARTIAL_KIND_QUERY);
  candidate.searchParams.delete(PARTIAL_OFFSET_QUERY);
  candidate.searchParams.delete(PARTIAL_SHA256_QUERY);
  return candidate.href === transportUrl;
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

function cachedResponse(bytes: Uint8Array, mediaType: string): Response {
  return new Response(exactArrayBuffer(bytes), {
    headers: {
      "content-length": String(bytes.byteLength),
      "content-type": mediaType,
    },
  });
}

async function readBoundedResponse(
  response: Response,
  maximumBytes: number,
  label: string,
): Promise<Uint8Array> {
  const declaredLength = response.headers.get("content-length");
  if (
    declaredLength !== null &&
    (!/^\d+$/.test(declaredLength) ||
      Number(declaredLength) > maximumBytes)
  ) {
    throw new Error(`${label} declares an invalid Content-Length`);
  }
  if (response.body === null) {
    throw new Error(`${label} has no response body`);
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let completed = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      completed += result.value.byteLength;
      if (completed > maximumBytes) {
        throw new Error(`${label} exceeds ${maximumBytes} bytes`);
      }
      chunks.push(result.value);
    }
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    throw error;
  }
  const output = new Uint8Array(completed);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

async function readExactResponse(
  response: Response,
  expectedBytes: number,
  label: string,
  signal?: AbortSignal,
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
    if (expectedBytes === 0) return new Uint8Array();
    throw new Error(`${label} has no response body`);
  }
  const output = new Uint8Array(expectedBytes);
  const reader = response.body.getReader();
  let completed = 0;
  try {
    while (true) {
      throwIfAborted(signal);
      const result = await reader.read();
      throwIfAborted(signal);
      if (result.done) break;
      if (completed + result.value.byteLength > expectedBytes) {
        throw new Error(`${label} exceeds ${expectedBytes} bytes`);
      }
      output.set(result.value, completed);
      completed += result.value.byteLength;
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

function finiteWholeNumber(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0
    ? value
    : null;
}

function validateMarker(
  value: unknown,
  transportUrl: string,
  description: ManifestFile,
): PartialTransportMarker | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const candidate = value as Partial<PartialTransportMarker>;
  if (
    candidate.schema_version !== PARTIAL_MARKER_SCHEMA_VERSION ||
    candidate.transport_url !== transportUrl ||
    candidate.transport_size_bytes !==
      description.transport.size_bytes ||
    candidate.transport_sha256 !== description.transport.sha256 ||
    candidate.block_size_bytes !== RESUMABLE_TRANSPORT_BLOCK_BYTES ||
    typeof candidate.transport_sha256_verified !== "boolean" ||
    !Array.isArray(candidate.blocks)
  ) {
    return null;
  }

  const maximumBlocks = Math.ceil(
    description.transport.size_bytes / RESUMABLE_TRANSPORT_BLOCK_BYTES,
  );
  if (candidate.blocks.length === 0 || candidate.blocks.length > maximumBlocks) {
    return null;
  }

  const blocks: PartialTransportBlock[] = [];
  let completed = 0;
  for (let index = 0; index < candidate.blocks.length; index += 1) {
    const rawBlock = candidate.blocks[index];
    if (
      typeof rawBlock !== "object" ||
      rawBlock === null ||
      Array.isArray(rawBlock)
    ) {
      return null;
    }
    const block = rawBlock as Partial<PartialTransportBlock>;
    const offset = finiteWholeNumber(block.offset);
    const sizeBytes = finiteWholeNumber(block.size_bytes);
    if (
      offset === null ||
      sizeBytes === null ||
      offset !== completed ||
      sizeBytes <= 0 ||
      sizeBytes > RESUMABLE_TRANSPORT_BLOCK_BYTES ||
      typeof block.sha256 !== "string" ||
      !SHA256_RE.test(block.sha256) ||
      completed + sizeBytes > description.transport.size_bytes ||
      (index < candidate.blocks.length - 1 &&
        sizeBytes !== RESUMABLE_TRANSPORT_BLOCK_BYTES)
    ) {
      return null;
    }
    blocks.push({
      offset,
      size_bytes: sizeBytes,
      sha256: block.sha256,
    });
    completed += sizeBytes;
  }
  if (
    candidate.transport_sha256_verified &&
    completed !== description.transport.size_bytes
  ) {
    return null;
  }

  return {
    schema_version: PARTIAL_MARKER_SCHEMA_VERSION,
    transport_url: transportUrl,
    transport_size_bytes: description.transport.size_bytes,
    transport_sha256: description.transport.sha256,
    block_size_bytes: RESUMABLE_TRANSPORT_BLOCK_BYTES,
    transport_sha256_verified: candidate.transport_sha256_verified,
    blocks,
  };
}

async function readMarker(
  cache: Cache,
  transportUrl: string,
  description: ManifestFile,
): Promise<PartialTransportMarker | null> {
  const response = await cache.match(partialMarkerUrl(transportUrl));
  if (response === undefined) return null;
  try {
    const bytes = await readBoundedResponse(
      response,
      MAX_MARKER_BYTES,
      "Partial transport marker",
    );
    const value = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    ) as unknown;
    return validateMarker(value, transportUrl, description);
  } catch {
    return null;
  }
}

async function writeMarker(
  cache: Cache,
  transportUrl: string,
  marker: PartialTransportMarker,
): Promise<void> {
  const bytes = new TextEncoder().encode(JSON.stringify(marker));
  if (bytes.byteLength > MAX_MARKER_BYTES) {
    throw new Error("Partial transport marker is too large");
  }
  await cache.put(
    partialMarkerUrl(transportUrl),
    cachedResponse(bytes, "application/json"),
  );
}

async function referencedBlocksExist(
  cache: Cache,
  transportUrl: string,
  marker: PartialTransportMarker,
): Promise<boolean> {
  const responses = await Promise.all(
    marker.blocks.map((block) =>
      cache.match(partialBlockUrl(transportUrl, block)),
    ),
  );
  return responses.every((response, index) => {
    if (response === undefined) return false;
    const declaredLength = response.headers.get("content-length");
    return (
      declaredLength !== null &&
      /^\d+$/.test(declaredLength) &&
      Number(declaredLength) === marker.blocks[index].size_bytes
    );
  });
}

async function removeOrphanBlocks(
  cache: Cache,
  transportUrl: string,
  marker: PartialTransportMarker,
): Promise<void> {
  const retained = new Set<string>([
    partialMarkerUrl(transportUrl),
    ...marker.blocks.map((block) =>
      partialBlockUrl(transportUrl, block),
    ),
  ]);
  const deletions: Array<Promise<boolean>> = [];
  for (const request of await cache.keys()) {
    if (
      isPartialEntryFor(request.url, transportUrl) &&
      !retained.has(request.url)
    ) {
      deletions.push(cache.delete(request));
    }
  }
  await Promise.all(deletions);
}

export async function clearResumableTransport(
  cache: Cache,
  transportUrlValue: string,
): Promise<void> {
  const transportUrl = canonicalTransportUrl(transportUrlValue);
  const deletions: Array<Promise<boolean>> = [];
  for (const request of await cache.keys()) {
    if (isPartialEntryFor(request.url, transportUrl)) {
      deletions.push(cache.delete(request));
    }
  }
  await Promise.all(deletions);
}

/**
 * Returns the contiguous, structurally valid prefix retained for a transport.
 * This intentionally avoids reading and hashing large block bodies; full block
 * integrity is checked before any retained bytes are used for a resumed fetch.
 */
export async function inspectResumableTransport(
  cache: Cache,
  transportUrlValue: string,
  description: ManifestFile,
): Promise<number> {
  const transportUrl = canonicalTransportUrl(transportUrlValue);
  const marker = await readMarker(cache, transportUrl, description);
  if (
    marker === null ||
    !(await referencedBlocksExist(cache, transportUrl, marker))
  ) {
    await clearResumableTransport(cache, transportUrl);
    return 0;
  }
  await removeOrphanBlocks(cache, transportUrl, marker);
  return marker.blocks.reduce(
    (total, block) => total + block.size_bytes,
    0,
  );
}

export interface ResumableTransportCacheStatus {
  sizeBytes: number;
  complete: boolean;
}

export async function inspectResumableTransportCache(
  cache: Cache,
  transportUrlValue: string,
  description: ManifestFile,
): Promise<ResumableTransportCacheStatus> {
  const transportUrl = canonicalTransportUrl(transportUrlValue);
  const marker = await readMarker(cache, transportUrl, description);
  if (
    marker === null ||
    !(await referencedBlocksExist(cache, transportUrl, marker))
  ) {
    await clearResumableTransport(cache, transportUrl);
    return { sizeBytes: 0, complete: false };
  }
  await removeOrphanBlocks(cache, transportUrl, marker);
  const sizeBytes = markerCompleted(marker);
  return {
    sizeBytes,
    complete:
      marker.transport_sha256_verified &&
      sizeBytes === description.transport.size_bytes,
  };
}

async function restorePartialTransport(
  cache: Cache,
  transportUrl: string,
  description: ManifestFile,
  output: Uint8Array,
  signal?: AbortSignal,
): Promise<RestoredPartialTransport | null> {
  const marker = await readMarker(cache, transportUrl, description);
  if (marker === null) {
    await clearResumableTransport(cache, transportUrl);
    return null;
  }
  try {
    let completed = 0;
    for (const block of marker.blocks) {
      throwIfAborted(signal);
      const response = await cache.match(
        partialBlockUrl(transportUrl, block),
      );
      if (response === undefined) {
        throw new Error("Partial transport block is missing");
      }
      const bytes = await readExactResponse(
        response,
        block.size_bytes,
        "Partial transport block",
        signal,
      );
      if ((await sha256Hex(bytes)) !== block.sha256) {
        throw new Error("Partial transport block SHA-256 mismatch");
      }
      throwIfAborted(signal);
      output.set(bytes, block.offset);
      completed += bytes.byteLength;
    }
    await removeOrphanBlocks(cache, transportUrl, marker);
    return { marker, completed };
  } catch (error) {
    if (signal?.aborted) throw error;
    await clearResumableTransport(cache, transportUrl);
    return null;
  }
}

export async function readCachedResumableTransport(options: {
  cache: Cache;
  transportUrl: string;
  description: ManifestFile;
  signal?: AbortSignal;
  label?: string;
}): Promise<Uint8Array | null> {
  const transportUrl = canonicalTransportUrl(options.transportUrl);
  const expectedBytes = options.description.transport.size_bytes;
  const output = new Uint8Array(expectedBytes);
  const restored = await restorePartialTransport(
    options.cache,
    transportUrl,
    options.description,
    output,
    options.signal,
  );
  if (restored?.completed !== expectedBytes) return null;
  if (!restored.marker.transport_sha256_verified) return null;
  try {
    await verifyTransport(
      output,
      options.description,
      options.label ?? `${transportUrl} gzip transport`,
    );
    throwIfAborted(options.signal);
    return output;
  } catch (error) {
    if (options.signal?.aborted) throw error;
    await clearResumableTransport(options.cache, transportUrl);
    return null;
  }
}

function emptyMarker(
  transportUrl: string,
  description: ManifestFile,
): PartialTransportMarker {
  return {
    schema_version: PARTIAL_MARKER_SCHEMA_VERSION,
    transport_url: transportUrl,
    transport_size_bytes: description.transport.size_bytes,
    transport_sha256: description.transport.sha256,
    block_size_bytes: RESUMABLE_TRANSPORT_BLOCK_BYTES,
    transport_sha256_verified: false,
    blocks: [],
  };
}

async function persistBlock(
  cache: Cache,
  transportUrl: string,
  marker: PartialTransportMarker,
  offset: number,
  bytes: Uint8Array,
): Promise<PartialTransportMarker> {
  if (
    bytes.byteLength <= 0 ||
    bytes.byteLength > RESUMABLE_TRANSPORT_BLOCK_BYTES
  ) {
    throw new Error("Partial transport block has an invalid size");
  }
  const sha256 = await sha256Hex(bytes);
  const block = {
    offset,
    size_bytes: bytes.byteLength,
    sha256,
  };
  const existingIndex = marker.blocks.findIndex(
    (candidate) => candidate.offset === offset,
  );
  if (
    existingIndex !== -1 &&
    existingIndex !== marker.blocks.length - 1
  ) {
    throw new Error("Only the partial transport tail may be replaced");
  }
  const previous =
    existingIndex === -1 ? undefined : marker.blocks[existingIndex];
  const blocks =
    existingIndex === -1
      ? [...marker.blocks, block]
      : marker.blocks.map((candidate, index) =>
          index === existingIndex ? block : candidate,
        );
  const nextMarker = {
    ...marker,
    transport_sha256_verified: false,
    blocks,
  };
  const nextBlockUrl = partialBlockUrl(transportUrl, block);
  await cache.put(
    nextBlockUrl,
    cachedResponse(bytes, "application/octet-stream"),
  );
  try {
    await writeMarker(cache, transportUrl, nextMarker);
  } catch (error) {
    await cache.delete(nextBlockUrl).catch(() => false);
    throw error;
  }
  if (previous !== undefined) {
    const previousUrl = partialBlockUrl(transportUrl, previous);
    if (previousUrl !== nextBlockUrl) {
      await cache.delete(previousUrl);
    }
  }
  return nextMarker;
}

function markerCompleted(marker: PartialTransportMarker): number {
  return marker.blocks.reduce(
    (total, block) => total + block.size_bytes,
    0,
  );
}

function currentBlockStart(marker: PartialTransportMarker): number {
  const last = marker.blocks.at(-1);
  if (
    last !== undefined &&
    last.size_bytes < RESUMABLE_TRANSPORT_BLOCK_BYTES
  ) {
    return last.offset;
  }
  return markerCompleted(marker);
}

interface ReadNetworkOptions {
  response: Response;
  output: Uint8Array;
  start: number;
  expectedBytes: number;
  cache: Cache;
  transportUrl: string;
  marker: PartialTransportMarker;
  signal?: AbortSignal;
  onProgress?: (completed: number) => void;
  label: string;
}

async function readNetworkResponse(
  options: ReadNetworkOptions,
): Promise<PartialTransportMarker> {
  const declaredLength = options.response.headers.get("content-length");
  if (
    declaredLength !== null &&
    (!/^\d+$/.test(declaredLength) ||
      Number(declaredLength) !== options.expectedBytes)
  ) {
    throw new Error(
      `${options.label} Content-Length does not match the expected ${options.expectedBytes} bytes`,
    );
  }
  if (options.response.body === null) {
    throw new Error(`${options.label} has no response body`);
  }

  const reader = options.response.body.getReader();
  let marker = options.marker;
  let completed = options.start;
  const end = options.start + options.expectedBytes;
  let blockStart = currentBlockStart(marker);
  let receivedNewBytes = false;

  const persistTail = async (): Promise<void> => {
    if (!receivedNewBytes || completed <= blockStart) return;
    marker = await persistBlock(
      options.cache,
      options.transportUrl,
      marker,
      blockStart,
      options.output.slice(blockStart, completed),
    );
    blockStart = currentBlockStart(marker);
    receivedNewBytes = false;
  };

  try {
    while (true) {
      throwIfAborted(options.signal);
      const result = await reader.read();
      throwIfAborted(options.signal);
      if (result.done) break;
      if (completed + result.value.byteLength > end) {
        throw new Error(`${options.label} exceeds its declared size`);
      }
      options.output.set(result.value, completed);
      completed += result.value.byteLength;
      receivedNewBytes = true;

      while (
        blockStart + RESUMABLE_TRANSPORT_BLOCK_BYTES <=
        completed
      ) {
        const blockEnd =
          blockStart + RESUMABLE_TRANSPORT_BLOCK_BYTES;
        marker = await persistBlock(
          options.cache,
          options.transportUrl,
          marker,
          blockStart,
          options.output.slice(blockStart, blockEnd),
        );
        blockStart = currentBlockStart(marker);
        receivedNewBytes = completed > markerCompleted(marker);
      }

      options.onProgress?.(completed);
      throwIfAborted(options.signal);
    }
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    await persistTail();
    throw error;
  }

  if (completed !== end) {
    const error = new Error(
      `${options.label} size mismatch: expected ${options.expectedBytes}, got ${completed - options.start}`,
    );
    await persistTail();
    throw error;
  }
  await persistTail();
  return marker;
}

function parseContentRange(
  response: Response,
  expectedStart: number,
  expectedTotal: number,
): number | null {
  const value = response.headers.get("content-range");
  if (value === null) return null;
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(value);
  if (match === null) return null;
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = Number(match[3]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    !Number.isSafeInteger(total) ||
    start !== expectedStart ||
    end !== expectedTotal - 1 ||
    total !== expectedTotal ||
    end < start
  ) {
    return null;
  }
  const rangeBytes = end - start + 1;
  const declaredLength = response.headers.get("content-length");
  if (
    declaredLength !== null &&
    (!/^\d+$/.test(declaredLength) ||
      Number(declaredLength) !== rangeBytes)
  ) {
    return null;
  }
  return rangeBytes;
}

async function discardResponse(response: Response): Promise<void> {
  await response.body?.cancel().catch(() => {});
}

async function verifyTransport(
  bytes: Uint8Array,
  description: ManifestFile,
  label: string,
): Promise<void> {
  if (bytes.byteLength !== description.transport.size_bytes) {
    throw new Error(
      `Compressed size mismatch for ${label}: expected ${description.transport.size_bytes}, got ${bytes.byteLength}`,
    );
  }
  const actual = await sha256Hex(bytes);
  if (actual !== description.transport.sha256) {
    throw new Error(
      `Compressed SHA-256 mismatch for ${label}: expected ${description.transport.sha256}, got ${actual}`,
    );
  }
}

async function promoteTransport(
  cache: Cache,
  transportUrl: string,
  bytes: Uint8Array,
): Promise<boolean> {
  // Chromium may reject very large Cache Storage responses even though the
  // same total fits when represented by the verified resumable blocks. Keep
  // any transport larger than one block in that bounded representation.
  if (bytes.byteLength > RESUMABLE_TRANSPORT_BLOCK_BYTES) return false;
  try {
    await cache.put(
      transportUrl,
      cachedResponse(bytes, "application/gzip"),
    );
  } catch {
    // The complete block set remains a valid persistent representation.
    return false;
  }
  await clearResumableTransport(cache, transportUrl);
  return true;
}

async function finalizeTransport(
  cache: Cache,
  transportUrl: string,
  marker: PartialTransportMarker,
  bytes: Uint8Array,
): Promise<void> {
  if (await promoteTransport(cache, transportUrl, bytes)) return;
  await writeMarker(cache, transportUrl, {
    ...marker,
    transport_sha256_verified: true,
  });
}

async function fetchCleanFullTransport(
  options: LoadResumableTransportOptions,
  transportUrl: string,
  output: Uint8Array,
  label: string,
): Promise<Uint8Array> {
  throwIfAborted(options.signal);
  const response = await options.fetchRange(0, options.signal);
  throwIfAborted(options.signal);
  if (response.status !== 200) {
    throw new Error(
      `${label} clean download failed with HTTP ${response.status}`,
    );
  }
  let marker = emptyMarker(transportUrl, options.description);
  marker = await readNetworkResponse({
    response,
    output,
    start: 0,
    expectedBytes: output.byteLength,
    cache: options.cache,
    transportUrl,
    marker,
    signal: options.signal,
    onProgress: options.onProgress,
    label,
  });
  try {
    await verifyTransport(output, options.description, label);
  } catch (error) {
    await clearResumableTransport(options.cache, transportUrl);
    throw error;
  }
  throwIfAborted(options.signal);
  await finalizeTransport(
    options.cache,
    transportUrl,
    marker,
    output,
  );
  throwIfAborted(options.signal);
  return output;
}

/**
 * Restore a verified prefix from Cache Storage and complete the immutable gzip
 * transport with one HTTP range request. The caller owns HTTP request headers;
 * this function owns response validation, partial persistence, integrity, and
 * finalization into a verified persistent representation.
 */
export async function loadResumableTransport(
  options: LoadResumableTransportOptions,
): Promise<Uint8Array> {
  const transportUrl = canonicalTransportUrl(options.transportUrl);
  const expectedBytes = options.description.transport.size_bytes;
  const label = options.label ?? `${transportUrl} gzip transport`;
  if (
    !Number.isSafeInteger(expectedBytes) ||
    expectedBytes <= 0
  ) {
    throw new RangeError(`${label} declares an invalid byte length`);
  }
  throwIfAborted(options.signal);

  let reportedBytes = 0;
  const reportProgress = (completed: number): void => {
    const bounded = Math.max(0, Math.min(expectedBytes, completed));
    if (bounded <= reportedBytes) return;
    reportedBytes = bounded;
    options.onProgress?.(bounded);
  };
  // The exact terminal value means a verified persistent representation is
  // already in Cache Storage. Keeping streamed progress one byte below that
  // value also leaves cancellation available during a clean integrity retry.
  const reportStreamingProgress = (completed: number): void => {
    reportProgress(Math.min(completed, expectedBytes - 1));
  };
  const operationOptions: LoadResumableTransportOptions = {
    ...options,
    onProgress: reportStreamingProgress,
  };
  const cleanFullTransport = async (): Promise<Uint8Array> => {
    const bytes = await fetchCleanFullTransport(
      operationOptions,
      transportUrl,
      new Uint8Array(expectedBytes),
      label,
    );
    reportProgress(expectedBytes);
    return bytes;
  };

  const output = new Uint8Array(expectedBytes);
  const restored = await restorePartialTransport(
    options.cache,
    transportUrl,
    options.description,
    output,
    options.signal,
  );
  let marker =
    restored?.marker ?? emptyMarker(transportUrl, options.description);
  let completed = restored?.completed ?? 0;
  if (completed > 0) reportStreamingProgress(completed);
  throwIfAborted(options.signal);

  if (completed === expectedBytes) {
    try {
      await verifyTransport(output, options.description, label);
    } catch {
      await clearResumableTransport(options.cache, transportUrl);
      return cleanFullTransport();
    }
    throwIfAborted(options.signal);
    await finalizeTransport(
      options.cache,
      transportUrl,
      marker,
      output,
    );
    throwIfAborted(options.signal);
    reportProgress(expectedBytes);
    return output;
  }

  const response = await options.fetchRange(completed, options.signal);
  throwIfAborted(options.signal);
  let joinedPartial = completed > 0;

  if (completed > 0 && response.status === 200) {
    await clearResumableTransport(options.cache, transportUrl);
    marker = emptyMarker(transportUrl, options.description);
    completed = 0;
    joinedPartial = false;
  } else if (completed > 0 && response.status === 206) {
    const rangeBytes = parseContentRange(
      response,
      completed,
      expectedBytes,
    );
    if (rangeBytes === null) {
      await discardResponse(response);
      await clearResumableTransport(options.cache, transportUrl);
      return cleanFullTransport();
    }
  } else if (completed > 0) {
    await discardResponse(response);
    await clearResumableTransport(options.cache, transportUrl);
    return cleanFullTransport();
  } else if (response.status !== 200) {
    throw new Error(`${label} download failed with HTTP ${response.status}`);
  }

  const responseBytes =
    response.status === 206 ? expectedBytes - completed : expectedBytes;
  marker = await readNetworkResponse({
    response,
    output,
    start: completed,
    expectedBytes: responseBytes,
    cache: options.cache,
    transportUrl,
    marker,
    signal: options.signal,
    onProgress: reportStreamingProgress,
    label,
  });
  try {
    await verifyTransport(output, options.description, label);
  } catch (error) {
    await clearResumableTransport(options.cache, transportUrl);
    if (joinedPartial) {
      return cleanFullTransport();
    }
    throw error;
  }
  throwIfAborted(options.signal);
  await finalizeTransport(
    options.cache,
    transportUrl,
    marker,
    output,
  );
  throwIfAborted(options.signal);
  reportProgress(expectedBytes);
  return output;
}

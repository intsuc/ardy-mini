// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { sha256Hex } from "./hash";
import type { ManifestFile } from "./manifest";
import {
  inspectResumableTransport,
  loadResumableTransport,
  RESUMABLE_TRANSPORT_BLOCK_BYTES,
} from "./resumable-transport";

class MemoryCache {
  readonly entries = new Map<string, Response>();
  onPut?: (url: string) => void;

  async match(request: RequestInfo | URL): Promise<Response | undefined> {
    return this.entries.get(requestUrl(request))?.clone();
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    const url = requestUrl(request);
    this.entries.set(url, response.clone());
    this.onPut?.(url);
  }

  async delete(request: RequestInfo | URL): Promise<boolean> {
    return this.entries.delete(requestUrl(request));
  }

  async keys(): Promise<readonly Request[]> {
    return [...this.entries.keys()].map((url) => new Request(url));
  }
}

function requestUrl(request: RequestInfo | URL): string {
  if (typeof request === "string") return request;
  if (request instanceof URL) return request.href;
  return request.url;
}

function response(
  bytes: Uint8Array,
  options: {
    status?: number;
    contentRange?: string;
  } = {},
): Response {
  const headers = new Headers({
    "content-length": String(bytes.byteLength),
  });
  if (options.contentRange !== undefined) {
    headers.set("content-range", options.contentRange);
  }
  return new Response(new Uint8Array(bytes).buffer, {
    status: options.status ?? 200,
    headers,
  });
}

function chunkedResponse(
  chunks: readonly Uint8Array[],
  totalBytes: number,
): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    }),
    {
      status: 200,
      headers: { "content-length": String(totalBytes) },
    },
  );
}

async function fixture(byteLength = 97): Promise<{
  bytes: Uint8Array;
  description: ManifestFile;
  transportUrl: string;
}> {
  const bytes = new Uint8Array(byteLength);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = (index * 37 + 11) % 256;
  }
  return {
    bytes,
    description: {
      sha256: "0".repeat(64),
      size_bytes: 1,
      transport: {
        path: "denoiser.onnx.gz",
        compression: "gzip",
        sha256: await sha256Hex(bytes),
        size_bytes: bytes.byteLength,
      },
    },
    transportUrl:
      "https://models.example.test/revision/denoiser.onnx.gz",
  };
}

function partialRequests(
  cache: MemoryCache,
  transportUrl: string,
): string[] {
  return [...cache.entries.keys()].filter((url) => {
    const candidate = new URL(url);
    return (
      candidate.origin + candidate.pathname ===
        new URL(transportUrl).origin + new URL(transportUrl).pathname &&
      candidate.searchParams.has("ardy-model-partial")
    );
  });
}

async function interruptedPrefix(options: {
  cache: MemoryCache;
  bytes: Uint8Array;
  description: ManifestFile;
  transportUrl: string;
  prefixBytes?: number;
}): Promise<number> {
  const prefixBytes = options.prefixBytes ?? 31;
  const controller = new AbortController();
  const first = options.bytes.slice(0, prefixBytes);
  const remainder = options.bytes.slice(prefixBytes);

  await expect(
    loadResumableTransport({
      cache: options.cache as unknown as Cache,
      transportUrl: options.transportUrl,
      description: options.description,
      signal: controller.signal,
      fetchRange: async (offset) => {
        expect(offset).toBe(0);
        return chunkedResponse(
          [first, remainder],
          options.bytes.byteLength,
        );
      },
      onProgress(completed) {
        if (completed === prefixBytes) {
          controller.abort(new DOMException("Paused", "AbortError"));
        }
      },
    }),
  ).rejects.toMatchObject({ name: "AbortError" });

  expect(
    await inspectResumableTransport(
      options.cache as unknown as Cache,
      options.transportUrl,
      options.description,
    ),
  ).toBe(prefixBytes);
  return prefixBytes;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("resumable model transports", () => {
  it("persists an exact interrupted prefix, resumes with Range, and promotes it", async () => {
    const { bytes, description, transportUrl } = await fixture();
    const cache = new MemoryCache();
    const prefixBytes = await interruptedPrefix({
      cache,
      bytes,
      description,
      transportUrl,
    });
    const requestedOffsets: number[] = [];
    const completionOrder: string[] = [];
    cache.onPut = (url) => {
      if (url === transportUrl) completionOrder.push("canonical");
    };

    const loaded = await loadResumableTransport({
      cache: cache as unknown as Cache,
      transportUrl,
      description,
      fetchRange: async (offset) => {
        requestedOffsets.push(offset);
        expect(offset).toBe(prefixBytes);
        const suffix = bytes.slice(offset);
        return response(suffix, {
          status: 206,
          contentRange: `bytes ${offset}-${bytes.byteLength - 1}/${bytes.byteLength}`,
        });
      },
      onProgress: (completed) => {
        if (completed === bytes.byteLength) {
          completionOrder.push("terminal-progress");
        }
      },
    });

    expect(loaded).toEqual(bytes);
    expect(requestedOffsets).toEqual([prefixBytes]);
    expect(completionOrder).toEqual([
      "canonical",
      "terminal-progress",
    ]);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
    expect(
      new Uint8Array(
        await (await cache.match(transportUrl))!.arrayBuffer(),
      ),
    ).toEqual(bytes);
    expect(
      await inspectResumableTransport(
        cache as unknown as Cache,
        transportUrl,
        description,
      ),
    ).toBe(0);
  });

  it("finishes canonical promotion when cancellation arrives during Cache.put", async () => {
    const { bytes, description, transportUrl } = await fixture();
    const cache = new MemoryCache();
    const controller = new AbortController();
    const progress: number[] = [];
    cache.onPut = (url) => {
      if (url === transportUrl) {
        controller.abort(new DOMException("Paused", "AbortError"));
      }
    };

    await expect(
      loadResumableTransport({
        cache: cache as unknown as Cache,
        transportUrl,
        description,
        signal: controller.signal,
        fetchRange: async (offset) => {
          expect(offset).toBe(0);
          return response(bytes);
        },
        onProgress: (completed) => progress.push(completed),
      }),
    ).rejects.toMatchObject({ name: "AbortError" });

    expect(progress).not.toContain(bytes.byteLength);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
    expect(
      new Uint8Array(
        await (await cache.match(transportUrl))!.arrayBuffer(),
      ),
    ).toEqual(bytes);
  });

  it("extends a retained short tail across fixed-size block boundaries", async () => {
    const { bytes, description, transportUrl } = await fixture(
      RESUMABLE_TRANSPORT_BLOCK_BYTES + 257,
    );
    const cache = new MemoryCache();
    const prefixBytes =
      RESUMABLE_TRANSPORT_BLOCK_BYTES + 123;
    await interruptedPrefix({
      cache,
      bytes,
      description,
      transportUrl,
      prefixBytes,
    });

    const loaded = await loadResumableTransport({
      cache: cache as unknown as Cache,
      transportUrl,
      description,
      fetchRange: async (offset) => {
        expect(offset).toBe(prefixBytes);
        return response(bytes.slice(offset), {
          status: 206,
          contentRange:
            `bytes ${offset}-${bytes.byteLength - 1}/${bytes.byteLength}`,
        });
      },
    });

    expect(loaded).toEqual(bytes);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
  }, 20_000);

  it("falls back to the full 200 response when the server ignores Range", async () => {
    const { bytes, description, transportUrl } = await fixture();
    const cache = new MemoryCache();
    const prefixBytes = await interruptedPrefix({
      cache,
      bytes,
      description,
      transportUrl,
    });
    const requestedOffsets: number[] = [];
    const progress: number[] = [];

    const loaded = await loadResumableTransport({
      cache: cache as unknown as Cache,
      transportUrl,
      description,
      fetchRange: async (offset) => {
        requestedOffsets.push(offset);
        return response(bytes);
      },
      onProgress: (completed) => progress.push(completed),
    });

    expect(loaded).toEqual(bytes);
    expect(requestedOffsets).toEqual([prefixBytes]);
    expect(progress).toEqual(
      [...progress].sort((left, right) => left - right),
    );
    expect(progress.at(-1)).toBe(bytes.byteLength);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
  });

  it.each([
    {
      name: "malformed 206",
      failedResponse(bytes: Uint8Array, offset: number) {
        return response(bytes.slice(offset), {
          status: 206,
          contentRange: `bytes 0-${bytes.byteLength - offset - 1}/${bytes.byteLength}`,
        });
      },
    },
    {
      name: "416",
      failedResponse() {
        return new Response(null, {
          status: 416,
          headers: { "content-range": "bytes */97" },
        });
      },
    },
  ])("retries $name once as a clean full request", async ({
    failedResponse,
  }) => {
    const { bytes, description, transportUrl } = await fixture();
    const cache = new MemoryCache();
    const prefixBytes = await interruptedPrefix({
      cache,
      bytes,
      description,
      transportUrl,
    });
    const requestedOffsets: number[] = [];

    const loaded = await loadResumableTransport({
      cache: cache as unknown as Cache,
      transportUrl,
      description,
      fetchRange: async (offset) => {
        requestedOffsets.push(offset);
        return offset === 0
          ? response(bytes)
          : failedResponse(bytes, offset);
      },
    });

    expect(loaded).toEqual(bytes);
    expect(requestedOffsets).toEqual([prefixBytes, 0]);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
  });

  it("retries one clean full request when joined bytes fail SHA-256", async () => {
    const { bytes, description, transportUrl } = await fixture();
    const cache = new MemoryCache();
    const prefixBytes = await interruptedPrefix({
      cache,
      bytes,
      description,
      transportUrl,
    });
    const requestedOffsets: number[] = [];
    const corruptSuffix = bytes.slice(prefixBytes);
    corruptSuffix[corruptSuffix.length - 1] ^= 0xff;

    const loaded = await loadResumableTransport({
      cache: cache as unknown as Cache,
      transportUrl,
      description,
      fetchRange: async (offset) => {
        requestedOffsets.push(offset);
        if (offset === 0) return response(bytes);
        return response(corruptSuffix, {
          status: 206,
          contentRange: `bytes ${offset}-${bytes.byteLength - 1}/${bytes.byteLength}`,
        });
      },
    });

    expect(loaded).toEqual(bytes);
    expect(requestedOffsets).toEqual([prefixBytes, 0]);
    expect(partialRequests(cache, transportUrl)).toEqual([]);
  });

  it.each(["missing", "corrupt"] as const)(
    "cleans a %s retained block and restarts from byte zero",
    async (damage) => {
      const { bytes, description, transportUrl } = await fixture();
      const cache = new MemoryCache();
      await interruptedPrefix({
        cache,
        bytes,
        description,
        transportUrl,
      });
      const blockUrl = partialRequests(cache, transportUrl).find(
        (url) =>
          new URL(url).searchParams.get("ardy-model-partial") ===
          "block",
      );
      expect(blockUrl).toBeDefined();
      if (damage === "missing") {
        await cache.delete(blockUrl!);
      } else {
        const retained = await cache.match(blockUrl!);
        const damaged = new Uint8Array(await retained!.arrayBuffer());
        damaged[0] ^= 0xff;
        await cache.put(blockUrl!, response(damaged));
      }

      const requestedOffsets: number[] = [];
      const loaded = await loadResumableTransport({
        cache: cache as unknown as Cache,
        transportUrl,
        description,
        fetchRange: async (offset) => {
          requestedOffsets.push(offset);
          return response(bytes);
        },
      });

      expect(loaded).toEqual(bytes);
      expect(requestedOffsets).toEqual([0]);
      expect(partialRequests(cache, transportUrl)).toEqual([]);
    },
  );
});

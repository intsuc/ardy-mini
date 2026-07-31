// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  createModelTestFixture,
  gzipTestBytes,
} from "./model-assets.test-fixture";
import { sha256Hex } from "./hash";
import {
  clearModelCache,
  fetchModelManifest,
  formatModelBytes,
  inspectModelCache,
  loadModelAssets,
  markModelCacheComplete,
  modelRawSize,
  modelTransportSize,
  normalizeModelBaseUrl,
} from "./model-assets";

class MemoryCache {
  readonly #entries = new Map<string, Response>();

  async match(request: RequestInfo | URL): Promise<Response | undefined> {
    return this.#entries.get(requestUrl(request))?.clone();
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    this.#entries.set(requestUrl(request), response.clone());
  }

  async delete(request: RequestInfo | URL): Promise<boolean> {
    return this.#entries.delete(requestUrl(request));
  }

  async keys(): Promise<readonly Request[]> {
    return [...this.#entries].map(([url]) => new Request(url));
  }
}

class MemoryCacheStorage {
  readonly #caches = new Map<string, MemoryCache>();

  async open(name: string): Promise<Cache> {
    let cache = this.#caches.get(name);
    if (cache === undefined) {
      cache = new MemoryCache();
      this.#caches.set(name, cache);
    }
    return cache as unknown as Cache;
  }

  async has(name: string): Promise<boolean> {
    return this.#caches.has(name);
  }

  async delete(name: string): Promise<boolean> {
    return this.#caches.delete(name);
  }

  async keys(): Promise<string[]> {
    return [...this.#caches.keys()];
  }
}

function requestUrl(request: RequestInfo | URL): string {
  if (typeof request === "string") return request;
  if (request instanceof URL) return request.href;
  return request.url;
}

const baseUrl = "https://models.example.test/ardy/core40/";
const originalCaches = Object.getOwnPropertyDescriptor(
  globalThis,
  "caches",
);

function setCaches(value: CacheStorage | undefined): void {
  if (value === undefined) {
    Reflect.deleteProperty(globalThis, "caches");
    return;
  }
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value,
  });
}

function restoreCaches(): void {
  if (originalCaches === undefined) {
    Reflect.deleteProperty(globalThis, "caches");
  } else {
    Object.defineProperty(globalThis, "caches", originalCaches);
  }
}

async function responseFor(
  bytes: Uint8Array,
  status = 200,
): Promise<Response> {
  return new Response(new Uint8Array(bytes).buffer, {
    status,
    headers: {
      "content-length": String(bytes.byteLength),
    },
  });
}

beforeEach(() => {
  setCaches(new MemoryCacheStorage() as unknown as CacheStorage);
});

afterEach(() => {
  vi.unstubAllGlobals();
  restoreCaches();
});

describe("individual browser model files", () => {
  it("fetches model.json.gz and exposes immutable size/cache metadata", async () => {
    const fixture = await createModelTestFixture();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      expect(requestUrl(input)).toBe(`${baseUrl}model.json.gz`);
      return responseFor(fixture.manifestTransportBytes);
    });
    vi.stubGlobal("fetch", fetchMock);

    const source = await fetchModelManifest(baseUrl);
    const status = await inspectModelCache(source);

    expect(source.manifest.model.revision).toBe("fixture-revision-1");
    expect(source.cacheName).toContain(source.manifestSha256);
    expect(status).toMatchObject({
      supported: true,
      cached: false,
      complete: false,
      fileCount: 5,
      cachedFileCount: 0,
    });
    expect(status.transportSizeBytes).toBe(
      modelTransportSize(fixture.manifest),
    );
    expect(status.rawSizeBytes).toBe(modelRawSize(fixture.manifest));
    expect(formatModelBytes(1024 ** 3)).toBe("1.0 GiB");
  });

  it("keeps FP16 and FP32 model-family caches isolated", async () => {
    const fp16BaseUrl = `${baseUrl}fp16/`;
    const fp32BaseUrl = `${baseUrl}fp32/`;
    const fp16Fixture = await createModelTestFixture("fp16");
    const fp32Fixture = await createModelTestFixture("fp32");
    const fixtures = new Map([
      [fp16BaseUrl, fp16Fixture],
      [fp32BaseUrl, fp32Fixture],
    ]);
    const requested: string[] = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = requestUrl(input);
        requested.push(url);
        const entry = [...fixtures].find(([variantBaseUrl]) =>
          url.startsWith(variantBaseUrl),
        );
        if (entry === undefined) {
          return new Response(null, { status: 404 });
        }
        const [variantBaseUrl, fixture] = entry;
        const path = decodeURIComponent(url.slice(variantBaseUrl.length));
        if (path === "model.json.gz") {
          return responseFor(fixture.manifestTransportBytes);
        }
        const transport = fixture.transports.get(path);
        return transport === undefined
          ? new Response(null, { status: 404 })
          : responseFor(transport);
      }),
    );

    const fp16Assets = await loadModelAssets(fp16BaseUrl);
    await Promise.all(
      Object.keys(fp16Assets.manifest.files).map((path) =>
        fp16Assets.read(path),
      ),
    );
    await markModelCacheComplete(fp16Assets);
    const fp32Source = await fetchModelManifest(fp32BaseUrl);

    expect(fp32Source.manifest.model).toEqual(
      fp16Assets.source.manifest.model,
    );
    expect(fp16Assets.source.manifest.precision.format).toBe(
      "mixed-fp16",
    );
    expect(fp32Source.manifest.precision.format).toBe("fp32");
    expect(fp32Source.cacheName).not.toBe(fp16Assets.source.cacheName);
    expect(await globalThis.caches.has(fp16Assets.source.cacheName)).toBe(
      true,
    );
    expect(await globalThis.caches.has(fp32Source.cacheName)).toBe(false);
    expect(await inspectModelCache(fp32Source)).toMatchObject({
      cached: false,
      cachedFileCount: 0,
      cachedTransportSizeBytes: 0,
    });

    requested.length = 0;
    const fp32Assets = await loadModelAssets(fp32BaseUrl);
    expect(
      requested.filter((url) => !url.endsWith("model.json.gz")),
    ).toHaveLength(Object.keys(fp32Fixture.manifest.files).length);
    expect(requested.every((url) => url.startsWith(fp32BaseUrl))).toBe(
      true,
    );
    await Promise.all(
      Object.keys(fp32Assets.manifest.files).map((path) =>
        fp32Assets.read(path),
      ),
    );
    await markModelCacheComplete(fp32Assets);
    expect(await globalThis.caches.has(fp16Assets.source.cacheName)).toBe(
      true,
    );
    expect(await globalThis.caches.has(fp32Assets.source.cacheName)).toBe(
      true,
    );
    await expect(
      Promise.all([
        inspectModelCache(fp16Assets.source),
        inspectModelCache(fp32Assets.source),
      ]),
    ).resolves.toEqual([
      expect.objectContaining({
        cached: true,
        complete: true,
        cachedFileCount: Object.keys(fp16Fixture.manifest.files).length,
      }),
      expect.objectContaining({
        cached: true,
        complete: true,
        cachedFileCount: Object.keys(fp32Fixture.manifest.files).length,
      }),
    ]);
  });

  it("downloads gzip files cumulatively, verifies once on read, and completes the cache explicitly", async () => {
    const fixture = await createModelTestFixture();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = requestUrl(input);
      if (url === `${baseUrl}model.json.gz`) {
        return responseFor(fixture.manifestTransportBytes);
      }
      const path = new URL(url).pathname.split("/").slice(-2).join("/");
      const bytes =
        fixture.transports.get(path) ??
        fixture.transports.get(new URL(url).pathname.split("/").at(-1)!);
      if (bytes === undefined) return new Response(null, { status: 404 });
      return responseFor(bytes);
    });
    vi.stubGlobal("fetch", fetchMock);
    const progress: Array<{
      stage: string;
      completed: number;
      total: number;
      message?: string;
    }> = [];

    const assets = await loadModelAssets(
      baseUrl,
      (event) => progress.push(event),
    );
    const download = progress.filter(
      (event) => event.stage === "downloading-model",
    );
    expect(download.at(-1)).toMatchObject({
      completed: modelTransportSize(fixture.manifest),
      total: modelTransportSize(fixture.manifest),
    });
    expect(
      progress.some((event) => event.stage === "verifying-model"),
    ).toBe(false);

    const denoiser = await assets.read("denoiser.onnx");
    expect(new TextDecoder().decode(denoiser)).toBe("fixture denoiser");
    expect(progress.at(-1)).toMatchObject({
      stage: "verifying-model",
      completed: 1,
      total: 5,
      message: "denoiser.onnx",
    });
    assets.release("denoiser.onnx");

    expect((await inspectModelCache(assets.source)).complete).toBe(false);
    await expect(markModelCacheComplete(assets)).rejects.toThrow(
      /every model file is verified/,
    );
    await Promise.all(
      Object.keys(assets.manifest.files)
        .filter((path) => path !== "denoiser.onnx")
        .map((path) => assets.read(path)),
    );
    const staleCacheName =
      assets.source.cacheName.slice(
        0,
        assets.source.cacheName.length -
          assets.source.manifestSha256.length,
      ) + "0".repeat(64);
    await globalThis.caches.open(staleCacheName);
    await markModelCacheComplete(assets);
    expect(await inspectModelCache(assets.source)).toMatchObject({
      cached: true,
      complete: true,
      cachedFileCount: 5,
    });
    expect(await globalThis.caches.has(staleCacheName)).toBe(false);
  });

  it("reuses complete cached transports and downloads only missing files", async () => {
    const fixture = await createModelTestFixture();
    const requested: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = requestUrl(input);
      requested.push(url);
      if (url === `${baseUrl}model.json.gz`) {
        return responseFor(fixture.manifestTransportBytes);
      }
      const path = decodeURIComponent(
        new URL(url).pathname.split("/").slice(-2).join("/"),
      );
      const bytes =
        fixture.transports.get(path) ??
        fixture.transports.get(new URL(url).pathname.split("/").at(-1)!);
      if (bytes === undefined) return new Response(null, { status: 404 });
      return responseFor(bytes);
    });
    vi.stubGlobal("fetch", fetchMock);

    const first = await loadModelAssets(baseUrl);
    const cache = await globalThis.caches.open(first.source.cacheName);
    await cache.delete(
      new URL(
        fixture.manifest.files["decoder.onnx"].transport.path,
        baseUrl,
      ).href,
    );
    requested.length = 0;

    await loadModelAssets(baseUrl);
    expect(requested).toEqual([
      `${baseUrl}model.json.gz`,
      `${baseUrl}decoder.onnx.gz`,
    ]);
  });

  it("persists an interrupted file prefix and resumes it with an HTTP range request", async () => {
    const fixture = await createModelTestFixture();
    const [targetPath] = Object.keys(fixture.manifest.files);
    const targetTransportPath =
      fixture.manifest.files[targetPath].transport.path;
    const targetTransport = fixture.transports.get(targetTransportPath)!;
    const prefixBytes = Math.max(
      1,
      Math.floor(targetTransport.byteLength / 2),
    );
    const controller = new AbortController();
    let interruptTarget = true;
    let resumedRange: string | null = null;

    const fetchMock = vi.fn(
      async (
        input: RequestInfo | URL,
        init?: RequestInit,
      ): Promise<Response> => {
        const url = requestUrl(input);
        if (url === `${baseUrl}model.json.gz`) {
          return responseFor(fixture.manifestTransportBytes);
        }
        const path = decodeURIComponent(
          new URL(url).pathname.split("/").slice(-2).join("/"),
        );
        const bytes =
          fixture.transports.get(path) ??
          fixture.transports.get(
            new URL(url).pathname.split("/").at(-1)!,
          );
        if (bytes === undefined) {
          return new Response(null, { status: 404 });
        }
        const range = new Headers(init?.headers).get("range");
        if (range !== null) {
          resumedRange = range;
          const match = /^bytes=(\d+)-$/.exec(range);
          expect(match).not.toBeNull();
          const offset = Number(match![1]);
          const suffix = bytes.slice(offset);
          return new Response(suffix, {
            status: 206,
            headers: {
              "content-length": String(suffix.byteLength),
              "content-range":
                `bytes ${offset}-${bytes.byteLength - 1}/${bytes.byteLength}`,
            },
          });
        }
        if (
          interruptTarget &&
          path === targetTransportPath
        ) {
          interruptTarget = false;
          return new Response(
            new ReadableStream<Uint8Array>({
              start(streamController) {
                streamController.enqueue(
                  bytes.slice(0, prefixBytes),
                );
              },
            }),
            {
              headers: {
                "content-length": String(bytes.byteLength),
              },
            },
          );
        }
        return responseFor(bytes);
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const source = await fetchModelManifest(baseUrl);
    await expect(
      loadModelAssets(
        baseUrl,
        (progress) => {
          if (
            progress.stage === "downloading-model" &&
            progress.message === targetPath &&
            progress.completed === prefixBytes
          ) {
            controller.abort(
              new DOMException("Paused", "AbortError"),
            );
          }
        },
        controller.signal,
      ),
    ).rejects.toMatchObject({ name: "AbortError" });

    expect(await inspectModelCache(source)).toMatchObject({
      complete: false,
      cachedFileCount: 0,
      cachedTransportSizeBytes: prefixBytes,
    });

    const resumedProgress: number[] = [];
    await loadModelAssets(baseUrl, (progress) => {
      if (
        progress.stage === "downloading-model" &&
        progress.message === targetPath
      ) {
        resumedProgress.push(progress.completed);
      }
    });

    expect(resumedRange).toBe(`bytes=${prefixBytes}-`);
    expect(resumedProgress.length).toBeGreaterThan(0);
    expect(
      resumedProgress.every(
        (completed) => completed >= prefixBytes,
      ),
    ).toBe(true);
    expect(await inspectModelCache(source)).toMatchObject({
      complete: false,
      cachedFileCount: 5,
      cachedTransportSizeBytes: modelTransportSize(
        fixture.manifest,
      ),
    });
    const cache = await globalThis.caches.open(source.cacheName);
    expect(
      (await cache.keys()).some((request) =>
        request.url.includes("ardy-model-partial"),
      ),
    ).toBe(false);
  });

  it("rejects compressed and raw corruption without writing a completion marker", async () => {
    const fixture = await createModelTestFixture();
    const compressedCorruption = new Map(fixture.transports);
    const damaged = compressedCorruption.get("decoder.onnx.gz")!.slice();
    damaged[0] ^= 1;
    compressedCorruption.set("decoder.onnx.gz", damaged);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = requestUrl(input);
        if (url === `${baseUrl}model.json.gz`) {
          return responseFor(fixture.manifestTransportBytes);
        }
        const path = new URL(url).pathname.split("/").slice(-2).join("/");
        const bytes =
          compressedCorruption.get(path) ??
          compressedCorruption.get(new URL(url).pathname.split("/").at(-1)!);
        return bytes
          ? responseFor(bytes)
          : new Response(null, { status: 404 });
      }),
    );
    await expect(loadModelAssets(baseUrl)).rejects.toThrow(
      /Compressed SHA-256/,
    );

    const rawCorruption = new Map(fixture.transports);
    const damagedRaw = new TextEncoder().encode("fixture decodex");
    const damagedTransport = await gzipTestBytes(damagedRaw);
    rawCorruption.set("decoder.onnx.gz", damagedTransport);
    const rawHashManifest = structuredClone(fixture.manifest);
    rawHashManifest.files["decoder.onnx"].transport.sha256 =
      await sha256Hex(damagedTransport);
    rawHashManifest.files["decoder.onnx"].transport.size_bytes =
      damagedTransport.byteLength;
    const rawHashManifestBytes = new TextEncoder().encode(
      JSON.stringify(rawHashManifest),
    );
    const rawHashManifestTransport = await gzipTestBytes(
      rawHashManifestBytes,
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = requestUrl(input);
        if (url === `${baseUrl}model.json.gz`) {
          return responseFor(rawHashManifestTransport);
        }
        const path = new URL(url).pathname.split("/").slice(-2).join("/");
        const bytes =
          rawCorruption.get(path) ??
          rawCorruption.get(new URL(url).pathname.split("/").at(-1)!);
        return bytes
          ? responseFor(bytes)
          : new Response(null, { status: 404 });
      }),
    );
    const assets = await loadModelAssets(baseUrl);
    await expect(assets.read("decoder.onnx")).rejects.toThrow(
      /SHA-256 mismatch/,
    );
    expect((await inspectModelCache(assets.source)).complete).toBe(false);
  });

  it("loads from a completed cache when the manifest network is unavailable and clears versioned data", async () => {
    const fixture = await createModelTestFixture();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = requestUrl(input);
        if (url === `${baseUrl}model.json.gz`) {
          return responseFor(fixture.manifestTransportBytes);
        }
        const path = new URL(url).pathname.split("/").slice(-2).join("/");
        const bytes =
          fixture.transports.get(path) ??
          fixture.transports.get(new URL(url).pathname.split("/").at(-1)!);
        return bytes
          ? responseFor(bytes)
          : new Response(null, { status: 404 });
      }),
    );
    const first = await loadModelAssets(baseUrl);
    await Promise.all(
      Object.keys(first.manifest.files).map((path) => first.read(path)),
    );
    await markModelCacheComplete(first);

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("offline");
      }),
    );
    const cachedSource = await fetchModelManifest(baseUrl);
    expect(cachedSource.manifestSha256).toBe(
      first.source.manifestSha256,
    );
    const cached = await loadModelAssets(baseUrl);
    expect(new TextDecoder().decode(await cached.read("decoder.onnx"))).toBe(
      "fixture decoder",
    );

    await clearModelCache(cachedSource);
    expect(await inspectModelCache(cachedSource)).toMatchObject({
      cached: false,
      complete: false,
    });
  });

  it("works without Cache Storage while retaining transports until release", async () => {
    setCaches(undefined);
    const fixture = await createModelTestFixture();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = requestUrl(input);
        if (url === `${baseUrl}model.json.gz`) {
          return responseFor(fixture.manifestTransportBytes);
        }
        const path = new URL(url).pathname.split("/").slice(-2).join("/");
        const bytes =
          fixture.transports.get(path) ??
          fixture.transports.get(new URL(url).pathname.split("/").at(-1)!);
        return bytes
          ? responseFor(bytes)
          : new Response(null, { status: 404 });
      }),
    );

    const assets = await loadModelAssets(baseUrl);
    expect((await inspectModelCache(assets.source)).supported).toBe(false);
    expect(await assets.read("decoder.onnx")).toBeInstanceOf(Uint8Array);
    assets.release("decoder.onnx");
  });

  it("rejects unsafe model base URLs", () => {
    expect(normalizeModelBaseUrl("https://example.test/model")).toBe(
      "https://example.test/model/",
    );
    for (const value of [
      "ftp://example.test/model/",
      "https://user@example.test/model/",
      "https://example.test/model/?revision=main",
    ]) {
      expect(() => normalizeModelBaseUrl(value)).toThrow();
    }
  });
});

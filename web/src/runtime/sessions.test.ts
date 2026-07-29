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

import { REQUIRED_WEBGPU_FEATURE } from "./manifest";

const { createInferenceSession, ortEnvironment } = vi.hoisted(() => ({
  createInferenceSession: vi.fn(),
  ortEnvironment: {
    logLevel: "warning",
    wasm: {
      proxy: false,
      numThreads: 1,
      wasmPaths: "",
    },
    webgpu: {
      adapter: undefined as unknown,
    },
  },
}));

vi.mock("onnxruntime-web/webgpu", () => ({
  env: ortEnvironment,
  InferenceSession: {
    create: createInferenceSession,
  },
}));

const secureContextDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  "isSecureContext",
);
const gpuDescriptor = Object.getOwnPropertyDescriptor(
  globalThis.navigator,
  "gpu",
);

interface TestAdapter {
  readonly features: {
    has(feature: string): boolean;
  };
  readonly limits: object;
  requestDevice(): Promise<object>;
}

function createAdapter(hasShaderF16: boolean): TestAdapter {
  return {
    features: {
      has: vi.fn(
        (feature: string) =>
          feature === REQUIRED_WEBGPU_FEATURE && hasShaderF16,
      ),
    },
    limits: {},
    requestDevice: vi.fn(async () => ({})),
  };
}

function setSecureContext(value: boolean): void {
  Object.defineProperty(globalThis, "isSecureContext", {
    configurable: true,
    value,
  });
}

function setGpu(
  gpu:
    | {
        requestAdapter(): Promise<TestAdapter | null>;
      }
    | undefined,
): void {
  Object.defineProperty(globalThis.navigator, "gpu", {
    configurable: true,
    value: gpu,
  });
}

function restoreProperty(
  target: object,
  property: PropertyKey,
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor === undefined) {
    Reflect.deleteProperty(target, property);
    return;
  }
  Object.defineProperty(target, property, descriptor);
}

async function loadSessions() {
  vi.resetModules();
  return import("./sessions");
}

beforeEach(() => {
  createInferenceSession.mockReset();
  ortEnvironment.webgpu.adapter = undefined;
  setSecureContext(true);
  setGpu(undefined);
});

afterEach(() => {
  restoreProperty(
    globalThis,
    "isSecureContext",
    secureContextDescriptor,
  );
  restoreProperty(globalThis.navigator, "gpu", gpuDescriptor);
});

describe("WebGPU FP16 capability checks", () => {
  it("rejects an insecure context before requesting an adapter", async () => {
    const requestAdapter = vi.fn(async () => createAdapter(true));
    setSecureContext(false);
    setGpu({ requestAdapter });
    const { assertWebGpuAvailable } = await loadSessions();

    await expect(assertWebGpuAvailable()).rejects.toThrow(
      /HTTPS or localhost/,
    );
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it("rejects browsers without WebGPU", async () => {
    const { assertWebGpuAvailable } = await loadSessions();

    await expect(assertWebGpuAvailable()).rejects.toThrow(
      /WebGPU is required/,
    );
  });

  it("rejects a null WebGPU adapter", async () => {
    setGpu({ requestAdapter: vi.fn(async () => null) });
    const { assertWebGpuAvailable } = await loadSessions();

    await expect(assertWebGpuAvailable()).rejects.toThrow(
      /no compatible GPU adapter/,
    );
  });

  it("reports adapter initialization failures", async () => {
    const cause = new Error("adapter request failed");
    setGpu({
      requestAdapter: vi.fn(async () => {
        throw cause;
      }),
    });
    const { assertWebGpuAvailable } = await loadSessions();

    await expect(assertWebGpuAvailable()).rejects.toThrow(
      /adapter initialization failed/,
    );
  });

  it("rejects adapters without native FP16 shaders", async () => {
    const adapter = createAdapter(false);
    setGpu({ requestAdapter: vi.fn(async () => adapter) });
    const { assertWebGpuAvailable } = await loadSessions();

    await expect(assertWebGpuAvailable()).rejects.toThrow(/shader-f16/);
    expect(adapter.features.has).toHaveBeenCalledWith(
      REQUIRED_WEBGPU_FEATURE,
    );
    expect(ortEnvironment.webgpu.adapter).toBeUndefined();
  });

  it("retains and assigns the checked adapter to ONNX Runtime", async () => {
    const adapter = createAdapter(true);
    const requestAdapter = vi.fn(async () => adapter);
    setGpu({ requestAdapter });
    const { assertWebGpuAvailable } = await loadSessions();

    await assertWebGpuAvailable();
    expect(adapter.features.has).toHaveBeenCalledWith(
      REQUIRED_WEBGPU_FEATURE,
    );
    expect(ortEnvironment.webgpu.adapter).toBe(adapter);

    ortEnvironment.webgpu.adapter = undefined;
    await assertWebGpuAvailable();
    expect(requestAdapter).toHaveBeenCalledTimes(1);
    expect(ortEnvironment.webgpu.adapter).toBe(adapter);
  });

  it("assigns the checked adapter before creating any session", async () => {
    const adapter = createAdapter(true);
    setGpu({ requestAdapter: vi.fn(async () => adapter) });
    createInferenceSession.mockImplementation(async () => {
      expect(ortEnvironment.webgpu.adapter).toBe(adapter);
      return { release: vi.fn(async () => undefined) };
    });
    const { createRuntimeSessions } = await loadSessions();
    const graphs = {
      text_encoder: { model: "text_encoder.onnx" },
      denoiser: { model: "denoiser.onnx" },
      decoder: { model: "decoder.onnx" },
    };
    const pack = {
      manifest: { graphs },
      read: vi.fn(async () => new Uint8Array([0])),
      release: vi.fn(),
    } as unknown as Parameters<typeof createRuntimeSessions>[0];

    await createRuntimeSessions(pack);

    expect(createInferenceSession).toHaveBeenCalledTimes(3);
  });
});

describe("runtime session asset lifetime", () => {
  it("creates graphs from smallest to largest and releases each asset immediately", async () => {
    const adapter = createAdapter(true);
    setGpu({ requestAdapter: vi.fn(async () => adapter) });
    const events: string[] = [];
    const sessionByModel = new Map<number, { release: ReturnType<typeof vi.fn> }>();
    createInferenceSession.mockImplementation(
      async (model: Uint8Array) => {
        const modelId = model[0];
        events.push(`create:${modelId}`);
        const session = { release: vi.fn(async () => undefined) };
        sessionByModel.set(modelId, session);
        return session;
      },
    );
    const { createRuntimeSessions } = await loadSessions();
    const graphs = {
      text_encoder: { model: "text.onnx" },
      denoiser: { model: "denoiser.onnx" },
      decoder: { model: "decoder.onnx" },
    };
    const modelIds: Record<string, number> = {
      "text.onnx": 1,
      "decoder.onnx": 2,
      "denoiser.onnx": 3,
    };
    const pack = {
      manifest: { graphs },
      read: vi.fn(async (path: string) => {
        events.push(`read:${path}`);
        return new Uint8Array([modelIds[path]]);
      }),
      release: vi.fn((path: string) => {
        events.push(`release:${path}`);
      }),
    } as unknown as Parameters<typeof createRuntimeSessions>[0];
    const progress: string[] = [];

    const sessions = await createRuntimeSessions(
      pack,
      (completed, total, message) =>
        progress.push(`${completed}/${total}:${message}`),
    );

    expect(events).toEqual([
      "read:decoder.onnx",
      "create:2",
      "release:decoder.onnx",
      "read:text.onnx",
      "create:1",
      "release:text.onnx",
      "read:denoiser.onnx",
      "create:3",
      "release:denoiser.onnx",
    ]);
    expect(progress).toEqual([
      "1/3:decoder.onnx",
      "2/3:text_encoder.onnx",
      "3/3:denoiser.onnx",
    ]);
    expect(sessions).toEqual({
      textEncoder: sessionByModel.get(1),
      denoiser: sessionByModel.get(3),
      decoder: sessionByModel.get(2),
    });
  });

  it("releases failing graph assets and disposes sessions already created", async () => {
    const adapter = createAdapter(true);
    setGpu({ requestAdapter: vi.fn(async () => adapter) });
    const decoderSession = { release: vi.fn(async () => undefined) };
    const textSession = { release: vi.fn(async () => undefined) };
    createInferenceSession
      .mockResolvedValueOnce(decoderSession)
      .mockResolvedValueOnce(textSession)
      .mockRejectedValueOnce(new Error("denoiser initialization failed"));
    const { createRuntimeSessions } = await loadSessions();
    const graphs = {
      text_encoder: {
        model: "text.onnx",
        external_data: [{ path: "text.data", file: "text.bin" }],
      },
      denoiser: {
        model: "denoiser.onnx",
        external_data: [{ path: "denoiser.data", file: "denoiser.bin" }],
      },
      decoder: {
        model: "decoder.onnx",
        external_data: [{ path: "decoder.data", file: "decoder.bin" }],
      },
    };
    const releaseAsset = vi.fn();
    const pack = {
      manifest: { graphs },
      read: vi.fn(async () => new Uint8Array([0])),
      release: releaseAsset,
    } as unknown as Parameters<typeof createRuntimeSessions>[0];

    await expect(createRuntimeSessions(pack)).rejects.toThrow(
      /denoiser initialization failed/,
    );

    expect(releaseAsset.mock.calls.map(([path]) => path)).toEqual([
      "decoder.onnx",
      "decoder.bin",
      "text.onnx",
      "text.bin",
      "denoiser.onnx",
      "denoiser.bin",
    ]);
    expect(textSession.release).toHaveBeenCalledTimes(1);
    expect(decoderSession.release).toHaveBeenCalledTimes(1);
  });
});

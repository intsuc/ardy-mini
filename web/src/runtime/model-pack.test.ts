// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { sha256Hex } from "./hash";
import {
  MODEL_PACK_FORMAT,
  MODEL_PACK_SCHEMA_VERSION,
  type BrowserModelPackManifest,
  validateModelPackManifest,
} from "./manifest";
import { loadModelPackFromTarGzip } from "./model-pack";

const encoder = new TextEncoder();
const TAR_BLOCK_SIZE = 512;

function writeTarString(
  header: Uint8Array,
  offset: number,
  length: number,
  value: string,
): void {
  const bytes = encoder.encode(value);
  if (bytes.byteLength > length) {
    throw new Error(`Tar fixture value is too long: ${value}`);
  }
  header.set(bytes, offset);
}

function writeTarOctal(
  header: Uint8Array,
  offset: number,
  length: number,
  value: number,
): void {
  const text = value.toString(8).padStart(length - 1, "0");
  writeTarString(header, offset, length, `${text}\0`);
}

function tarHeader(path: string, size: number): Uint8Array {
  const header = new Uint8Array(TAR_BLOCK_SIZE);
  writeTarString(header, 0, 100, path);
  writeTarOctal(header, 100, 8, 0o644);
  writeTarOctal(header, 108, 8, 0);
  writeTarOctal(header, 116, 8, 0);
  writeTarOctal(header, 124, 12, size);
  writeTarOctal(header, 136, 12, 0);
  header.fill(0x20, 148, 156);
  header[156] = 0x30;
  writeTarString(header, 257, 6, "ustar\0");
  writeTarString(header, 263, 2, "00");
  writeTarString(header, 265, 32, "root");
  writeTarString(header, 297, 32, "root");
  const checksum = header.reduce((total, value) => total + value, 0);
  writeTarString(
    header,
    148,
    8,
    `${checksum.toString(8).padStart(6, "0")}\0 `,
  );
  return header;
}

function createTar(
  entries: readonly (readonly [string, Uint8Array])[],
  options: { endBlocks?: number } = {},
): Uint8Array {
  const parts: Uint8Array[] = [];
  let byteLength = 0;
  for (const [path, bytes] of entries) {
    const header = tarHeader(path, bytes.byteLength);
    const padding = new Uint8Array(
      (TAR_BLOCK_SIZE - (bytes.byteLength % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE,
    );
    parts.push(header, bytes, padding);
    byteLength += header.byteLength + bytes.byteLength + padding.byteLength;
  }
  const end = new Uint8Array(
    (options.endBlocks ?? 2) * TAR_BLOCK_SIZE,
  );
  parts.push(end);
  byteLength += end.byteLength;
  const tar = new Uint8Array(byteLength);
  let offset = 0;
  for (const part of parts) {
    tar.set(part, offset);
    offset += part.byteLength;
  }
  return tar;
}

async function gzipTar(
  tar: Uint8Array,
  name = "fixture.tar.gz",
): Promise<File> {
  const input = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(tar);
      controller.close();
    },
  });
  const compressor = new CompressionStream("gzip");
  const stream = input.pipeThrough({
    readable: compressor.readable as ReadableStream<Uint8Array>,
    writable: compressor.writable as WritableStream<Uint8Array>,
  });
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    byteLength += value.byteLength;
  }
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return streamFile(bytes, name, "application/gzip");
}

function streamFile(
  bytes: Uint8Array,
  name: string,
  type = "",
): File {
  const payload = bytes.slice();
  const file = new File([payload.buffer], name, {
    type,
  });
  Object.defineProperty(file, "stream", {
    configurable: true,
    value: () =>
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(bytes);
          controller.close();
        },
      }),
  });
  return file;
}

async function archiveFor(
  entries: ReadonlyMap<string, Uint8Array>,
): Promise<File> {
  return gzipTar(createTar(packEntries(entries)));
}

function packEntries(
  entries: ReadonlyMap<string, Uint8Array>,
): Array<readonly [string, Uint8Array]> {
  const manifest = entries.get("manifest.json");
  const assets = [...entries]
    .filter(([path]) => path !== "manifest.json")
    .sort(([left], [right]) => left.localeCompare(right));
  return manifest === undefined
    ? assets
    : [["manifest.json", manifest], ...assets];
}

async function fixture(): Promise<{
  entries: Map<string, Uint8Array>;
  manifest: BrowserModelPackManifest;
}> {
  const assets = new Map<string, Uint8Array>([
    ["tokenizer/tokenizer.json", encoder.encode("{}")],
    ["tokenizer/tokenizer_config.json", encoder.encode("{}")],
    ["graphs.onnx", encoder.encode("fake onnx")],
  ]);
  const files: BrowserModelPackManifest["files"] = {};
  for (const [path, bytes] of assets) {
    files[path] = {
      size_bytes: bytes.byteLength,
      sha256: await sha256Hex(bytes),
    };
  }
  const alphas = [0.99, 0.95, 0.88, 0.78, 0.66, 0.53, 0.4, 0.28, 0.17, 0.08];
  const manifest: BrowserModelPackManifest = {
    format: MODEL_PACK_FORMAT,
    schema_version: MODEL_PACK_SCHEMA_VERSION,
    model: { id: "fixture", variant: "test" },
    files,
    tokenizer: { directory: "tokenizer", max_length: 128 },
    graphs: {
      text_encoder: {
        model: "graphs.onnx",
        inputs: {
          inputIds: "input_ids",
          attentionMask: "attention_mask",
          tokenTypeIds: "token_type_ids",
        },
        outputs: { textConditions: "text_conditions" },
      },
      denoiser: {
        model: "graphs.onnx",
        inputs: {
          cfgWeight: "cfg_weight",
          x: "x",
          historyLength: "history_len",
          generationLength: "generation_len",
          historyMask: "history_mask",
          generationMask: "generation_mask",
          historyTokenMask: "history_token_mask",
          generationTokenMask: "generation_token_mask",
          textConditions: "text_conditions",
          timestep: "timestep",
          firstHeadingAngle: "first_heading_angle",
        },
        outputs: { predX0: "pred_x0" },
      },
      decoder: {
        model: "graphs.onnx",
        inputs: {
          hybridTokens: "hybrid_tokens",
          motionPadMask: "motion_pad_mask",
          globalTranslation: "global_translation",
        },
        outputs: {
          normalizedMotion: "normalized_motion",
          posedJoints: "posed_joints",
          localRotations: "local_rotations",
          globalRotations: "global_rotations",
          rootPositions: "root_positions",
          footContacts: "foot_contacts",
          globalRootHeading: "global_root_heading",
        },
      },
    },
    dimensions: {
      fps: 20,
      num_frames_per_token: 4,
      max_tokens: 20,
      max_frames: 80,
      generation_tokens: 10,
      generation_frames: 40,
      history_tokens: 10,
      history_frames: 40,
      root_features_per_frame: 5,
      nframe_root_dim: 20,
      latent_dim: 128,
      hybrid_dim: 148,
      motion_dim: 330,
      body_dim: 325,
      text_condition_dim: 2048,
      num_joints: 27,
    },
    generation: {
      min_frames: 40,
      max_frames: 200,
      default_cfg_weight: 2,
      denoising_steps: 10,
    },
    diffusion: {
      timesteps: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
      alphas_cumprod: alphas,
      alphas_cumprod_prev: [1, ...alphas.slice(0, -1)],
    },
    recenter: {
      root_mean: [0, 0, 0, 0, 0],
      root_std: [1, 1, 1, 1, 1],
      position_indices: [0, 1, 2],
      heading_indices: [3, 4],
    },
    latent_quantization: {
      levels: new Array(128).fill(4),
      mean: new Array(128).fill(0),
      std: new Array(128).fill(1),
    },
    runtime: {
      contract_revision: 3,
      text_only: true,
    },
    notices: ["fixture only"],
    license_notices: [
      { component: "fixture", license: "MIT", notice: "fixture only" },
    ],
  };
  const entries = new Map(assets);
  entries.set("manifest.json", encoder.encode(JSON.stringify(manifest)));
  return { entries, manifest };
}

describe("model-pack validation", () => {
  it("decompresses, parses, and hashes the canonical tar.gz pack", async () => {
    const { entries } = await fixture();
    const progress: string[] = [];
    const pack = await loadModelPackFromTarGzip(await archiveFor(entries), (event) =>
      progress.push(`${event.stage}:${event.completed}/${event.total}`),
    );
    expect(pack.manifest.model.id).toBe("fixture");
    expect(new TextDecoder().decode(await pack.read("graphs.onnx"))).toBe(
      "fake onnx",
    );
    expect(progress.some((value) => value.startsWith("hashing-pack:3/3"))).toBe(
      true,
    );
  });

  it("rejects tampered bytes and ascending sampler timesteps", async () => {
    const { entries, manifest } = await fixture();
    entries.set("graphs.onnx", encoder.encode("tampered!"));
    await expect(
      loadModelPackFromTarGzip(await archiveFor(entries)),
    ).rejects.toThrow(
      /SHA-256 mismatch/,
    );
    entries.set("graphs.onnx", encoder.encode("wrong size"));
    await expect(
      loadModelPackFromTarGzip(await archiveFor(entries)),
    ).rejects.toThrow(/Size mismatch/);
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        diffusion: {
          ...manifest.diffusion,
          timesteps: [...manifest.diffusion.timesteps].reverse(),
        },
      }),
    ).toThrow(/strictly decreasing/);
  });

  it("rejects invalid gzip, unsafe paths, and duplicate paths", async () => {
    await expect(
      loadModelPackFromTarGzip(
        streamFile(encoder.encode("not gzip"), "invalid.tar.gz"),
      ),
    ).rejects.toThrow(/decompress/i);

    const { entries } = await fixture();
    const manifest = entries.get("manifest.json");
    if (manifest === undefined) throw new Error("Missing fixture manifest");
    await expect(
      loadModelPackFromTarGzip(
        await gzipTar(createTar([["../manifest.json", manifest]])),
      ),
    ).rejects.toThrow(/Unsafe tar entry path/);
    await expect(
      loadModelPackFromTarGzip(
        await gzipTar(createTar([["legacy-root/manifest.json", manifest]])),
      ),
    ).rejects.toThrow(/first tar entry must be manifest\.json/i);
    await expect(
      loadModelPackFromTarGzip(
        await gzipTar(
          createTar([
            ["manifest.json", manifest],
            ["manifest.json", manifest],
          ]),
        ),
      ),
    ).rejects.toThrow(/Duplicate tar entry path/);
  });

  it("rejects bad checksums, non-regular entries, and truncated tar data", async () => {
    const { entries } = await fixture();
    const validTar = createTar(packEntries(entries));

    const badChecksum = validTar.slice();
    badChecksum[0] ^= 1;
    await expect(
      loadModelPackFromTarGzip(await gzipTar(badChecksum)),
    ).rejects.toThrow(/checksum mismatch/);

    const nonRegular = validTar.slice();
    nonRegular[156] = 0x32;
    nonRegular.fill(0x20, 148, 156);
    const checksum = nonRegular
      .subarray(0, TAR_BLOCK_SIZE)
      .reduce((total, value) => total + value, 0);
    writeTarString(
      nonRegular,
      148,
      8,
      `${checksum.toString(8).padStart(6, "0")}\0 `,
    );
    await expect(
      loadModelPackFromTarGzip(await gzipTar(nonRegular)),
    ).rejects.toThrow(/regular files/);

    await expect(
      loadModelPackFromTarGzip(
        await gzipTar(createTar(packEntries(entries), { endBlocks: 1 })),
      ),
    ).rejects.toThrow(/Truncated tar archive/);

    const bodyTruncated = validTar.subarray(
      0,
      validTar.byteLength - 2 * TAR_BLOCK_SIZE - 511,
    );
    await expect(
      loadModelPackFromTarGzip(await gzipTar(bodyTruncated)),
    ).rejects.toThrow(/Truncated tar archive/);

    const trailingData = new Uint8Array(validTar.byteLength + 1);
    trailingData.set(validTar);
    trailingData[trailingData.length - 1] = 1;
    await expect(
      loadModelPackFromTarGzip(await gzipTar(trailingData)),
    ).rejects.toThrow(/data after its end marker/);
  });

  it("requires a root manifest and rejects undeclared archive entries", async () => {
    const { entries } = await fixture();
    const withoutManifest = new Map(entries);
    withoutManifest.delete("manifest.json");
    await expect(
      loadModelPackFromTarGzip(await archiveFor(withoutManifest)),
    ).rejects.toThrow(/manifest\.json/);

    entries.set("unexpected.txt", encoder.encode("unexpected"));
    await expect(
      loadModelPackFromTarGzip(await archiveFor(entries)),
    ).rejects.toThrow(/undeclared file/);
  });

  it("rejects dimensions outside the fixed Core40 contract", async () => {
    const { manifest } = await fixture();
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        dimensions: {
          ...manifest.dimensions,
          num_joints: 26,
        },
      }),
    ).toThrow(/browser Core40 runtime/);
  });

  it("requires every revision-3 structured motion output", async () => {
    const { manifest } = await fixture();
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        graphs: {
          ...manifest.graphs,
          decoder: {
            ...manifest.graphs.decoder,
            outputs: {
              normalizedMotion: "normalized_motion",
              posedJoints: "posed_joints",
            },
          },
        },
      }),
    ).toThrow(/localRotations/);

    const current = {
      ...manifest,
      capabilities: {
        text_conditioning: true,
        detailed_motion_outputs: true,
      },
      runtime: {
        contract_revision: 3 as const,
        text_only: true as const,
        onnx_opset: 18,
        batch_size: 1,
        detailed_motion_outputs: true,
      },
    };
    expect(validateModelPackManifest(current).runtime.contract_revision).toBe(3);
    expect(
      validateModelPackManifest(current).graphs.decoder.outputs.footContacts,
    ).toBe("foot_contacts");
  });

  it("rejects extra graphs and unreferenced model-pack assets", async () => {
    const { manifest } = await fixture();
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        graphs: {
          ...manifest.graphs,
          constraint_denoiser: manifest.graphs.denoiser,
        },
      }),
    ).toThrow(/graphs must contain exactly/);

    expect(() =>
      validateModelPackManifest({
        ...manifest,
        graphs: {
          ...manifest.graphs,
          denoiser: {
            ...manifest.graphs.denoiser,
            inputs: {
              ...manifest.graphs.denoiser.inputs,
              motionMask: "motion_mask",
            },
          },
        },
      }),
    ).toThrow(/graphs\.denoiser\.inputs must contain exactly/);

    expect(() =>
      validateModelPackManifest({
        ...manifest,
        files: {
          ...manifest.files,
          "denoiser_constraints.onnx": {
            size_bytes: 1,
            sha256: "0".repeat(64),
          },
        },
      }),
    ).toThrow(/unreferenced asset/);
  });

  it("rejects every legacy constraint-graph manifest field", async () => {
    const { manifest } = await fixture();
    const cases: Array<readonly [string, BrowserModelPackManifest]> = [
      [
        "dimensions.constraint_max_tokens",
        {
          ...manifest,
          dimensions: {
            ...manifest.dimensions,
            constraint_max_tokens: 50,
          },
        } as BrowserModelPackManifest,
      ],
      [
        "dimensions.constraint_max_frames",
        {
          ...manifest,
          dimensions: {
            ...manifest.dimensions,
            constraint_max_frames: 200,
          },
        } as BrowserModelPackManifest,
      ],
      [
        "generation.default_text_cfg_weight",
        {
          ...manifest,
          generation: {
            ...manifest.generation,
            default_text_cfg_weight: 2,
          },
        } as BrowserModelPackManifest,
      ],
      [
        "generation.default_constraint_cfg_weight",
        {
          ...manifest,
          generation: {
            ...manifest.generation,
            default_constraint_cfg_weight: 2,
          },
        } as BrowserModelPackManifest,
      ],
      ...[
        "constraints_supported",
        "separated_cfg",
        "future_constraints_supported",
      ].map(
        (field) =>
          [
            `runtime.${field}`,
            {
              ...manifest,
              runtime: { ...manifest.runtime, [field]: true },
            } as BrowserModelPackManifest,
          ] as const,
      ),
      ...[
        "kinematic_constraints",
        "future_constraints",
        "separated_classifier_free_guidance",
      ].map(
        (field) =>
          [
            `capabilities.${field}`,
            {
              ...manifest,
              capabilities: { [field]: true },
            } as BrowserModelPackManifest,
          ] as const,
      ),
    ];

    for (const [field, legacyManifest] of cases) {
      expect(
        () => validateModelPackManifest(legacyManifest),
        field,
      ).toThrow(field);
    }
  });

  it("rejects manifests from an older runtime contract", async () => {
    const { manifest } = await fixture();
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        runtime: { contract_revision: 2, text_only: true },
      }),
    ).toThrow(/contract_revision must be 3/);
  });
});

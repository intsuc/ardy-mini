// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { sha256Hex } from "./hash";
import {
  GRAPH_PRECISION_CONTRACT,
  MIXED_PRECISION_FORMAT,
  MIXED_PRECISION_POLICY_VERSION,
  MIXED_PRECISION_PUBLIC_IO_DTYPE,
  MODEL_PACK_FORMAT,
  MODEL_PACK_SCHEMA_VERSION,
  REQUIRED_WEBGPU_FEATURE,
  type BrowserGraphPrecisionSummary,
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
    ["text_encoder.onnx", encoder.encode("fake text encoder onnx")],
    ["denoiser.onnx", encoder.encode("fake denoiser onnx")],
    ["decoder.onnx", encoder.encode("fake decoder onnx")],
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
        model: "text_encoder.onnx",
        inputs: {
          inputIds: "input_ids",
          attentionMask: "attention_mask",
          tokenTypeIds: "token_type_ids",
        },
        outputs: { textConditions: "text_conditions" },
      },
      denoiser: {
        model: "denoiser.onnx",
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
        model: "decoder.onnx",
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
    precision: ((): BrowserModelPackManifest["precision"] => {
      const summary = <
        GraphName extends keyof BrowserModelPackManifest["graphs"],
      >(
        graphName: GraphName,
        modelPath: string,
      ): BrowserGraphPrecisionSummary<GraphName> => {
        const outputSize = files[modelPath].size_bytes;
        const isIdentity =
          GRAPH_PRECISION_CONTRACT[graphName].conversion_mode ===
          "fp32-identity";
        const sourceSize = isIdentity ? outputSize : outputSize * 2;
        const sourceInitializers = {
          count_by_dtype: { float: 1 },
          bytes_by_dtype: { float: 4 },
          total_count: 1,
          total_bytes: 4,
        };
        const outputInitializers = isIdentity
          ? sourceInitializers
          : {
              count_by_dtype: { float16: 1 },
              bytes_by_dtype: { float16: 2 },
              total_count: 1,
              total_bytes: 2,
            };
        return {
          schema_version: 1,
          graph_name: graphName,
          policy_id: GRAPH_PRECISION_CONTRACT[graphName].policy_id,
          conversion_mode:
            GRAPH_PRECISION_CONTRACT[graphName].conversion_mode,
          source_sha256: isIdentity
            ? files[modelPath].sha256
            : "f".repeat(64),
          output_sha256: files[modelPath].sha256,
          source_size_bytes: sourceSize,
          output_size_bytes: outputSize,
          size_reduction_bytes: sourceSize - outputSize,
          size_reduction_fraction: (sourceSize - outputSize) / sourceSize,
          source_initializers: sourceInitializers,
          output_initializers: outputInitializers,
        };
      };
      const precisionGraphs = {
        text_encoder: summary("text_encoder", "text_encoder.onnx"),
        denoiser: summary("denoiser", "denoiser.onnx"),
        decoder: summary("decoder", "decoder.onnx"),
      };
      const sourceBytes = Object.values(precisionGraphs).reduce(
        (total, graph) => total + graph.source_size_bytes,
        0,
      );
      const mixedBytes = Object.values(precisionGraphs).reduce(
        (total, graph) => total + graph.output_size_bytes,
        0,
      );
      return {
        format: MIXED_PRECISION_FORMAT,
        policy_version: MIXED_PRECISION_POLICY_VERSION,
        public_io_dtype: MIXED_PRECISION_PUBLIC_IO_DTYPE,
        required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
        source_onnx_bytes: sourceBytes,
        mixed_onnx_bytes: mixedBytes,
        saved_onnx_bytes: sourceBytes - mixedBytes,
        saved_onnx_fraction: (sourceBytes - mixedBytes) / sourceBytes,
        toolchain: {
          torch: "fixture-torch",
          onnx: "fixture-onnx",
          onnxruntime: "fixture-onnxruntime",
        },
        graphs: precisionGraphs,
      };
    })(),
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
      required_webgpu_features: [REQUIRED_WEBGPU_FEATURE],
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
    expect(new TextDecoder().decode(await pack.read("denoiser.onnx"))).toBe(
      "fake denoiser onnx",
    );
    expect(progress.some((value) => value.startsWith("hashing-pack:5/5"))).toBe(
      true,
    );
  });

  it("rejects tampered bytes and ascending sampler timesteps", async () => {
    const { entries, manifest } = await fixture();
    const tampered = entries.get("denoiser.onnx")!.slice();
    tampered[0] ^= 1;
    entries.set("denoiser.onnx", tampered);
    await expect(
      loadModelPackFromTarGzip(await archiveFor(entries)),
    ).rejects.toThrow(
      /SHA-256 mismatch/,
    );
    entries.set("denoiser.onnx", encoder.encode("wrong size"));
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

  it("requires the current mixed-FP16 precision contract and toolchain", async () => {
    const { manifest } = await fixture();
    expect(validateModelPackManifest(manifest).precision.format).toBe(
      "mixed-fp16",
    );

    const { precision: _precision, ...withoutPrecision } = manifest;
    expect(() => validateModelPackManifest(withoutPrecision)).toThrow(
      /precision must be an object/,
    );

    const invalidPrecisionValues: Array<
      readonly [string, Record<string, unknown>]
    > = [
      ["precision.format", { ...manifest.precision, format: "fp32" }],
      [
        "precision.policy_version",
        { ...manifest.precision, policy_version: 999 },
      ],
      [
        "precision.public_io_dtype",
        { ...manifest.precision, public_io_dtype: "float16" },
      ],
      [
        "precision.toolchain.onnxruntime",
        {
          ...manifest.precision,
          toolchain: {
            ...manifest.precision.toolchain,
            onnxruntime: "",
          },
        },
      ],
      [
        "precision.graphs",
        {
          ...manifest.precision,
          graphs: {
            text_encoder: manifest.precision.graphs.text_encoder,
            denoiser: manifest.precision.graphs.denoiser,
          },
        },
      ],
      [
        "precision.graphs.text_encoder.graph_name",
        {
          ...manifest.precision,
          graphs: {
            ...manifest.precision.graphs,
            text_encoder: {
              ...manifest.precision.graphs.text_encoder,
              graph_name: "denoiser",
            },
          },
        },
      ],
      [
        "precision.graphs.text_encoder.policy_id",
        {
          ...manifest.precision,
          graphs: {
            ...manifest.precision.graphs,
            text_encoder: {
              ...manifest.precision.graphs.text_encoder,
              policy_id: "unreviewed-policy",
            },
          },
        },
      ],
      [
        "precision.graphs.decoder.conversion_mode",
        {
          ...manifest.precision,
          graphs: {
            ...manifest.precision.graphs,
            decoder: {
              ...manifest.precision.graphs.decoder,
              conversion_mode: "fp32-identity",
            },
          },
        },
      ],
    ];
    for (const [path, precision] of invalidPrecisionValues) {
      expect(
        () => validateModelPackManifest({ ...manifest, precision }),
        path,
      ).toThrow(path);
    }
  });

  it("enforces FP32 denoiser identity and initializer precision", async () => {
    const { manifest } = await fixture();
    const precision = manifest.precision;
    const denoiser = precision.graphs.denoiser;
    const decoder = precision.graphs.decoder;
    const textEncoder = precision.graphs.text_encoder;

    const invalidSummaries: Array<
      readonly [string, keyof typeof precision.graphs, Record<string, unknown>]
    > = [
      [
        "fp32-identity conversion must have zero byte reduction",
        "denoiser",
        {
          ...denoiser,
          source_size_bytes: denoiser.source_size_bytes + 1,
          size_reduction_bytes: 1,
          size_reduction_fraction: 1 / (denoiser.source_size_bytes + 1),
        },
      ],
      [
        "source_sha256 must equal output_sha256",
        "denoiser",
        {
          ...denoiser,
          source_sha256: "0".repeat(64),
        },
      ],
      [
        "output_initializers must not contain float16",
        "denoiser",
        {
          ...denoiser,
          output_initializers: {
            count_by_dtype: { float16: 1 },
            bytes_by_dtype: { float16: 2 },
            total_count: 1,
            total_bytes: 2,
          },
        },
      ],
      [
        "output_initializers must not contain bfloat16",
        "decoder",
        {
          ...decoder,
          output_initializers: {
            count_by_dtype: { bfloat16: 1 },
            bytes_by_dtype: { bfloat16: 2 },
            total_count: 1,
            total_bytes: 2,
          },
        },
      ],
      [
        "source_initializers must not contain float16",
        "text_encoder",
        {
          ...textEncoder,
          source_initializers: {
            count_by_dtype: { float16: 1 },
            bytes_by_dtype: { float16: 2 },
            total_count: 1,
            total_bytes: 2,
          },
        },
      ],
      [
        "total_bytes must equal the sum",
        "decoder",
        {
          ...decoder,
          output_initializers: {
            ...decoder.output_initializers,
            total_bytes: decoder.output_initializers.total_bytes + 1,
          },
        },
      ],
      [
        "output_sha256 must match",
        "decoder",
        {
          ...decoder,
          output_sha256: "0".repeat(64),
        },
      ],
    ];

    for (const [message, graphName, summary] of invalidSummaries) {
      expect(
        () =>
          validateModelPackManifest({
            ...manifest,
            precision: {
              ...precision,
              graphs: {
                ...precision.graphs,
                [graphName]: summary,
              },
            },
          }),
        `${graphName}: ${message}`,
      ).toThrow(message);
    }
  });

  it("rejects inconsistent mixed-FP16 byte accounting", async () => {
    const { manifest } = await fixture();
    const precision = manifest.precision;
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: {
          ...precision,
          saved_onnx_bytes: precision.saved_onnx_bytes + 1,
        },
      }),
    ).toThrow(/source_onnx_bytes - mixed_onnx_bytes/);
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: { ...precision, saved_onnx_fraction: 1.1 },
      }),
    ).toThrow(/between 0 and 1/);
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: { ...precision, saved_onnx_fraction: 0.4 },
      }),
    ).toThrow(/does not match the declared byte reduction/);

    const decoder = precision.graphs.decoder;
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: {
          ...precision,
          graphs: {
            ...precision.graphs,
            decoder: {
              ...decoder,
              size_reduction_bytes: decoder.size_reduction_bytes + 1,
            },
          },
        },
      }),
    ).toThrow(/size_reduction_bytes must equal/);

    const smallerOutput = decoder.output_size_bytes - 1;
    const largerFileReduction = decoder.source_size_bytes - smallerOutput;
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: {
          ...precision,
          graphs: {
            ...precision.graphs,
            decoder: {
              ...decoder,
              output_size_bytes: smallerOutput,
              size_reduction_bytes: largerFileReduction,
              size_reduction_fraction:
                largerFileReduction / decoder.source_size_bytes,
            },
          },
        },
      }),
    ).toThrow(/must match files\.decoder\.onnx\.size_bytes/);

    const largerSource = decoder.source_size_bytes + 1;
    const largerReduction = largerSource - decoder.output_size_bytes;
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        precision: {
          ...precision,
          graphs: {
            ...precision.graphs,
            decoder: {
              ...decoder,
              source_size_bytes: largerSource,
              size_reduction_bytes: largerReduction,
              size_reduction_fraction: largerReduction / largerSource,
            },
          },
        },
      }),
    ).toThrow(/precision graph source total/);

    const largerOutput = decoder.output_size_bytes + 1;
    const smallerReduction = decoder.source_size_bytes - largerOutput;
    expect(() =>
      validateModelPackManifest({
        ...manifest,
        files: {
          ...manifest.files,
          "decoder.onnx": {
            ...manifest.files["decoder.onnx"],
            size_bytes: largerOutput,
          },
        },
        precision: {
          ...precision,
          graphs: {
            ...precision.graphs,
            decoder: {
              ...decoder,
              output_size_bytes: largerOutput,
              size_reduction_bytes: smallerReduction,
              size_reduction_fraction:
                smallerReduction / decoder.source_size_bytes,
            },
          },
        },
      }),
    ).toThrow(/precision graph output total/);
  });

  it("rejects missing, duplicate, and unknown required WebGPU features", async () => {
    const { manifest } = await fixture();
    for (const features of [
      undefined,
      [],
      ["shader-f16", "shader-f16"],
      ["unknown-feature"],
    ]) {
      expect(
        () =>
          validateModelPackManifest({
            ...manifest,
            runtime: {
              ...manifest.runtime,
              required_webgpu_features: features,
            },
          }),
        JSON.stringify(features),
      ).toThrow(/runtime\.required_webgpu_features/);
    }

    for (const features of [
      undefined,
      [],
      ["shader-f16", "shader-f16"],
      ["unknown-feature"],
    ]) {
      expect(
        () =>
          validateModelPackManifest({
            ...manifest,
            precision: {
              ...manifest.precision,
              required_webgpu_features: features,
            },
          }),
        JSON.stringify(features),
      ).toThrow(/precision\.required_webgpu_features/);
    }
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
        ...manifest.runtime,
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

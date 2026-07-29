// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const MODEL_PACK_FORMAT = "ardy-browser-model-pack";
export const MODEL_PACK_SCHEMA_VERSION = 2;
export const MODEL_PACK_MANIFEST_FILE = "manifest.json";
export const MIXED_PRECISION_FORMAT = "mixed-fp16";
export const MIXED_PRECISION_POLICY_VERSION = 3;
export const MIXED_PRECISION_PUBLIC_IO_DTYPE = "float32";
export const REQUIRED_WEBGPU_FEATURE = "shader-f16";

export type Sha256Hex = string;

export interface ManifestFile {
  sha256: Sha256Hex;
  size_bytes: number;
  media_type?: string;
}

export interface ExternalDataSpec {
  /** Path embedded in the ONNX protobuf. This is not necessarily the asset path. */
  path: string;
  /** Relative path of the asset in the model pack. */
  file: string;
}

export interface GraphSpec<
  Inputs extends Record<string, string> = Record<string, string>,
  Outputs extends Record<string, string | undefined> = Record<string, string>,
> {
  model: string;
  external_data?: ExternalDataSpec[];
  inputs: Inputs;
  outputs: Outputs;
}

export interface TextEncoderInputs extends Record<string, string> {
  inputIds: string;
  attentionMask: string;
  tokenTypeIds: string;
}

export interface TextEncoderOutputs extends Record<string, string> {
  textConditions: string;
}

export interface DenoiserInputs extends Record<string, string> {
  cfgWeight: string;
  x: string;
  historyLength: string;
  generationLength: string;
  historyMask: string;
  generationMask: string;
  historyTokenMask: string;
  generationTokenMask: string;
  textConditions: string;
  timestep: string;
  firstHeadingAngle: string;
}

export interface DenoiserOutputs extends Record<string, string> {
  predX0: string;
}

export interface DecoderInputs extends Record<string, string> {
  hybridTokens: string;
  motionPadMask: string;
  globalTranslation: string;
}

export interface DecoderOutputs extends Record<string, string> {
  normalizedMotion: string;
  posedJoints: string;
  localRotations: string;
  globalRotations: string;
  rootPositions: string;
  footContacts: string;
  globalRootHeading: string;
}

export interface BrowserGraphSpecs {
  text_encoder: GraphSpec<TextEncoderInputs, TextEncoderOutputs>;
  denoiser: GraphSpec<DenoiserInputs, DenoiserOutputs>;
  decoder: GraphSpec<DecoderInputs, DecoderOutputs>;
}

export const GRAPH_PRECISION_CONTRACT = {
  text_encoder: {
    policy_id: "text-continuation-fp32-v3",
    conversion_mode: "fp32-identity",
  },
  denoiser: {
    policy_id: "denoiser-continuation-fp32-v3",
    conversion_mode: "fp32-identity",
  },
  decoder: {
    policy_id: "decoder-qk-norm-io-geometry-v3",
    conversion_mode: "mixed-fp16",
  },
} as const satisfies Record<
  keyof BrowserGraphSpecs,
  {
    policy_id: string;
    conversion_mode: "mixed-fp16" | "fp32-identity";
  }
>;

export interface BrowserDimensions {
  fps: number;
  num_frames_per_token: number;
  max_tokens: number;
  max_frames: number;
  generation_tokens: number;
  generation_frames: number;
  history_tokens: number;
  history_frames: number;
  root_features_per_frame: number;
  nframe_root_dim: number;
  latent_dim: number;
  hybrid_dim: number;
  motion_dim: number;
  body_dim: number;
  text_condition_dim: number;
  num_joints: number;
}

export interface BrowserGenerationConfig {
  min_frames: number;
  max_frames: number;
  default_cfg_weight: number;
  denoising_steps: number;
  window_policy?: string;
}

export interface BrowserDiffusionSchedule {
  /** Base-model timestep passed to the denoiser, in inference order (noisy to clean). */
  timesteps: number[];
  /** Alpha-bar indexed by base-model timestep. */
  alphas_cumprod: number[];
  /** Previous alpha-bar indexed by base-model timestep, with 1.0 at index zero. */
  alphas_cumprod_prev: number[];
}

export interface BrowserRecenterConfig {
  root_mean: number[];
  root_std: number[];
  /** Indices of x, y, z in one root feature vector. */
  position_indices: [number, number, number];
  /** Indices of cos(heading), sin(heading). */
  heading_indices: [number, number];
}

export interface BrowserLatentQuantization {
  /** FSQ level count for each latent dimension. */
  levels: number[];
  /** Post-quantization normalization statistics. */
  mean: number[];
  std: number[];
}

export interface BrowserTokenizerSpec {
  /** Relative directory containing tokenizer.json and tokenizer_config.json. */
  directory: string;
  max_length: number;
  /** Source model identifier recorded for provenance. */
  model_id?: string;
}

export interface BrowserLicenseNotice {
  component: string;
  license: string;
  notice: string;
}

export interface BrowserSkeleton {
  name: string;
  root_index: number;
  parents: number[];
  joint_names: string[];
  neutral_joints: number[][];
}

export interface BrowserStatsPayload {
  mean: number[];
  std: number[];
  normalization_denominator: number[];
}

export interface BrowserStats {
  motion: BrowserStatsPayload;
  global_root: BrowserStatsPayload;
  local_root: BrowserStatsPayload;
  body: BrowserStatsPayload;
  post_quantization: BrowserStatsPayload;
}

export type BrowserMotionLayout = Record<string, [number, number]>;

export type BrowserRequiredWebGpuFeatures = [
  typeof REQUIRED_WEBGPU_FEATURE,
];

export interface BrowserPrecisionToolchain {
  torch: string;
  onnx: string;
  onnxruntime: string;
}

export interface BrowserInitializerPrecisionStats {
  count_by_dtype: Record<string, number>;
  bytes_by_dtype: Record<string, number>;
  total_count: number;
  total_bytes: number;
}

export interface BrowserGraphPrecisionSummary<
  GraphName extends keyof BrowserGraphSpecs = keyof BrowserGraphSpecs,
> {
  schema_version: 1;
  graph_name: GraphName;
  policy_id: (typeof GRAPH_PRECISION_CONTRACT)[GraphName]["policy_id"];
  conversion_mode: (typeof GRAPH_PRECISION_CONTRACT)[GraphName]["conversion_mode"];
  source_sha256: Sha256Hex;
  output_sha256: Sha256Hex;
  source_size_bytes: number;
  output_size_bytes: number;
  size_reduction_bytes: number;
  size_reduction_fraction: number;
  source_initializers: BrowserInitializerPrecisionStats;
  output_initializers: BrowserInitializerPrecisionStats;
}

export interface BrowserPrecisionGraphSummaries {
  text_encoder: BrowserGraphPrecisionSummary<"text_encoder">;
  denoiser: BrowserGraphPrecisionSummary<"denoiser">;
  decoder: BrowserGraphPrecisionSummary<"decoder">;
}

export interface BrowserMixedPrecision {
  format: typeof MIXED_PRECISION_FORMAT;
  policy_version: typeof MIXED_PRECISION_POLICY_VERSION;
  public_io_dtype: typeof MIXED_PRECISION_PUBLIC_IO_DTYPE;
  required_webgpu_features: BrowserRequiredWebGpuFeatures;
  source_onnx_bytes: number;
  mixed_onnx_bytes: number;
  saved_onnx_bytes: number;
  saved_onnx_fraction: number;
  toolchain: BrowserPrecisionToolchain;
  graphs: BrowserPrecisionGraphSummaries;
}

export interface BrowserRuntimeCapabilities {
  onnx_opset?: number;
  contract_revision: 3;
  batch_size?: number;
  text_only: true;
  required_webgpu_features: BrowserRequiredWebGpuFeatures;
  detailed_motion_outputs?: boolean;
  motion_correction_included?: boolean;
  global_translation_y_must_be_zero?: boolean;
}

export interface BrowserCapabilities {
  text_conditioning?: boolean;
  initial_root_transform?: boolean;
  detailed_motion_outputs?: boolean;
  motion_correction?: boolean;
}

export interface BrowserModelPackManifest {
  format: typeof MODEL_PACK_FORMAT;
  schema_version: typeof MODEL_PACK_SCHEMA_VERSION;
  model: {
    id: string;
    variant: string;
  };
  files: Record<string, ManifestFile>;
  tokenizer: BrowserTokenizerSpec;
  graphs: BrowserGraphSpecs;
  precision: BrowserMixedPrecision;
  dimensions: BrowserDimensions;
  generation: BrowserGenerationConfig;
  diffusion: BrowserDiffusionSchedule;
  recenter: BrowserRecenterConfig;
  latent_quantization: BrowserLatentQuantization;
  notices?: Array<string | BrowserLicenseNotice>;
  license_notices?: BrowserLicenseNotice[];
  skeleton?: BrowserSkeleton;
  stats?: BrowserStats;
  motion_layout?: BrowserMotionLayout;
  capabilities?: BrowserCapabilities;
  runtime: BrowserRuntimeCapabilities;
}

export class ManifestValidationError extends Error {
  constructor(message: string) {
    super(`Invalid ARDY browser model-pack manifest: ${message}`);
    this.name = "ManifestValidationError";
  }
}

const SHA256_RE = /^[0-9a-f]{64}$/;
const EXPECTED_GRAPH_NAMES = [
  "text_encoder",
  "denoiser",
  "decoder",
] as const satisfies ReadonlyArray<keyof BrowserGraphSpecs>;
const FRACTION_TOLERANCE = 1e-9;

function fail(message: string): never {
  throw new ManifestValidationError(message);
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function stringAt(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(`${path} must be a non-empty string`);
  }
  return value;
}

function finiteAt(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${path} must be a finite number`);
  }
  return value;
}

function positiveAt(value: unknown, path: string): number {
  const number = finiteAt(value, path);
  if (!(number > 0)) {
    fail(`${path} must be positive`);
  }
  return number;
}

function positiveIntegerAt(value: unknown, path: string): number {
  const number = positiveAt(value, path);
  if (!Number.isSafeInteger(number)) {
    fail(`${path} must be a positive safe integer`);
  }
  return number;
}

function nonNegativeIntegerAt(value: unknown, path: string): number {
  const number = finiteAt(value, path);
  if (!Number.isSafeInteger(number) || number < 0) {
    fail(`${path} must be a non-negative safe integer`);
  }
  return number;
}

function fractionAt(value: unknown, path: string): number {
  const number = finiteAt(value, path);
  if (number < 0 || number > 1) {
    fail(`${path} must be between 0 and 1`);
  }
  return number;
}

function validateFraction(
  actual: number,
  numerator: number,
  denominator: number,
  path: string,
): void {
  const expected = numerator / denominator;
  if (Math.abs(actual - expected) > FRACTION_TOLERANCE) {
    fail(`${path} does not match the declared byte reduction`);
  }
}

function validateRequiredWebGpuFeatures(
  value: unknown,
  path: string,
): BrowserRequiredWebGpuFeatures {
  if (!Array.isArray(value)) {
    fail(`${path} must be an array`);
  }
  const seen = new Set<string>();
  for (const [index, rawFeature] of value.entries()) {
    const feature = stringAt(rawFeature, `${path}[${index}]`);
    if (seen.has(feature)) {
      fail(`${path} must not contain duplicate feature ${JSON.stringify(feature)}`);
    }
    seen.add(feature);
    if (feature !== REQUIRED_WEBGPU_FEATURE) {
      fail(`${path} contains unknown WebGPU feature ${JSON.stringify(feature)}`);
    }
  }
  if (!seen.has(REQUIRED_WEBGPU_FEATURE)) {
    fail(`${path} must contain ${JSON.stringify(REQUIRED_WEBGPU_FEATURE)}`);
  }
  return value as BrowserRequiredWebGpuFeatures;
}

function rejectLegacyFields(
  record: Readonly<Record<string, unknown>>,
  path: string,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (Object.hasOwn(record, field)) {
      fail(`${path}.${field} is not supported by the current text-only contract`);
    }
  }
}

function numberArrayAt(value: unknown, path: string, length: number): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    fail(`${path} must contain exactly ${length} numbers`);
  }
  return value.map((item, index) => finiteAt(item, `${path}[${index}]`));
}

function integerArrayAt(value: unknown, path: string, length: number): number[] {
  return numberArrayAt(value, path, length).map((item, index) => {
    if (!Number.isSafeInteger(item)) {
      fail(`${path}[${index}] must be a safe integer`);
    }
    return item;
  });
}

export function normalizePackPath(path: string): string {
  const normalized = path.replaceAll("\\", "/").replace(/^\.\/+/, "");
  if (
    normalized.length === 0 ||
    normalized.startsWith("/") ||
    normalized.endsWith("/") ||
    normalized.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    fail(`unsafe relative asset path ${JSON.stringify(path)}`);
  }
  return normalized;
}

function validateNameMap(
  value: unknown,
  path: string,
  requiredKeys: readonly string[],
): Record<string, string> {
  const record = objectAt(value, path);
  const actualKeys = Object.keys(record);
  if (
    actualKeys.length !== requiredKeys.length ||
    requiredKeys.some((key) => !Object.hasOwn(record, key))
  ) {
    fail(`${path} must contain exactly ${requiredKeys.join(", ")}`);
  }
  const names = requiredKeys.map((key) =>
    stringAt(record[key], `${path}.${key}`),
  );
  if (new Set(names).size !== names.length) {
    fail(`${path} must not map multiple semantic fields to the same ONNX name`);
  }
  return record as Record<string, string>;
}

function validateGraph(
  value: unknown,
  path: string,
  files: ReadonlySet<string>,
  requiredInputs: readonly string[],
  requiredOutputs: readonly string[],
): Set<string> {
  const graph = objectAt(value, path);
  const model = normalizePackPath(stringAt(graph.model, `${path}.model`));
  if (!files.has(model)) {
    fail(`${path}.model references undeclared file ${JSON.stringify(model)}`);
  }
  const referencedFiles = new Set([model]);
  validateNameMap(graph.inputs, `${path}.inputs`, requiredInputs);
  validateNameMap(graph.outputs, `${path}.outputs`, requiredOutputs);

  if (graph.external_data !== undefined) {
    if (!Array.isArray(graph.external_data)) {
      fail(`${path}.external_data must be an array`);
    }
    const embeddedPaths = new Set<string>();
    for (const [index, rawExternal] of graph.external_data.entries()) {
      const external = objectAt(rawExternal, `${path}.external_data[${index}]`);
      const embeddedPath = stringAt(
        external.path,
        `${path}.external_data[${index}].path`,
      );
      if (embeddedPaths.has(embeddedPath)) {
        fail(`${path}.external_data contains duplicate ONNX path ${embeddedPath}`);
      }
      embeddedPaths.add(embeddedPath);
      const file = normalizePackPath(
        stringAt(external.file, `${path}.external_data[${index}].file`),
      );
      if (!files.has(file)) {
        fail(`${path}.external_data references undeclared file ${JSON.stringify(file)}`);
      }
      referencedFiles.add(file);
    }
  }
  return referencedFiles;
}

function validateManifestFiles(value: unknown): Set<string> {
  const files = objectAt(value, "files");
  const normalizedPaths = new Set<string>();
  if (Object.keys(files).length === 0) {
    fail("files must not be empty");
  }
  for (const [rawPath, rawDescription] of Object.entries(files)) {
    const path = normalizePackPath(rawPath);
    if (path !== rawPath) {
      fail(`files key ${JSON.stringify(rawPath)} is not in canonical form`);
    }
    if (normalizedPaths.has(path)) {
      fail(`files contains duplicate normalized path ${JSON.stringify(path)}`);
    }
    normalizedPaths.add(path);

    const description = objectAt(rawDescription, `files.${path}`);
    const sha256 = stringAt(description.sha256, `files.${path}.sha256`);
    if (!SHA256_RE.test(sha256)) {
      fail(`files.${path}.sha256 must be a lowercase SHA-256 hex digest`);
    }
    nonNegativeIntegerAt(description.size_bytes, `files.${path}.size_bytes`);
    if (description.media_type !== undefined) {
      stringAt(description.media_type, `files.${path}.media_type`);
    }
  }
  return normalizedPaths;
}

function validateSha256(value: unknown, path: string): Sha256Hex {
  const sha256 = stringAt(value, path);
  if (!SHA256_RE.test(sha256)) {
    fail(`${path} must be a lowercase SHA-256 hex digest`);
  }
  return sha256;
}

function validateInitializerPrecisionStats(
  value: unknown,
  path: string,
): BrowserInitializerPrecisionStats {
  const stats = objectAt(value, path);
  const countByDtype = objectAt(
    stats.count_by_dtype,
    `${path}.count_by_dtype`,
  );
  const bytesByDtype = objectAt(
    stats.bytes_by_dtype,
    `${path}.bytes_by_dtype`,
  );
  const countDtypes = Object.keys(countByDtype);
  const byteDtypes = Object.keys(bytesByDtype);
  if (
    countDtypes.length !== byteDtypes.length ||
    countDtypes.some((dtype) => !Object.hasOwn(bytesByDtype, dtype))
  ) {
    fail(`${path} dtype keys must match between count_by_dtype and bytes_by_dtype`);
  }

  let initializerCount = 0;
  let initializerBytes = 0;
  for (const dtype of countDtypes) {
    if (dtype.trim().length === 0) {
      fail(`${path}.count_by_dtype must not contain an empty dtype`);
    }
    initializerCount += positiveIntegerAt(
      countByDtype[dtype],
      `${path}.count_by_dtype.${dtype}`,
    );
    initializerBytes += nonNegativeIntegerAt(
      bytesByDtype[dtype],
      `${path}.bytes_by_dtype.${dtype}`,
    );
    if (
      !Number.isSafeInteger(initializerCount) ||
      !Number.isSafeInteger(initializerBytes)
    ) {
      fail(`${path} totals must be safe integers`);
    }
  }

  const totalCount = nonNegativeIntegerAt(
    stats.total_count,
    `${path}.total_count`,
  );
  const totalBytes = nonNegativeIntegerAt(
    stats.total_bytes,
    `${path}.total_bytes`,
  );
  if (initializerCount !== totalCount) {
    fail(`${path}.total_count must equal the sum of count_by_dtype`);
  }
  if (initializerBytes !== totalBytes) {
    fail(`${path}.total_bytes must equal the sum of bytes_by_dtype`);
  }
  return stats as unknown as BrowserInitializerPrecisionStats;
}

function rejectInitializerDtype(
  stats: BrowserInitializerPrecisionStats,
  dtype: string,
  path: string,
): void {
  if (
    (stats.count_by_dtype[dtype] ?? 0) !== 0 ||
    (stats.bytes_by_dtype[dtype] ?? 0) !== 0
  ) {
    fail(`${path} must not contain ${dtype} initializers`);
  }
}

function initializerStatsEqual(
  left: BrowserInitializerPrecisionStats,
  right: BrowserInitializerPrecisionStats,
): boolean {
  if (
    left.total_count !== right.total_count ||
    left.total_bytes !== right.total_bytes
  ) {
    return false;
  }
  for (const field of ["count_by_dtype", "bytes_by_dtype"] as const) {
    const leftEntries = Object.entries(left[field]).sort(([a], [b]) =>
      a.localeCompare(b),
    );
    const rightEntries = Object.entries(right[field]).sort(([a], [b]) =>
      a.localeCompare(b),
    );
    if (
      leftEntries.length !== rightEntries.length ||
      leftEntries.some(
        ([dtype, amount], index) =>
          dtype !== rightEntries[index]?.[0] ||
          amount !== rightEntries[index]?.[1],
      )
    ) {
      return false;
    }
  }
  return true;
}

function validateMixedPrecision(
  value: unknown,
  graphsValue: unknown,
  filesValue: unknown,
): BrowserMixedPrecision {
  const precision = objectAt(value, "precision");
  if (precision.format !== MIXED_PRECISION_FORMAT) {
    fail(`precision.format must be ${JSON.stringify(MIXED_PRECISION_FORMAT)}`);
  }
  if (precision.policy_version !== MIXED_PRECISION_POLICY_VERSION) {
    fail(`precision.policy_version must be ${MIXED_PRECISION_POLICY_VERSION}`);
  }
  if (precision.public_io_dtype !== MIXED_PRECISION_PUBLIC_IO_DTYPE) {
    fail(
      `precision.public_io_dtype must be ${JSON.stringify(MIXED_PRECISION_PUBLIC_IO_DTYPE)}`,
    );
  }
  validateRequiredWebGpuFeatures(
    precision.required_webgpu_features,
    "precision.required_webgpu_features",
  );

  const toolchain = objectAt(precision.toolchain, "precision.toolchain");
  for (const name of ["torch", "onnx", "onnxruntime"] as const) {
    stringAt(toolchain[name], `precision.toolchain.${name}`);
  }

  const sourceBytes = positiveIntegerAt(
    precision.source_onnx_bytes,
    "precision.source_onnx_bytes",
  );
  const mixedBytes = positiveIntegerAt(
    precision.mixed_onnx_bytes,
    "precision.mixed_onnx_bytes",
  );
  const savedBytes = nonNegativeIntegerAt(
    precision.saved_onnx_bytes,
    "precision.saved_onnx_bytes",
  );
  if (sourceBytes - mixedBytes !== savedBytes) {
    fail(
      "precision.saved_onnx_bytes must equal source_onnx_bytes - mixed_onnx_bytes",
    );
  }
  const savedFraction = fractionAt(
    precision.saved_onnx_fraction,
    "precision.saved_onnx_fraction",
  );
  validateFraction(
    savedFraction,
    savedBytes,
    sourceBytes,
    "precision.saved_onnx_fraction",
  );

  const summaries = objectAt(precision.graphs, "precision.graphs");
  const actualGraphNames = Object.keys(summaries);
  if (
    actualGraphNames.length !== EXPECTED_GRAPH_NAMES.length ||
    EXPECTED_GRAPH_NAMES.some((name) => !Object.hasOwn(summaries, name))
  ) {
    fail(`precision.graphs must contain exactly ${EXPECTED_GRAPH_NAMES.join(", ")}`);
  }

  const graphs = objectAt(graphsValue, "graphs");
  const files = objectAt(filesValue, "files");
  let graphSourceBytes = 0;
  let graphMixedBytes = 0;
  let graphSavedBytes = 0;
  for (const graphName of EXPECTED_GRAPH_NAMES) {
    const path = `precision.graphs.${graphName}`;
    const summary = objectAt(summaries[graphName], path);
    if (summary.schema_version !== 1) {
      fail(`${path}.schema_version must be 1`);
    }
    if (summary.graph_name !== graphName) {
      fail(`${path}.graph_name must be ${JSON.stringify(graphName)}`);
    }
    const contract = GRAPH_PRECISION_CONTRACT[graphName];
    if (summary.policy_id !== contract.policy_id) {
      fail(`${path}.policy_id must be ${JSON.stringify(contract.policy_id)}`);
    }
    if (summary.conversion_mode !== contract.conversion_mode) {
      fail(
        `${path}.conversion_mode must be ${JSON.stringify(contract.conversion_mode)}`,
      );
    }
    const sourceSha256 = validateSha256(
      summary.source_sha256,
      `${path}.source_sha256`,
    );
    const outputSha256 = validateSha256(
      summary.output_sha256,
      `${path}.output_sha256`,
    );
    const sourceInitializers = validateInitializerPrecisionStats(
      summary.source_initializers,
      `${path}.source_initializers`,
    );
    const outputInitializers = validateInitializerPrecisionStats(
      summary.output_initializers,
      `${path}.output_initializers`,
    );
    rejectInitializerDtype(
      sourceInitializers,
      "float16",
      `${path}.source_initializers`,
    );
    rejectInitializerDtype(
      sourceInitializers,
      "bfloat16",
      `${path}.source_initializers`,
    );
    rejectInitializerDtype(
      outputInitializers,
      "bfloat16",
      `${path}.output_initializers`,
    );

    const sourceSize = positiveIntegerAt(
      summary.source_size_bytes,
      `${path}.source_size_bytes`,
    );
    const outputSize = positiveIntegerAt(
      summary.output_size_bytes,
      `${path}.output_size_bytes`,
    );
    const reductionSize = nonNegativeIntegerAt(
      summary.size_reduction_bytes,
      `${path}.size_reduction_bytes`,
    );
    if (sourceSize - outputSize !== reductionSize) {
      fail(
        `${path}.size_reduction_bytes must equal source_size_bytes - output_size_bytes`,
      );
    }
    const reductionFraction = fractionAt(
      summary.size_reduction_fraction,
      `${path}.size_reduction_fraction`,
    );
    validateFraction(
      reductionFraction,
      reductionSize,
      sourceSize,
      `${path}.size_reduction_fraction`,
    );

    if (contract.conversion_mode === "fp32-identity") {
      if (
        sourceSize !== outputSize ||
        reductionSize !== 0 ||
        reductionFraction !== 0
      ) {
        fail(`${path} fp32-identity conversion must have zero byte reduction`);
      }
      if (sourceSha256 !== outputSha256) {
        fail(`${path} fp32-identity source_sha256 must equal output_sha256`);
      }
      rejectInitializerDtype(
        outputInitializers,
        "float16",
        `${path}.output_initializers`,
      );
      if (!initializerStatsEqual(sourceInitializers, outputInitializers)) {
        fail(
          `${path} fp32-identity source and output initializer statistics must match`,
        );
      }
    } else {
      if (reductionSize === 0) {
        fail(`${path} mixed-fp16 conversion must reduce the ONNX byte size`);
      }
      if ((outputInitializers.count_by_dtype.float16 ?? 0) === 0) {
        fail(`${path}.output_initializers must contain float16 initializers`);
      }
    }

    const graph = objectAt(graphs[graphName], `graphs.${graphName}`);
    const modelPath = normalizePackPath(
      stringAt(graph.model, `graphs.${graphName}.model`),
    );
    const file = objectAt(files[modelPath], `files.${modelPath}`);
    const declaredSize = nonNegativeIntegerAt(
      file.size_bytes,
      `files.${modelPath}.size_bytes`,
    );
    if (outputSize !== declaredSize) {
      fail(`${path}.output_size_bytes must match files.${modelPath}.size_bytes`);
    }
    const declaredSha256 = validateSha256(
      file.sha256,
      `files.${modelPath}.sha256`,
    );
    if (outputSha256 !== declaredSha256) {
      fail(`${path}.output_sha256 must match files.${modelPath}.sha256`);
    }

    graphSourceBytes += sourceSize;
    graphMixedBytes += outputSize;
    graphSavedBytes += reductionSize;
    if (
      !Number.isSafeInteger(graphSourceBytes) ||
      !Number.isSafeInteger(graphMixedBytes) ||
      !Number.isSafeInteger(graphSavedBytes)
    ) {
      fail("precision graph byte totals must be safe integers");
    }
  }

  if (graphSourceBytes !== sourceBytes) {
    fail("precision.source_onnx_bytes must equal the precision graph source total");
  }
  if (graphMixedBytes !== mixedBytes) {
    fail("precision.mixed_onnx_bytes must equal the precision graph output total");
  }
  if (graphSavedBytes !== savedBytes) {
    fail("precision.saved_onnx_bytes must equal the precision graph reduction total");
  }

  return precision as unknown as BrowserMixedPrecision;
}

function validateDimensions(value: unknown): BrowserDimensions {
  const dimensions = objectAt(value, "dimensions");
  rejectLegacyFields(dimensions, "dimensions", [
    "constraint_max_tokens",
    "constraint_max_frames",
  ]);
  const result = {} as unknown as BrowserDimensions;
  for (const key of [
    "fps",
    "num_frames_per_token",
    "max_tokens",
    "max_frames",
    "generation_tokens",
    "generation_frames",
    "history_tokens",
    "history_frames",
    "root_features_per_frame",
    "nframe_root_dim",
    "latent_dim",
    "hybrid_dim",
    "motion_dim",
    "body_dim",
    "text_condition_dim",
    "num_joints",
  ] as const) {
    result[key] = positiveIntegerAt(dimensions[key], `dimensions.${key}`);
  }
  if (result.max_frames !== result.max_tokens * result.num_frames_per_token) {
    fail("dimensions.max_frames must equal max_tokens * num_frames_per_token");
  }
  if (
    result.generation_frames !==
    result.generation_tokens * result.num_frames_per_token
  ) {
    fail(
      "dimensions.generation_frames must equal generation_tokens * num_frames_per_token",
    );
  }
  if (result.history_frames !== result.history_tokens * result.num_frames_per_token) {
    fail("dimensions.history_frames must equal history_tokens * num_frames_per_token");
  }
  if (result.history_tokens + result.generation_tokens > result.max_tokens) {
    fail("history_tokens + generation_tokens exceeds max_tokens");
  }
  if (
    result.nframe_root_dim !==
    result.root_features_per_frame * result.num_frames_per_token
  ) {
    fail(
      "dimensions.nframe_root_dim must equal root_features_per_frame * num_frames_per_token",
    );
  }
  if (result.hybrid_dim !== result.nframe_root_dim + result.latent_dim) {
    fail("dimensions.hybrid_dim must equal nframe_root_dim + latent_dim");
  }
  const core40Contract: BrowserDimensions = {
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
  };
  for (const key of Object.keys(core40Contract) as Array<keyof BrowserDimensions>) {
    if (result[key] !== core40Contract[key]) {
      fail(
        `dimensions.${key} must be ${core40Contract[key]} for the browser Core40 runtime`,
      );
    }
  }
  return result;
}

function validateDiffusion(value: unknown, steps: number): void {
  const diffusion = objectAt(value, "diffusion");
  const timesteps = integerArrayAt(diffusion.timesteps, "diffusion.timesteps", steps);
  const alphas = numberArrayAt(
    diffusion.alphas_cumprod,
    "diffusion.alphas_cumprod",
    steps,
  );
  const previous = numberArrayAt(
    diffusion.alphas_cumprod_prev,
    "diffusion.alphas_cumprod_prev",
    steps,
  );

  const seenTimesteps = new Set<number>();
  for (let index = 0; index < steps; index += 1) {
    const timestep = timesteps[index];
    if (timestep < 0 || timestep >= steps || seenTimesteps.has(timestep)) {
      fail(`diffusion.timesteps must be a permutation of 0 through ${steps - 1}`);
    }
    seenTimesteps.add(timestep);
    if (index > 0 && timestep >= timesteps[index - 1]) {
      fail("diffusion.timesteps must be in strictly decreasing inference order");
    }
    if (!(alphas[index] > 0 && alphas[index] <= 1)) {
      fail(`diffusion.alphas_cumprod[${index}] must be in (0, 1]`);
    }
    if (index > 0 && alphas[index] > alphas[index - 1]) {
      fail("diffusion.alphas_cumprod must be monotonically non-increasing");
    }
    if (!(previous[index] > 0 && previous[index] <= 1)) {
      fail(`diffusion.alphas_cumprod_prev[${index}] must be in (0, 1]`);
    }
    const expected = index === 0 ? 1 : alphas[index - 1];
    if (Math.abs(previous[index] - expected) > 1e-6) {
      fail(`diffusion.alphas_cumprod_prev[${index}] does not match the prior alpha`);
    }
  }
}

/**
 * Validate an untrusted JSON value and narrow it to the current runtime contract.
 *
 * Model packs are user-selected archives, so all fields used for allocation,
 * file lookup, tensor shapes, or numerical kernels are checked before any model
 * bytes are handed to ONNX Runtime.
 */
export function validateModelPackManifest(value: unknown): BrowserModelPackManifest {
  const manifest = objectAt(value, "manifest");
  if (manifest.format !== MODEL_PACK_FORMAT) {
    fail(`format must be ${JSON.stringify(MODEL_PACK_FORMAT)}`);
  }
  if (manifest.schema_version !== MODEL_PACK_SCHEMA_VERSION) {
    fail(`schema_version must be ${MODEL_PACK_SCHEMA_VERSION}`);
  }

  const model = objectAt(manifest.model, "model");
  stringAt(model.id, "model.id");
  stringAt(model.variant, "model.variant");
  const files = validateManifestFiles(manifest.files);

  const tokenizer = objectAt(manifest.tokenizer, "tokenizer");
  const tokenizerDirectory = normalizePackPath(
    stringAt(tokenizer.directory, "tokenizer.directory"),
  );
  positiveIntegerAt(tokenizer.max_length, "tokenizer.max_length");
  if (tokenizer.model_id !== undefined) {
    stringAt(tokenizer.model_id, "tokenizer.model_id");
  }
  for (const requiredName of ["tokenizer.json", "tokenizer_config.json"]) {
    const requiredPath = `${tokenizerDirectory}/${requiredName}`;
    if (!files.has(requiredPath)) {
      fail(`tokenizer requires declared file ${JSON.stringify(requiredPath)}`);
    }
  }

  const graphs = objectAt(manifest.graphs, "graphs");
  const actualGraphNames = Object.keys(graphs);
  if (
    actualGraphNames.length !== EXPECTED_GRAPH_NAMES.length ||
    EXPECTED_GRAPH_NAMES.some((name) => !Object.hasOwn(graphs, name))
  ) {
    fail(
      `graphs must contain exactly ${EXPECTED_GRAPH_NAMES.join(", ")}`,
    );
  }
  const referencedFiles = new Set([
    `${tokenizerDirectory}/tokenizer.json`,
    `${tokenizerDirectory}/tokenizer_config.json`,
  ]);
  const collectGraphFiles = (graphFiles: ReadonlySet<string>): void => {
    for (const file of graphFiles) referencedFiles.add(file);
  };
  collectGraphFiles(validateGraph(
    graphs.text_encoder,
    "graphs.text_encoder",
    files,
    ["inputIds", "attentionMask", "tokenTypeIds"],
    ["textConditions"],
  ));
  collectGraphFiles(validateGraph(
    graphs.denoiser,
    "graphs.denoiser",
    files,
    [
      "cfgWeight",
      "x",
      "historyLength",
      "generationLength",
      "historyMask",
      "generationMask",
      "historyTokenMask",
      "generationTokenMask",
      "textConditions",
      "timestep",
      "firstHeadingAngle",
    ],
    ["predX0"],
  ));
  collectGraphFiles(validateGraph(
    graphs.decoder,
    "graphs.decoder",
    files,
    ["hybridTokens", "motionPadMask", "globalTranslation"],
    [
      "normalizedMotion",
      "posedJoints",
      "localRotations",
      "globalRotations",
      "rootPositions",
      "footContacts",
      "globalRootHeading",
    ],
  ));
  for (const file of files) {
    if (!referencedFiles.has(file)) {
      fail(`files contains unreferenced asset ${JSON.stringify(file)}`);
    }
  }
  validateMixedPrecision(manifest.precision, manifest.graphs, manifest.files);

  const dimensions = validateDimensions(manifest.dimensions);
  const generation = objectAt(manifest.generation, "generation");
  rejectLegacyFields(generation, "generation", [
    "default_text_cfg_weight",
    "default_constraint_cfg_weight",
  ]);
  const minFrames = positiveIntegerAt(generation.min_frames, "generation.min_frames");
  const maxFrames = positiveIntegerAt(generation.max_frames, "generation.max_frames");
  if (minFrames !== dimensions.generation_frames) {
    fail("generation.min_frames must equal one generation window");
  }
  if (maxFrames !== 10 * dimensions.fps) {
    fail("generation.max_frames must equal the browser 10-second limit");
  }
  positiveAt(generation.default_cfg_weight, "generation.default_cfg_weight");
  if (generation.window_policy !== undefined) {
    stringAt(generation.window_policy, "generation.window_policy");
  }
  const denoisingSteps = positiveIntegerAt(
    generation.denoising_steps,
    "generation.denoising_steps",
  );
  if (denoisingSteps !== 10) {
    fail("generation.denoising_steps must be 10 for the browser Core40 runtime");
  }
  validateDiffusion(manifest.diffusion, denoisingSteps);

  const recenter = objectAt(manifest.recenter, "recenter");
  const rootSize = dimensions.root_features_per_frame;
  const rootMean = numberArrayAt(recenter.root_mean, "recenter.root_mean", rootSize);
  const rootStd = numberArrayAt(recenter.root_std, "recenter.root_std", rootSize);
  rootStd.forEach((std, index) => {
    if (!(std > 0)) {
      fail(`recenter.root_std[${index}] must be positive`);
    }
  });
  const positions = integerArrayAt(
    recenter.position_indices,
    "recenter.position_indices",
    3,
  );
  const headings = integerArrayAt(
    recenter.heading_indices,
    "recenter.heading_indices",
    2,
  );
  if (
    new Set([...positions, ...headings]).size !== 5 ||
    [...positions, ...headings].some((index) => index < 0 || index >= rootSize)
  ) {
    fail("recenter position/heading indices must be distinct valid root feature indices");
  }
  if (rootMean.length !== rootSize) {
    fail("recenter root statistics are inconsistent");
  }

  const quantization = objectAt(manifest.latent_quantization, "latent_quantization");
  const latentSize = dimensions.latent_dim;
  const levels = integerArrayAt(
    quantization.levels,
    "latent_quantization.levels",
    latentSize,
  );
  levels.forEach((level, index) => {
    if (level < 2) {
      fail(`latent_quantization.levels[${index}] must be at least 2`);
    }
  });
  numberArrayAt(quantization.mean, "latent_quantization.mean", latentSize);
  const latentStd = numberArrayAt(
    quantization.std,
    "latent_quantization.std",
    latentSize,
  );
  latentStd.forEach((std, index) => {
    if (!(std > 0)) {
      fail(`latent_quantization.std[${index}] must be positive`);
    }
  });

  const validateLicenseNotices = (value: unknown, path: string): void => {
    if (!Array.isArray(value)) {
      fail(`${path} must be an array`);
    }
    value.forEach((rawNotice, index) => {
      const notice = objectAt(rawNotice, `${path}[${index}]`);
      stringAt(notice.component, `${path}[${index}].component`);
      stringAt(notice.license, `${path}[${index}].license`);
      stringAt(notice.notice, `${path}[${index}].notice`);
    });
  };
  if (manifest.notices !== undefined) {
    if (!Array.isArray(manifest.notices)) {
      fail("notices must be an array");
    }
    manifest.notices.forEach((rawNotice, index) => {
      if (typeof rawNotice === "string") {
        stringAt(rawNotice, `notices[${index}]`);
        return;
      }
      const notice = objectAt(rawNotice, `notices[${index}]`);
      stringAt(notice.component, `notices[${index}].component`);
      stringAt(notice.license, `notices[${index}].license`);
      stringAt(notice.notice, `notices[${index}].notice`);
    });
  }
  if (manifest.license_notices !== undefined) {
    validateLicenseNotices(manifest.license_notices, "license_notices");
  }

  if (manifest.skeleton !== undefined) {
    const skeleton = objectAt(manifest.skeleton, "skeleton");
    stringAt(skeleton.name, "skeleton.name");
    const rootIndex = nonNegativeIntegerAt(
      skeleton.root_index,
      "skeleton.root_index",
    );
    const parents = integerArrayAt(
      skeleton.parents,
      "skeleton.parents",
      dimensions.num_joints,
    );
    if (
      rootIndex >= dimensions.num_joints ||
      parents[rootIndex] !== -1 ||
      parents.filter((parent) => parent === -1).length !== 1
    ) {
      fail("skeleton must contain one valid root");
    }
    parents.forEach((parent, index) => {
      if (index !== rootIndex && (parent < 0 || parent >= index)) {
        fail(`skeleton.parents[${index}] must reference an earlier joint`);
      }
    });
    if (
      !Array.isArray(skeleton.joint_names) ||
      skeleton.joint_names.length !== dimensions.num_joints
    ) {
      fail(`skeleton.joint_names must contain ${dimensions.num_joints} names`);
    }
    skeleton.joint_names.forEach((name, index) =>
      stringAt(name, `skeleton.joint_names[${index}]`),
    );
    if (
      !Array.isArray(skeleton.neutral_joints) ||
      skeleton.neutral_joints.length !== dimensions.num_joints
    ) {
      fail(
        `skeleton.neutral_joints must contain ${dimensions.num_joints} joints`,
      );
    }
    skeleton.neutral_joints.forEach((joint, index) =>
      numberArrayAt(joint, `skeleton.neutral_joints[${index}]`, 3),
    );
  }

  if (manifest.stats !== undefined) {
    const stats = objectAt(manifest.stats, "stats");
    for (const key of [
      "motion",
      "global_root",
      "local_root",
      "body",
      "post_quantization",
    ]) {
      const payload = objectAt(stats[key], `stats.${key}`);
      if (
        !Array.isArray(payload.mean) ||
        !Array.isArray(payload.std) ||
        !Array.isArray(payload.normalization_denominator) ||
        payload.mean.length === 0 ||
        payload.mean.length !== payload.std.length ||
        payload.mean.length !== payload.normalization_denominator.length
      ) {
        fail(`stats.${key} arrays must have the same non-zero length`);
      }
      numberArrayAt(payload.mean, `stats.${key}.mean`, payload.mean.length);
      numberArrayAt(payload.std, `stats.${key}.std`, payload.mean.length);
      const denominators = numberArrayAt(
        payload.normalization_denominator,
        `stats.${key}.normalization_denominator`,
        payload.mean.length,
      );
      if (denominators.some((value) => !(value > 0))) {
        fail(`stats.${key}.normalization_denominator must be positive`);
      }
    }
  }

  if (manifest.motion_layout !== undefined) {
    const layout = objectAt(manifest.motion_layout, "motion_layout");
    for (const [name, value] of Object.entries(layout)) {
      const [start, stop] = integerArrayAt(
        value,
        `motion_layout.${name}`,
        2,
      );
      if (start < 0 || stop <= start || stop > dimensions.motion_dim) {
        fail(`motion_layout.${name} must be a valid motion feature slice`);
      }
    }
  }

  const runtime = objectAt(manifest.runtime, "runtime");
  rejectLegacyFields(runtime, "runtime", [
    "constraints_supported",
    "separated_cfg",
    "future_constraints_supported",
  ]);
  if (runtime.onnx_opset !== undefined) {
    positiveIntegerAt(runtime.onnx_opset, "runtime.onnx_opset");
  }
  if (runtime.contract_revision !== 3) {
    fail("runtime.contract_revision must be 3");
  }
  if (runtime.text_only !== true) {
    fail("runtime.text_only must be true");
  }
  validateRequiredWebGpuFeatures(
    runtime.required_webgpu_features,
    "runtime.required_webgpu_features",
  );
  for (const key of [
    "detailed_motion_outputs",
    "motion_correction_included",
    "global_translation_y_must_be_zero",
  ]) {
    if (runtime[key] !== undefined && typeof runtime[key] !== "boolean") {
      fail(`runtime.${key} must be a boolean`);
    }
  }
  if (runtime.batch_size !== undefined) {
    positiveIntegerAt(runtime.batch_size, "runtime.batch_size");
  }

  if (manifest.capabilities !== undefined) {
    const capabilities = objectAt(manifest.capabilities, "capabilities");
    rejectLegacyFields(capabilities, "capabilities", [
      "kinematic_constraints",
      "future_constraints",
      "separated_classifier_free_guidance",
    ]);
    for (const [key, enabled] of Object.entries(capabilities)) {
      if (typeof enabled !== "boolean") {
        fail(`capabilities.${key} must be a boolean`);
      }
    }
  }

  return manifest as unknown as BrowserModelPackManifest;
}

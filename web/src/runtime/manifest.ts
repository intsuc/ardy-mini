// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const MODEL_PACK_FORMAT = "ardy-browser-model-pack";
export const MODEL_PACK_SCHEMA_VERSION = 1;
export const MODEL_PACK_MANIFEST_FILE = "manifest.json";

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

export interface ConstraintDenoiserInputs extends Record<string, string> {
  textCfgWeight: string;
  constraintCfgWeight: string;
  x: string;
  historyLength: string;
  generationLength: string;
  futureLength: string;
  historyMask: string;
  generationMask: string;
  futureMask: string;
  historyTokenMask: string;
  generationTokenMask: string;
  futureTokenMask: string;
  textConditions: string;
  textConditionMask: string;
  timestep: string;
  firstHeadingAngle: string;
  motionMask: string;
  observedMotion: string;
}

export interface DecoderInputs extends Record<string, string> {
  hybridTokens: string;
  motionPadMask: string;
  globalTranslation: string;
}

export interface DecoderOutputs
  extends Record<string, string | undefined> {
  normalizedMotion: string;
  posedJoints: string;
  localRotations?: string;
  globalRotations?: string;
  rootPositions?: string;
  footContacts?: string;
  globalRootHeading?: string;
}

export interface BrowserGraphSpecs {
  text_encoder: GraphSpec<TextEncoderInputs, TextEncoderOutputs>;
  denoiser: GraphSpec<DenoiserInputs, DenoiserOutputs>;
  constraint_denoiser?: GraphSpec<
    ConstraintDenoiserInputs,
    DenoiserOutputs
  >;
  decoder: GraphSpec<DecoderInputs, DecoderOutputs>;
}

export interface BrowserDimensions {
  fps: number;
  num_frames_per_token: number;
  max_tokens: number;
  max_frames: number;
  /** Fixed window owned by the optional constraint graph. */
  constraint_max_tokens?: number;
  constraint_max_frames?: number;
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
  default_text_cfg_weight?: number;
  default_constraint_cfg_weight?: number;
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

export interface BrowserRuntimeCapabilities {
  onnx_opset?: number;
  contract_revision?: number;
  batch_size?: number;
  text_only?: boolean;
  constraints_supported?: boolean;
  separated_cfg?: boolean;
  future_constraints_supported?: boolean;
  detailed_motion_outputs?: boolean;
  motion_correction_included?: boolean;
  global_translation_y_must_be_zero?: boolean;
}

export interface BrowserCapabilities {
  text_conditioning?: boolean;
  kinematic_constraints?: boolean;
  future_constraints?: boolean;
  separated_classifier_free_guidance?: boolean;
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
  runtime?: BrowserRuntimeCapabilities;
}

export class ManifestValidationError extends Error {
  constructor(message: string) {
    super(`Invalid ARDY browser model-pack manifest: ${message}`);
    this.name = "ManifestValidationError";
  }
}

const SHA256_RE = /^[0-9a-f]{64}$/;

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
  for (const key of requiredKeys) {
    stringAt(record[key], `${path}.${key}`);
  }
  const names = Object.values(record).map((name, index) =>
    stringAt(name, `${path}[${index}]`),
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
): void {
  const graph = objectAt(value, path);
  const model = normalizePackPath(stringAt(graph.model, `${path}.model`));
  if (!files.has(model)) {
    fail(`${path}.model references undeclared file ${JSON.stringify(model)}`);
  }
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
    }
  }
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

function validateDimensions(value: unknown): BrowserDimensions {
  const dimensions = objectAt(value, "dimensions");
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
  if (dimensions.constraint_max_tokens !== undefined) {
    result.constraint_max_tokens = positiveIntegerAt(
      dimensions.constraint_max_tokens,
      "dimensions.constraint_max_tokens",
    );
  }
  if (dimensions.constraint_max_frames !== undefined) {
    result.constraint_max_frames = positiveIntegerAt(
      dimensions.constraint_max_frames,
      "dimensions.constraint_max_frames",
    );
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
  const constraintMaxTokens =
    result.constraint_max_tokens ?? result.max_tokens;
  const constraintMaxFrames =
    result.constraint_max_frames ?? result.max_frames;
  if (
    constraintMaxFrames !==
    constraintMaxTokens * result.num_frames_per_token
  ) {
    fail(
      "dimensions.constraint_max_frames must equal constraint_max_tokens * num_frames_per_token",
    );
  }
  if (constraintMaxTokens < result.generation_tokens + 1) {
    fail("constraint graph must leave room for generation and context");
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
        `dimensions.${key} must be ${core40Contract[key]} for the browser Core40 v1 runtime`,
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
 * Validate an untrusted JSON value and narrow it to the v1 runtime contract.
 *
 * Model packs are user-selectable directories, so all fields used for allocation,
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
  validateGraph(
    graphs.text_encoder,
    "graphs.text_encoder",
    files,
    ["inputIds", "attentionMask", "tokenTypeIds"],
    ["textConditions"],
  );
  validateGraph(
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
  );
  if (graphs.constraint_denoiser !== undefined) {
    validateGraph(
      graphs.constraint_denoiser,
      "graphs.constraint_denoiser",
      files,
      [
        "textCfgWeight",
        "constraintCfgWeight",
        "x",
        "historyLength",
        "generationLength",
        "futureLength",
        "historyMask",
        "generationMask",
        "futureMask",
        "historyTokenMask",
        "generationTokenMask",
        "futureTokenMask",
        "textConditions",
        "textConditionMask",
        "timestep",
        "firstHeadingAngle",
        "motionMask",
        "observedMotion",
      ],
      ["predX0"],
    );
  }
  validateGraph(
    graphs.decoder,
    "graphs.decoder",
    files,
    ["hybridTokens", "motionPadMask", "globalTranslation"],
    ["normalizedMotion", "posedJoints"],
  );
  const decoder = objectAt(graphs.decoder, "graphs.decoder");
  const decoderOutputs = objectAt(
    decoder.outputs,
    "graphs.decoder.outputs",
  );
  const richDecoderOutputs = [
    "localRotations",
    "globalRotations",
    "rootPositions",
    "footContacts",
    "globalRootHeading",
  ] as const;
  const presentRichOutputs = richDecoderOutputs.filter(
    (key) => decoderOutputs[key] !== undefined,
  );
  if (
    presentRichOutputs.length !== 0 &&
    presentRichOutputs.length !== richDecoderOutputs.length
  ) {
    fail(
      "graphs.decoder.outputs must provide either all structured motion outputs or none",
    );
  }
  for (const key of presentRichOutputs) {
    stringAt(
      decoderOutputs[key],
      `graphs.decoder.outputs.${key}`,
    );
  }

  const dimensions = validateDimensions(manifest.dimensions);
  const generation = objectAt(manifest.generation, "generation");
  const minFrames = positiveIntegerAt(generation.min_frames, "generation.min_frames");
  const maxFrames = positiveIntegerAt(generation.max_frames, "generation.max_frames");
  if (minFrames !== dimensions.generation_frames) {
    fail("generation.min_frames must equal one generation window");
  }
  if (maxFrames !== 10 * dimensions.fps) {
    fail("generation.max_frames must equal the browser v1 10-second limit");
  }
  positiveAt(generation.default_cfg_weight, "generation.default_cfg_weight");
  if (generation.default_text_cfg_weight !== undefined) {
    positiveAt(
      generation.default_text_cfg_weight,
      "generation.default_text_cfg_weight",
    );
  }
  if (generation.default_constraint_cfg_weight !== undefined) {
    positiveAt(
      generation.default_constraint_cfg_weight,
      "generation.default_constraint_cfg_weight",
    );
  }
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

  if (manifest.runtime !== undefined) {
    const runtime = objectAt(manifest.runtime, "runtime");
    if (runtime.onnx_opset !== undefined) {
      positiveIntegerAt(runtime.onnx_opset, "runtime.onnx_opset");
    }
    if (runtime.contract_revision !== undefined) {
      positiveIntegerAt(runtime.contract_revision, "runtime.contract_revision");
    }
    for (const key of [
      "text_only",
      "constraints_supported",
      "separated_cfg",
      "future_constraints_supported",
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
    if (
      runtime.constraints_supported === true &&
      graphs.constraint_denoiser === undefined
    ) {
      fail("runtime declares constraints without a constraint_denoiser graph");
    }
  }

  if (manifest.capabilities !== undefined) {
    const capabilities = objectAt(manifest.capabilities, "capabilities");
    for (const [key, enabled] of Object.entries(capabilities)) {
      if (typeof enabled !== "boolean") {
        fail(`capabilities.${key} must be a boolean`);
      }
    }
  }

  return manifest as unknown as BrowserModelPackManifest;
}

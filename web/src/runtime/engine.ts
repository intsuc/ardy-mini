// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { applyDdimStepInPlace, ddimStepForIndex } from "./ddim";
import type { BrowserModelPackManifest } from "./manifest";
import type { ModelPack } from "./model-pack";
import type { RuntimeProgressStage } from "./protocol";
import {
  PortableRandom,
  type PortableRandomState,
} from "./random";
import {
  createRuntimeSessions,
  disposeRuntimeSessions,
  ort,
  type RuntimeSessions,
} from "./sessions";
import { LocalTokenizer } from "./tokenizer";
import {
  createArWindow,
  createMotionPadMask,
  recenterAndRequantize,
  type ArWindow,
} from "./windows";

export class RuntimeCancelledError extends Error {
  constructor(message = "Generation cancelled") {
    super(message);
    this.name = "AbortError";
  }
}

export interface RuntimeProgress {
  stage: RuntimeProgressStage;
  completed: number;
  total: number;
  message?: string;
}

export interface RuntimeLoadOptions {
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeProgress) => void;
}

export interface RuntimeGenerateOptions {
  prompt: string;
  seed: number | string;
  durationFrames?: number;
  durationSeconds?: number;
  cfgWeight?: number;
  historyFrames?: number;
  initialTranslation?: readonly [number, number, number] | Float32Array;
  initialHeading?: number;
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeProgress) => void;
  onChunk?: (chunk: RuntimeGenerationChunk) => void | Promise<void>;
}

export interface RuntimeSessionOptions {
  seed: number | string;
  initialTranslation?: readonly [number, number, number] | Float32Array;
  initialHeading?: number;
}

export interface RuntimeSessionGenerateOptions {
  prompt: string;
  durationFrames: number;
  cfgWeight?: number;
  historyFrames?: number;
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeProgress) => void;
  onChunk?: (chunk: RuntimeGenerationChunk) => void | Promise<void>;
}

export interface RuntimeTimings {
  total: number;
  text: number;
  denoising: number;
  decoding: number;
}

export interface RuntimeMotionArrays {
  motion: Float32Array;
  motionShape: [1, number, number];
  joints: Float32Array;
  jointsShape: [1, number, number, 3];
  localRotations?: Float32Array;
  localRotationsShape?: [1, number, number, 3, 3];
  globalRotations?: Float32Array;
  globalRotationsShape?: [1, number, number, 3, 3];
  rootPositions?: Float32Array;
  rootPositionsShape?: [1, number, 3];
  footContacts?: Uint8Array;
  footContactsShape?: [1, number, number];
  globalRootHeading?: Float32Array;
  globalRootHeadingShape?: [1, number, 2];
}

export interface RuntimeGenerationChunk extends RuntimeMotionArrays {
  seed: number;
  prompt: string;
  fps: number;
  startFrame: number;
  frameCount: number;
  hybridTokens: Float32Array;
  hybridShape: [number, number];
  timingsMs: RuntimeTimings;
}

export interface RuntimeContinuationState {
  frameCount: number;
  hybridTokens: Float32Array;
  hybridDim: number;
  random: PortableRandomState;
  initialTranslation: [number, number, number];
  initialHeading: number;
}

export interface RuntimeGenerationResult extends RuntimeMotionArrays {
  seed: number;
  prompt: string;
  fps: number;
  frameCount: number;
  startFrame: number;
  chunks: number;
  continuation: RuntimeContinuationState;
  timingsMs: RuntimeTimings;
}

interface PreparedHistory {
  history?: Float32Array;
  historyTokens: number;
  historyFrames: number;
  windowStartFrame: number;
  globalTranslation: Float32Array;
  firstHeadingAngle: number;
}

interface DecodedWindow {
  motion: Float32Array;
  joints: Float32Array;
  localRotations?: Float32Array;
  globalRotations?: Float32Array;
  rootPositions?: Float32Array;
  footContacts?: Uint8Array;
  globalRootHeading?: Float32Array;
}

function now(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new RuntimeCancelledError();
  }
}

function tensorFrom(
  outputs: ort.InferenceSession.OnnxValueMapType,
  name: string,
  expectedLength?: number,
): ort.Tensor {
  const value = outputs[name];
  if (!(value instanceof ort.Tensor)) {
    throw new TypeError(`ONNX graph did not return tensor ${JSON.stringify(name)}`);
  }
  if (expectedLength !== undefined && value.size !== expectedLength) {
    throw new RangeError(
      `ONNX output ${name} has ${value.size} elements; expected ${expectedLength}`,
    );
  }
  return value;
}

function floatData(tensor: ort.Tensor, name: string): Float32Array {
  if (!(tensor.data instanceof Float32Array)) {
    throw new TypeError(`ONNX output ${name} must have float32 data`);
  }
  return tensor.data;
}

function boolData(tensor: ort.Tensor, name: string): Uint8Array {
  const data = tensor.data as unknown;
  if (!ArrayBuffer.isView(data) && !Array.isArray(data)) {
    throw new TypeError(`ONNX output ${name} must contain boolean data`);
  }
  const values = data as ArrayLike<number | bigint | boolean>;
  const result = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    result[index] = values[index] ? 1 : 0;
  }
  return result;
}

function int64Scalar(value: number): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.of(BigInt(value)), [1]);
}

function floatScalar(value: number): ort.Tensor {
  return new ort.Tensor("float32", Float32Array.of(Math.fround(value)), [1]);
}

function finiteCfgWeight(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < 0 || resolved > 100) {
    throw new RangeError(`${name} must be finite and between 0 and 100`);
  }
  return resolved;
}

function finiteTransform(
  translation: readonly [number, number, number] | Float32Array | undefined,
  heading: number | undefined,
): { translation: [number, number, number]; heading: number } {
  const resolved = translation ?? [0, 0, 0];
  if (
    resolved.length !== 3 ||
    resolved.some((value) => !Number.isFinite(value)) ||
    (heading !== undefined && !Number.isFinite(heading))
  ) {
    throw new TypeError("Initial root transform must contain finite values");
  }
  const roundedTranslation: [number, number, number] = [
    Math.fround(resolved[0]),
    Math.fround(resolved[1]),
    Math.fround(resolved[2]),
  ];
  if (roundedTranslation.some((value) => !Number.isFinite(value))) {
    throw new RangeError(
      "Initial root translation must be representable as float32",
    );
  }
  const normalizedHeading =
    heading === undefined
      ? 0
      : Math.atan2(Math.sin(heading), Math.cos(heading));
  return {
    translation: roundedTranslation,
    heading: normalizedHeading,
  };
}

function resolveFrameCount(
  manifest: BrowserModelPackManifest,
  options: RuntimeGenerateOptions,
): number {
  const hasFrames = options.durationFrames !== undefined;
  const hasSeconds = options.durationSeconds !== undefined;
  if (hasFrames === hasSeconds) {
    throw new TypeError("Exactly one of durationFrames or durationSeconds is required");
  }
  const frames = hasFrames
    ? options.durationFrames!
    : Math.round(options.durationSeconds! * manifest.dimensions.fps);
  if (
    !Number.isSafeInteger(frames) ||
    frames < manifest.generation.min_frames ||
    frames > manifest.generation.max_frames
  ) {
    throw new RangeError(
      `Requested duration must be ${manifest.generation.min_frames}–${manifest.generation.max_frames} frames`,
    );
  }
  return frames;
}

function sliceFrames(
  source: Float32Array,
  sourceFrame: number,
  frameCount: number,
  stride: number,
): Float32Array {
  return source.slice(sourceFrame * stride, (sourceFrame + frameCount) * stride);
}

function sliceBoolFrames(
  source: Uint8Array,
  sourceFrame: number,
  frameCount: number,
  stride: number,
): Uint8Array {
  return source.slice(sourceFrame * stride, (sourceFrame + frameCount) * stride);
}

function concatFloat(chunks: readonly Float32Array[]): Float32Array {
  const result = new Float32Array(
    chunks.reduce((total, chunk) => total + chunk.length, 0),
  );
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function concatBytes(chunks: readonly Uint8Array[]): Uint8Array {
  const result = new Uint8Array(
    chunks.reduce((total, chunk) => total + chunk.length, 0),
  );
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function optionalFloatOutput(
  outputs: ort.InferenceSession.OnnxValueMapType,
  name: string | undefined,
  expectedLength: number,
): Float32Array | undefined {
  return name === undefined
    ? undefined
    : floatData(tensorFrom(outputs, name, expectedLength), name);
}

function optionalBoolOutput(
  outputs: ort.InferenceSession.OnnxValueMapType,
  name: string | undefined,
  expectedLength: number,
): Uint8Array | undefined {
  if (name === undefined) {
    return undefined;
  }
  const tensor = tensorFrom(outputs, name, expectedLength);
  return boolData(tensor, name);
}

function buildGlobalHybridTokens(
  decodedMotion: Float32Array,
  decoderHybrid: Float32Array,
  generatedFrameOffset: number,
  generationTokenOffset: number,
  tokenCount: number,
  dimensions: BrowserModelPackManifest["dimensions"],
): Float32Array {
  const result = new Float32Array(tokenCount * dimensions.hybrid_dim);
  for (let token = 0; token < tokenCount; token += 1) {
    const destinationToken = token * dimensions.hybrid_dim;
    const sourceToken =
      (generationTokenOffset + token) * dimensions.hybrid_dim;
    for (
      let frameInToken = 0;
      frameInToken < dimensions.num_frames_per_token;
      frameInToken += 1
    ) {
      const sourceFrame =
        generatedFrameOffset +
        token * dimensions.num_frames_per_token +
        frameInToken;
      const destinationRoot =
        destinationToken +
        frameInToken * dimensions.root_features_per_frame;
      const sourceMotion = sourceFrame * dimensions.motion_dim;
      result.set(
        decodedMotion.subarray(
          sourceMotion,
          sourceMotion + dimensions.root_features_per_frame,
        ),
        destinationRoot,
      );
    }
    result.set(
      decoderHybrid.subarray(
        sourceToken + dimensions.nframe_root_dim,
        sourceToken + dimensions.hybrid_dim,
      ),
      destinationToken + dimensions.nframe_root_dim,
    );
  }
  return result;
}

export class BrowserArdyGenerationSession {
  readonly manifest: BrowserModelPackManifest;
  readonly #tokenizer: LocalTokenizer;
  readonly #sessions: RuntimeSessions;
  #random: PortableRandom;
  #globalHybrid = new Float32Array();
  #frameCount = 0;
  #initialTranslation: [number, number, number];
  #initialHeading: number;

  constructor(
    manifest: BrowserModelPackManifest,
    tokenizer: LocalTokenizer,
    sessions: RuntimeSessions,
    options: RuntimeSessionOptions,
  ) {
    const transform = finiteTransform(
      options.initialTranslation,
      options.initialHeading,
    );
    this.manifest = manifest;
    this.#tokenizer = tokenizer;
    this.#sessions = sessions;
    this.#random = new PortableRandom(options.seed);
    this.#initialTranslation = transform.translation;
    this.#initialHeading = transform.heading;
  }

  get frameCount(): number {
    return this.#frameCount;
  }

  get seed(): number {
    return this.#random.seed;
  }

  reset(options: RuntimeSessionOptions): void {
    const transform = finiteTransform(
      options.initialTranslation,
      options.initialHeading,
    );
    this.#random = new PortableRandom(options.seed);
    this.#globalHybrid = new Float32Array();
    this.#frameCount = 0;
    this.#initialTranslation = transform.translation;
    this.#initialHeading = transform.heading;
  }

  /**
   * Crop to the nearest complete four-frame token at or before `frame`.
   * The returned value is the actual branch frame and is safe for continuation.
   */
  branch(frame: number): number {
    if (!Number.isSafeInteger(frame) || frame < 0 || frame > this.#frameCount) {
      throw new RangeError("Branch frame is outside the generated session");
    }
    const { num_frames_per_token: framesPerToken, hybrid_dim: hybridDim } =
      this.manifest.dimensions;
    const tokenCount = Math.floor(frame / framesPerToken);
    this.#globalHybrid = this.#globalHybrid.slice(0, tokenCount * hybridDim);
    this.#frameCount = tokenCount * framesPerToken;
    return this.#frameCount;
  }

  continuation(): RuntimeContinuationState {
    return {
      frameCount: this.#frameCount,
      hybridTokens: this.#globalHybrid.slice(),
      hybridDim: this.manifest.dimensions.hybrid_dim,
      random: this.#random.snapshot(),
      initialTranslation: [...this.#initialTranslation],
      initialHeading: this.#initialHeading,
    };
  }

  restore(state: RuntimeContinuationState): void {
    const { dimensions } = this.manifest;
    const transform = finiteTransform(
      state.initialTranslation,
      state.initialHeading,
    );
    const tokenCount = state.hybridTokens.length / dimensions.hybrid_dim;
    const expectedTokenCount = Math.ceil(
      state.frameCount / dimensions.num_frames_per_token,
    );
    if (
      state.hybridDim !== dimensions.hybrid_dim ||
      state.hybridTokens.length % dimensions.hybrid_dim !== 0 ||
      !Number.isSafeInteger(state.frameCount) ||
      state.frameCount < 0 ||
      tokenCount !== expectedTokenCount
    ) {
      throw new RangeError("Continuation state does not match this model pack");
    }
    for (const value of state.hybridTokens) {
      if (!Number.isFinite(value)) {
        throw new TypeError("Continuation contains non-finite hybrid tokens");
      }
    }
    this.#random = new PortableRandom(state.random.seed);
    this.#random.restore(state.random);
    this.#globalHybrid = state.hybridTokens.slice();
    this.#frameCount = state.frameCount;
    this.#initialTranslation = transform.translation;
    this.#initialHeading = transform.heading;
  }

  #prepareHistory(requestedFrames: number): PreparedHistory {
    const { dimensions, recenter } = this.manifest;
    const framesPerToken = dimensions.num_frames_per_token;
    const availableTokens = Math.min(
      Math.floor(this.#frameCount / framesPerToken),
      this.#globalHybrid.length / dimensions.hybrid_dim,
    );
    const requestedTokens = Math.min(
      dimensions.history_tokens,
      Math.floor(requestedFrames / framesPerToken),
      availableTokens,
    );
    if (requestedTokens === 0) {
      if (availableTokens > 0) {
        const lastFrame = availableTokens * framesPerToken - 1;
        const token = Math.floor(lastFrame / framesPerToken);
        const frameInToken = lastFrame % framesPerToken;
        const offset =
          token * dimensions.hybrid_dim +
          frameInToken * dimensions.root_features_per_frame;
        const rawRoot = (feature: number): number =>
          this.#globalHybrid[offset + feature] *
            recenter.root_std[feature] +
          recenter.root_mean[feature];
        const [positionX, , positionZ] = recenter.position_indices;
        const [headingCos, headingSin] = recenter.heading_indices;
        return {
          historyTokens: 0,
          historyFrames: 0,
          windowStartFrame: this.#frameCount,
          globalTranslation: new Float32Array([
            rawRoot(positionX),
            this.#initialTranslation[1],
            rawRoot(positionZ),
          ]),
          firstHeadingAngle: Math.atan2(
            rawRoot(headingSin),
            rawRoot(headingCos),
          ),
        };
      }
      return {
        historyTokens: 0,
        historyFrames: 0,
        windowStartFrame: this.#frameCount,
        globalTranslation: new Float32Array(this.#initialTranslation),
        firstHeadingAngle: this.#initialHeading,
      };
    }

    const firstToken = availableTokens - requestedTokens;
    const history = this.#globalHybrid.slice(
      firstToken * dimensions.hybrid_dim,
      availableTokens * dimensions.hybrid_dim,
    );
    const rootOffset = (frame: number, feature: number): number => {
      const token = Math.floor(frame / framesPerToken);
      const frameInToken = frame % framesPerToken;
      return (
        token * dimensions.hybrid_dim +
        frameInToken * dimensions.root_features_per_frame +
        feature
      );
    };
    const rawRoot = (frame: number, feature: number): number =>
      history[rootOffset(frame, feature)] * recenter.root_std[feature] +
      recenter.root_mean[feature];
    const historyFrames = requestedTokens * framesPerToken;
    const lastFrame = historyFrames - 1;
    const [positionX, positionY, positionZ] = recenter.position_indices;
    const [headingCos, headingSin] = recenter.heading_indices;
    const centerX = rawRoot(lastFrame, positionX);
    const centerZ = rawRoot(lastFrame, positionZ);
    const firstCos = rawRoot(0, headingCos);
    const firstSin = rawRoot(0, headingSin);

    for (let frame = 0; frame < historyFrames; frame += 1) {
      for (const [feature, center] of [
        [positionX, centerX],
        // ARDY's native recentering only removes x/z because its external
        // translation is constrained to y=0. The browser contract permits an
        // external y offset, so remove that offset from stored world-space
        // history before feeding the model. The decoder adds it back once.
        [positionY, this.#initialTranslation[1]],
        [positionZ, centerZ],
      ] as const) {
        const translated = rawRoot(frame, feature) - center;
        history[rootOffset(frame, feature)] = Math.fround(
          (translated - recenter.root_mean[feature]) /
            recenter.root_std[feature],
        );
      }
    }
    return {
      history,
      historyTokens: requestedTokens,
      historyFrames,
      windowStartFrame: this.#frameCount - historyFrames,
      // The decoder applies this world transform after model-local history has
      // been reconstructed above.
      globalTranslation: new Float32Array([
        centerX,
        this.#initialTranslation[1],
        centerZ,
      ]),
      firstHeadingAngle: Math.atan2(firstSin, firstCos),
    };
  }

  async #encodeText(
    prompt: string,
    signal: AbortSignal | undefined,
    onProgress: RuntimeSessionGenerateOptions["onProgress"],
  ): Promise<{ tensor: ort.Tensor; elapsed: number }> {
    throwIfCancelled(signal);
    onProgress?.({ stage: "encoding-text", completed: 0, total: 1 });
    const started = now();
    const encoded = await this.#tokenizer.encode(prompt);
    throwIfCancelled(signal);
    const inputs = this.manifest.graphs.text_encoder.inputs;
    const outputs = await this.#sessions.textEncoder.run({
      [inputs.inputIds]: new ort.Tensor(
        "int64",
        encoded.inputIds,
        [1, encoded.sequenceLength],
      ),
      [inputs.attentionMask]: new ort.Tensor(
        "int64",
        encoded.attentionMask,
        [1, encoded.sequenceLength],
      ),
      [inputs.tokenTypeIds]: new ort.Tensor(
        "int64",
        encoded.tokenTypeIds,
        [1, encoded.sequenceLength],
      ),
    });
    throwIfCancelled(signal);
    const name = this.manifest.graphs.text_encoder.outputs.textConditions;
    const tensor = tensorFrom(
      outputs,
      name,
      this.manifest.dimensions.text_condition_dim,
    );
    onProgress?.({ stage: "encoding-text", completed: 1, total: 1 });
    return { tensor, elapsed: now() - started };
  }

  async #denoise(
    window: ArWindow,
    textConditions: ort.Tensor,
    prepared: PreparedHistory,
    cfgWeight: number,
    signal: AbortSignal | undefined,
    onProgress: RuntimeSessionGenerateOptions["onProgress"],
    progressOffset: number,
    progressTotal: number,
  ): Promise<number> {
    const { dimensions, diffusion, graphs } = this.manifest;
    const generationStart =
      window.generationTokenOffset * dimensions.hybrid_dim;
    const generationEnd =
      (window.generationTokenOffset + window.generationTokens) *
      dimensions.hybrid_dim;
    const timestepData = BigInt64Array.of(0n);
    const inputs = graphs.denoiser.inputs;
    const feeds: Record<string, ort.Tensor> = {
      [inputs.cfgWeight]: floatScalar(cfgWeight),
      [inputs.x]: new ort.Tensor(
        "float32",
        window.x,
        [1, dimensions.max_tokens, dimensions.hybrid_dim],
      ),
      [inputs.historyLength]: int64Scalar(window.historyFrames),
      [inputs.generationLength]: int64Scalar(window.generationFrames),
      [inputs.historyMask]: new ort.Tensor(
        "float32",
        window.historyMask,
        [1, dimensions.max_frames],
      ),
      [inputs.generationMask]: new ort.Tensor(
        "float32",
        window.generationMask,
        [1, dimensions.max_frames],
      ),
      [inputs.historyTokenMask]: new ort.Tensor(
        "float32",
        window.historyTokenMask,
        [1, dimensions.max_tokens],
      ),
      [inputs.generationTokenMask]: new ort.Tensor(
        "float32",
        window.generationTokenMask,
        [1, dimensions.max_tokens],
      ),
      [inputs.textConditions]: textConditions,
      [inputs.timestep]: new ort.Tensor("int64", timestepData, [1]),
      [inputs.firstHeadingAngle]: floatScalar(prepared.firstHeadingAngle),
    };
    const predictionName = graphs.denoiser.outputs.predX0;

    let elapsed = 0;
    for (
      let inferenceIndex = 0;
      inferenceIndex < diffusion.timesteps.length;
      inferenceIndex += 1
    ) {
      throwIfCancelled(signal);
      const step = ddimStepForIndex(
        diffusion.timesteps,
        diffusion.alphas_cumprod,
        diffusion.alphas_cumprod_prev,
        inferenceIndex,
      );
      timestepData[0] = BigInt(step.timestep);
      const started = now();
      const outputs = await this.#sessions.denoiser.run(feeds);
      throwIfCancelled(signal);
      const prediction = floatData(
        tensorFrom(outputs, predictionName, window.x.length),
        predictionName,
      );
      applyDdimStepInPlace(
        window.x,
        prediction,
        step,
        generationStart,
        generationEnd,
      );
      elapsed += now() - started;
      onProgress?.({
        stage: "denoising",
        completed: progressOffset + inferenceIndex + 1,
        total: progressTotal,
        message: "text window",
      });
    }
    return elapsed;
  }

  async #decode(
    decoderHybrid: Float32Array,
    validTokens: number,
    globalTranslation: Float32Array,
    signal: AbortSignal | undefined,
  ): Promise<DecodedWindow> {
    const { dimensions, graphs } = this.manifest;
    const inputs = graphs.decoder.inputs;
    const outputs = await this.#sessions.decoder.run({
      [inputs.hybridTokens]: new ort.Tensor(
        "float32",
        decoderHybrid,
        [1, dimensions.max_tokens, dimensions.hybrid_dim],
      ),
      [inputs.motionPadMask]: new ort.Tensor(
        "float32",
        createMotionPadMask(dimensions, validTokens),
        [1, dimensions.max_frames],
      ),
      [inputs.globalTranslation]: new ort.Tensor(
        "float32",
        globalTranslation,
        [1, 3],
      ),
    });
    throwIfCancelled(signal);
    const names = graphs.decoder.outputs;
    const frameCount = dimensions.max_frames;
    const jointCount = dimensions.num_joints;
    return {
      motion: floatData(
        tensorFrom(
          outputs,
          names.normalizedMotion,
          frameCount * dimensions.motion_dim,
        ),
        names.normalizedMotion,
      ),
      joints: floatData(
        tensorFrom(
          outputs,
          names.posedJoints,
          frameCount * jointCount * 3,
        ),
        names.posedJoints,
      ),
      localRotations: optionalFloatOutput(
        outputs,
        names.localRotations,
        frameCount * jointCount * 9,
      ),
      globalRotations: optionalFloatOutput(
        outputs,
        names.globalRotations,
        frameCount * jointCount * 9,
      ),
      rootPositions: optionalFloatOutput(
        outputs,
        names.rootPositions,
        frameCount * 3,
      ),
      footContacts: optionalBoolOutput(
        outputs,
        names.footContacts,
        frameCount * 4,
      ),
      globalRootHeading: optionalFloatOutput(
        outputs,
        names.globalRootHeading,
        frameCount * 2,
      ),
    };
  }

  async generate(
    options: RuntimeSessionGenerateOptions,
  ): Promise<RuntimeGenerationResult> {
    if (options.prompt.trim().length === 0) {
      throw new TypeError("Prompt must not be empty");
    }
    const { dimensions, generation } = this.manifest;
    if (
      !Number.isSafeInteger(options.durationFrames) ||
      options.durationFrames <= 0 ||
      options.durationFrames > generation.max_frames
    ) {
      throw new RangeError(
        `Session generation must be 1–${generation.max_frames} frames`,
      );
    }
    const cfgWeight = finiteCfgWeight(
      options.cfgWeight,
      generation.default_cfg_weight,
      "CFG weight",
    );
    const requestedHistoryFrames = options.historyFrames ?? dimensions.history_frames;
    if (
      !Number.isSafeInteger(requestedHistoryFrames) ||
      requestedHistoryFrames < 0 ||
      requestedHistoryFrames > dimensions.history_frames
    ) {
      throw new RangeError("History frame count is invalid");
    }

    // A partial token cannot be encoded back into a continuation without the
    // browser motion encoder. Crop it before extending the session.
    if (this.#frameCount % dimensions.num_frames_per_token !== 0) {
      this.branch(this.#frameCount);
    }

    const totalStart = now();
    const startFrame = this.#frameCount;
    const windowCount = Math.ceil(
      options.durationFrames / dimensions.generation_frames,
    );
    const encoded = await this.#encodeText(
      options.prompt,
      options.signal,
      options.onProgress,
    );
    const chunks: RuntimeGenerationChunk[] = [];
    let writtenFrames = 0;
    let denoisingMs = 0;
    let decodingMs = 0;

    for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
      throwIfCancelled(options.signal);
      const prepared = this.#prepareHistory(requestedHistoryFrames);
      const window = createArWindow(
        dimensions,
        this.#random,
        prepared.history,
      );

      denoisingMs += await this.#denoise(
        window,
        encoded.tensor,
        prepared,
        cfgWeight,
        options.signal,
        options.onProgress,
        windowIndex * this.manifest.diffusion.timesteps.length,
        windowCount * this.manifest.diffusion.timesteps.length,
      );

      const validTokens = window.historyTokens + window.generationTokens;
      const decoderHybrid = new Float32Array(
        dimensions.max_tokens * dimensions.hybrid_dim,
      );
      decoderHybrid.set(
        window.x.subarray(0, validTokens * dimensions.hybrid_dim),
      );
      const nextHistoryStart = Math.max(
        0,
        validTokens - dimensions.history_tokens,
      );
      const recentered = recenterAndRequantize(
        decoderHybrid,
        validTokens,
        dimensions,
        this.manifest.recenter,
        this.manifest.latent_quantization,
        prepared.globalTranslation,
        nextHistoryStart,
      );

      const decodeStart = now();
      const decoded = await this.#decode(
        decoderHybrid,
        validTokens,
        recentered.globalTranslation,
        options.signal,
      );
      const thisDecodingMs = now() - decodeStart;
      decodingMs += thisDecodingMs;

      const framesToCopy = Math.min(
        dimensions.generation_frames,
        options.durationFrames - writtenFrames,
      );
      const generatedFrameOffset = window.historyFrames;
      const generatedTokens = Math.ceil(
        framesToCopy / dimensions.num_frames_per_token,
      );
      const hybridTokens = buildGlobalHybridTokens(
        decoded.motion,
        decoderHybrid,
        generatedFrameOffset,
        window.generationTokenOffset,
        generatedTokens,
        dimensions,
      );
      const oldLength = this.#globalHybrid.length;
      const combined = new Float32Array(oldLength + hybridTokens.length);
      combined.set(this.#globalHybrid);
      combined.set(hybridTokens, oldLength);
      this.#globalHybrid = combined;

      const jointStride = dimensions.num_joints * 3;
      const rotationStride = dimensions.num_joints * 9;
      const chunkStartFrame = this.#frameCount;
      const chunk: RuntimeGenerationChunk = {
        seed: this.#random.seed,
        prompt: options.prompt,
        fps: dimensions.fps,
        startFrame: chunkStartFrame,
        frameCount: framesToCopy,
        motion: sliceFrames(
          decoded.motion,
          generatedFrameOffset,
          framesToCopy,
          dimensions.motion_dim,
        ),
        motionShape: [1, framesToCopy, dimensions.motion_dim],
        joints: sliceFrames(
          decoded.joints,
          generatedFrameOffset,
          framesToCopy,
          jointStride,
        ),
        jointsShape: [1, framesToCopy, dimensions.num_joints, 3],
        hybridTokens: hybridTokens.slice(),
        hybridShape: [generatedTokens, dimensions.hybrid_dim],
        timingsMs: {
          total: 0,
          text: windowIndex === 0 ? encoded.elapsed : 0,
          denoising: denoisingMs,
          decoding: thisDecodingMs,
        },
      };
      if (decoded.localRotations !== undefined) {
        chunk.localRotations = sliceFrames(
          decoded.localRotations,
          generatedFrameOffset,
          framesToCopy,
          rotationStride,
        );
        chunk.localRotationsShape = [
          1,
          framesToCopy,
          dimensions.num_joints,
          3,
          3,
        ];
      }
      if (decoded.globalRotations !== undefined) {
        chunk.globalRotations = sliceFrames(
          decoded.globalRotations,
          generatedFrameOffset,
          framesToCopy,
          rotationStride,
        );
        chunk.globalRotationsShape = [
          1,
          framesToCopy,
          dimensions.num_joints,
          3,
          3,
        ];
      }
      if (decoded.rootPositions !== undefined) {
        chunk.rootPositions = sliceFrames(
          decoded.rootPositions,
          generatedFrameOffset,
          framesToCopy,
          3,
        );
        chunk.rootPositionsShape = [1, framesToCopy, 3];
      }
      if (decoded.footContacts !== undefined) {
        chunk.footContacts = sliceBoolFrames(
          decoded.footContacts,
          generatedFrameOffset,
          framesToCopy,
          4,
        );
        chunk.footContactsShape = [1, framesToCopy, 4];
      }
      if (decoded.globalRootHeading !== undefined) {
        chunk.globalRootHeading = sliceFrames(
          decoded.globalRootHeading,
          generatedFrameOffset,
          framesToCopy,
          2,
        );
        chunk.globalRootHeadingShape = [1, framesToCopy, 2];
      }
      this.#frameCount += framesToCopy;
      writtenFrames += framesToCopy;
      chunk.timingsMs.total = now() - totalStart;
      chunks.push(chunk);
      await options.onChunk?.(chunk);
      options.onProgress?.({
        stage: "decoding",
        completed: windowIndex + 1,
        total: windowCount,
        message: `${this.#frameCount} frames`,
      });
    }

    const motion = concatFloat(chunks.map((chunk) => chunk.motion));
    const joints = concatFloat(chunks.map((chunk) => chunk.joints));
    const localRotationChunks = chunks.flatMap((chunk) =>
      chunk.localRotations === undefined ? [] : [chunk.localRotations],
    );
    const globalRotationChunks = chunks.flatMap((chunk) =>
      chunk.globalRotations === undefined ? [] : [chunk.globalRotations],
    );
    const rootChunks = chunks.flatMap((chunk) =>
      chunk.rootPositions === undefined ? [] : [chunk.rootPositions],
    );
    const contactChunks = chunks.flatMap((chunk) =>
      chunk.footContacts === undefined ? [] : [chunk.footContacts],
    );
    const headingChunks = chunks.flatMap((chunk) =>
      chunk.globalRootHeading === undefined ? [] : [chunk.globalRootHeading],
    );
    const result: RuntimeGenerationResult = {
      seed: this.#random.seed,
      prompt: options.prompt,
      fps: dimensions.fps,
      startFrame,
      frameCount: writtenFrames,
      chunks: chunks.length,
      motion,
      motionShape: [1, writtenFrames, dimensions.motion_dim],
      joints,
      jointsShape: [1, writtenFrames, dimensions.num_joints, 3],
      continuation: this.continuation(),
      timingsMs: {
        total: now() - totalStart,
        text: encoded.elapsed,
        denoising: denoisingMs,
        decoding: decodingMs,
      },
    };
    if (localRotationChunks.length === chunks.length) {
      result.localRotations = concatFloat(localRotationChunks);
      result.localRotationsShape = [
        1,
        writtenFrames,
        dimensions.num_joints,
        3,
        3,
      ];
    }
    if (globalRotationChunks.length === chunks.length) {
      result.globalRotations = concatFloat(globalRotationChunks);
      result.globalRotationsShape = [
        1,
        writtenFrames,
        dimensions.num_joints,
        3,
        3,
      ];
    }
    if (rootChunks.length === chunks.length) {
      result.rootPositions = concatFloat(rootChunks);
      result.rootPositionsShape = [1, writtenFrames, 3];
    }
    if (contactChunks.length === chunks.length) {
      result.footContacts = concatBytes(contactChunks);
      result.footContactsShape = [1, writtenFrames, 4];
    }
    if (headingChunks.length === chunks.length) {
      result.globalRootHeading = concatFloat(headingChunks);
      result.globalRootHeadingShape = [1, writtenFrames, 2];
    }
    return result;
  }
}

export class BrowserArdyRuntime {
  readonly manifest: BrowserModelPackManifest;
  readonly #tokenizer: LocalTokenizer;
  readonly #sessions: RuntimeSessions;
  #disposed = false;

  private constructor(
    manifest: BrowserModelPackManifest,
    tokenizer: LocalTokenizer,
    sessions: RuntimeSessions,
  ) {
    this.manifest = manifest;
    this.#tokenizer = tokenizer;
    this.#sessions = sessions;
  }

  static async create(
    pack: ModelPack,
    options: RuntimeLoadOptions = {},
  ): Promise<BrowserArdyRuntime> {
    throwIfCancelled(options.signal);
    options.onProgress?.({
      stage: "loading-tokenizer",
      completed: 0,
      total: 1,
    });
    const tokenizer = await LocalTokenizer.create(pack);
    throwIfCancelled(options.signal);
    options.onProgress?.({
      stage: "loading-tokenizer",
      completed: 1,
      total: 1,
    });
    try {
      const sessions = await createRuntimeSessions(
        pack,
        (completed, total, message) => {
          throwIfCancelled(options.signal);
          options.onProgress?.({
            stage: "loading-sessions",
            completed,
            total,
            message,
          });
        },
      );
      if (options.signal?.aborted) {
        await disposeRuntimeSessions(sessions);
        throw new RuntimeCancelledError("Model loading cancelled");
      }
      return new BrowserArdyRuntime(pack.manifest, tokenizer, sessions);
    } catch (error) {
      await tokenizer.dispose();
      throw error;
    }
  }

  createGenerationSession(
    options: RuntimeSessionOptions,
  ): BrowserArdyGenerationSession {
    if (this.#disposed) {
      throw new Error("Runtime has been disposed");
    }
    return new BrowserArdyGenerationSession(
      this.manifest,
      this.#tokenizer,
      this.#sessions,
      options,
    );
  }

  async generate(options: RuntimeGenerateOptions): Promise<RuntimeGenerationResult> {
    const frameCount = resolveFrameCount(this.manifest, options);
    const session = this.createGenerationSession({
      seed: options.seed,
      initialTranslation: options.initialTranslation,
      initialHeading: options.initialHeading,
    });
    return session.generate({
      prompt: options.prompt,
      durationFrames: frameCount,
      cfgWeight: options.cfgWeight,
      historyFrames: options.historyFrames,
      signal: options.signal,
      onProgress: options.onProgress,
      onChunk: options.onChunk,
    });
  }

  async dispose(): Promise<void> {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    await Promise.all([
      this.#tokenizer.dispose(),
      disposeRuntimeSessions(this.#sessions),
    ]);
  }
}

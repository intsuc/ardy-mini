// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type { BrowserModelPackManifest } from "./manifest";
import type { ModelPack } from "./model-pack";
import type {
  RuntimeBackend,
  RuntimeBackendPreference,
  RuntimeProgressStage,
} from "./protocol";
import { applyDdimStepInPlace, ddimStepForIndex } from "./ddim";
import { PortableRandom } from "./random";
import {
  createRuntimeSessions,
  disposeRuntimeSessions,
  ort,
  type RuntimeSessions,
} from "./sessions";
import { LocalTokenizer } from "./tokenizer";
import {
  copyTailHistory,
  createArWindow,
  createMotionPadMask,
  decoderValidTokensForFrames,
  recenterAndRequantize,
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
  backend?: RuntimeBackendPreference;
  wasmPaths?: string;
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeProgress) => void;
}

export interface RuntimeGenerateOptions {
  prompt: string;
  seed: number | string;
  durationFrames?: number;
  durationSeconds?: number;
  cfgWeight?: number;
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeProgress) => void;
}

export interface RuntimeGenerationResult {
  seed: number;
  prompt: string;
  backend: RuntimeBackend;
  fps: number;
  frameCount: number;
  motion: Float32Array;
  motionShape: [1, number, number];
  joints: Float32Array;
  jointsShape: [1, number, number, 3];
  timingsMs: {
    total: number;
    text: number;
    denoising: number;
    decoding: number;
  };
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

function int64Scalar(value: number): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.of(BigInt(value)), [1]);
}

function floatScalar(value: number): ort.Tensor {
  return new ort.Tensor("float32", Float32Array.of(value), [1]);
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

export class BrowserArdyRuntime {
  readonly manifest: BrowserModelPackManifest;
  readonly backend: RuntimeBackend;
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
    this.backend = sessions.backend;
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
        options.backend,
        options.wasmPaths,
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

  async generate(options: RuntimeGenerateOptions): Promise<RuntimeGenerationResult> {
    if (this.#disposed) {
      throw new Error("Runtime has been disposed");
    }
    if (options.prompt.trim().length === 0) {
      throw new TypeError("Prompt must not be empty");
    }
    if (
      options.cfgWeight !== undefined &&
      (!Number.isFinite(options.cfgWeight) || options.cfgWeight <= 0)
    ) {
      throw new RangeError("CFG weight must be positive and finite");
    }
    if (
      typeof options.seed === "number" &&
      !Number.isFinite(options.seed)
    ) {
      throw new RangeError("Numeric seed must be finite");
    }
    const frameCount = resolveFrameCount(this.manifest, options);
    const { dimensions, graphs, diffusion, generation } = this.manifest;
    const windowCount = Math.ceil(frameCount / dimensions.generation_frames);
    const totalStart = now();

    throwIfCancelled(options.signal);
    options.onProgress?.({
      stage: "encoding-text",
      completed: 0,
      total: 1,
    });
    const textStart = now();
    const encoded = await this.#tokenizer.encode(options.prompt);
    throwIfCancelled(options.signal);
    const textInputs = graphs.text_encoder.inputs;
    const textOutputs = await this.#sessions.textEncoder.run({
      [textInputs.inputIds]: new ort.Tensor(
        "int64",
        encoded.inputIds,
        [1, encoded.sequenceLength],
      ),
      [textInputs.attentionMask]: new ort.Tensor(
        "int64",
        encoded.attentionMask,
        [1, encoded.sequenceLength],
      ),
      [textInputs.tokenTypeIds]: new ort.Tensor(
        "int64",
        encoded.tokenTypeIds,
        [1, encoded.sequenceLength],
      ),
    });
    throwIfCancelled(options.signal);
    const textConditions = tensorFrom(
      textOutputs,
      graphs.text_encoder.outputs.textConditions,
      dimensions.text_condition_dim,
    );
    const textMs = now() - textStart;
    options.onProgress?.({
      stage: "encoding-text",
      completed: 1,
      total: 1,
    });

    const random = new PortableRandom(options.seed);
    const motion = new Float32Array(frameCount * dimensions.motion_dim);
    const joints = new Float32Array(
      frameCount * dimensions.num_joints * 3,
    );
    let history: Float32Array | undefined;
    let globalTranslation: Float32Array = new Float32Array(3);
    let firstHeadingAngle = 0;
    let writtenFrames = 0;
    let denoisingMs = 0;
    let decodingMs = 0;

    for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
      throwIfCancelled(options.signal);
      const window = createArWindow(dimensions, random, history);
      const framesToCopy = Math.min(
        dimensions.generation_frames,
        frameCount - writtenFrames,
      );
      const generationStart =
        window.generationTokenOffset * dimensions.hybrid_dim;
      const generationEnd =
        (window.generationTokenOffset + window.generationTokens) *
        dimensions.hybrid_dim;
      const denoiserInputs = graphs.denoiser.inputs;
      const timestepData = BigInt64Array.of(0n);
      const denoiserFeeds: Record<string, ort.Tensor> = {
        [denoiserInputs.cfgWeight]: floatScalar(
          options.cfgWeight ?? generation.default_cfg_weight,
        ),
        [denoiserInputs.x]: new ort.Tensor(
          "float32",
          window.x,
          [1, dimensions.max_tokens, dimensions.hybrid_dim],
        ),
        [denoiserInputs.historyLength]: int64Scalar(window.historyFrames),
        [denoiserInputs.generationLength]: int64Scalar(window.generationFrames),
        [denoiserInputs.historyMask]: new ort.Tensor(
          "float32",
          window.historyMask,
          [1, dimensions.max_frames],
        ),
        [denoiserInputs.generationMask]: new ort.Tensor(
          "float32",
          window.generationMask,
          [1, dimensions.max_frames],
        ),
        [denoiserInputs.historyTokenMask]: new ort.Tensor(
          "float32",
          window.historyTokenMask,
          [1, dimensions.max_tokens],
        ),
        [denoiserInputs.generationTokenMask]: new ort.Tensor(
          "float32",
          window.generationTokenMask,
          [1, dimensions.max_tokens],
        ),
        [denoiserInputs.textConditions]: textConditions,
        [denoiserInputs.timestep]: new ort.Tensor(
          "int64",
          timestepData,
          [1],
        ),
        [denoiserInputs.firstHeadingAngle]: floatScalar(firstHeadingAngle),
      };

      for (
        let inferenceIndex = 0;
        inferenceIndex < diffusion.timesteps.length;
        inferenceIndex += 1
      ) {
        throwIfCancelled(options.signal);
        const step = ddimStepForIndex(
          diffusion.timesteps,
          diffusion.alphas_cumprod,
          diffusion.alphas_cumprod_prev,
          inferenceIndex,
        );
        timestepData[0] = BigInt(step.timestep);
        const denoisingStart = now();
        const outputs = await this.#sessions.denoiser.run(denoiserFeeds);
        throwIfCancelled(options.signal);
        const predictionName = graphs.denoiser.outputs.predX0;
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
        denoisingMs += now() - denoisingStart;
        options.onProgress?.({
          stage: "denoising",
          completed:
            windowIndex * diffusion.timesteps.length + inferenceIndex + 1,
          total: windowCount * diffusion.timesteps.length,
          message: `window ${windowIndex + 1}/${windowCount}`,
        });
      }

      const validTokens = window.historyTokens + window.generationTokens;
      const nextHistoryStart = validTokens - dimensions.history_tokens;
      const recentered = recenterAndRequantize(
        window.x,
        validTokens,
        dimensions,
        this.manifest.recenter,
        this.manifest.latent_quantization,
        globalTranslation,
        nextHistoryStart,
      );
      globalTranslation = recentered.globalTranslation;
      firstHeadingAngle = recentered.firstHeadingAngle;

      throwIfCancelled(options.signal);
      const decodingStart = now();
      const decoderInputs = graphs.decoder.inputs;
      const decoderValidTokens = decoderValidTokensForFrames(
        dimensions,
        window.historyTokens,
        framesToCopy,
      );
      const decoded = await this.#sessions.decoder.run({
        [decoderInputs.hybridTokens]: new ort.Tensor(
          "float32",
          window.x,
          [1, dimensions.max_tokens, dimensions.hybrid_dim],
        ),
        [decoderInputs.motionPadMask]: new ort.Tensor(
          "float32",
          createMotionPadMask(dimensions, decoderValidTokens),
          [1, dimensions.max_frames],
        ),
        [decoderInputs.globalTranslation]: new ort.Tensor(
          "float32",
          globalTranslation,
          [1, 3],
        ),
      });
      throwIfCancelled(options.signal);
      const motionName = graphs.decoder.outputs.normalizedMotion;
      const jointsName = graphs.decoder.outputs.posedJoints;
      const decodedMotion = floatData(
        tensorFrom(
          decoded,
          motionName,
          dimensions.max_frames * dimensions.motion_dim,
        ),
        motionName,
      );
      const decodedJoints = floatData(
        tensorFrom(
          decoded,
          jointsName,
          dimensions.max_frames * dimensions.num_joints * 3,
        ),
        jointsName,
      );
      decodingMs += now() - decodingStart;

      const generatedFrameOffset = window.historyFrames;
      const motionSourceStart =
        generatedFrameOffset * dimensions.motion_dim;
      const motionSourceEnd =
        motionSourceStart + framesToCopy * dimensions.motion_dim;
      motion.set(
        decodedMotion.subarray(motionSourceStart, motionSourceEnd),
        writtenFrames * dimensions.motion_dim,
      );
      const jointStride = dimensions.num_joints * 3;
      const jointsSourceStart = generatedFrameOffset * jointStride;
      const jointsSourceEnd = jointsSourceStart + framesToCopy * jointStride;
      joints.set(
        decodedJoints.subarray(jointsSourceStart, jointsSourceEnd),
        writtenFrames * jointStride,
      );
      writtenFrames += framesToCopy;
      history = copyTailHistory(window.x, validTokens, dimensions);
      options.onProgress?.({
        stage: "decoding",
        completed: windowIndex + 1,
        total: windowCount,
      });
    }

    return {
      seed: random.seed,
      prompt: options.prompt,
      backend: this.backend,
      fps: dimensions.fps,
      frameCount,
      motion,
      motionShape: [1, frameCount, dimensions.motion_dim],
      joints,
      jointsShape: [1, frameCount, dimensions.num_joints, 3],
      timingsMs: {
        total: now() - totalStart,
        text: textMs,
        denoising: denoisingMs,
        decoding: decodingMs,
      },
    };
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

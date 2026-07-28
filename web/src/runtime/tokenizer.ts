// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { Tokenizer } from "@huggingface/tokenizers";

import type { ModelPack } from "./model-pack";

export interface EncodedPrompt {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  tokenTypeIds: BigInt64Array;
  sequenceLength: number;
}

function parseJson(bytes: Uint8Array, path: string): object {
  try {
    const value: unknown = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new TypeError("root must be an object");
    }
    return value;
  } catch (error) {
    throw new Error(
      `Could not parse ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function toBigInt(values: readonly number[]): BigInt64Array {
  const result = new BigInt64Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    result[index] = BigInt(values[index]);
  }
  return result;
}

/**
 * Keep the final post-processor token (BERT's `[SEP]`) when a MiniLM prompt
 * exceeds its fixed context. This matches single-sequence right truncation.
 */
function truncateWithFinalSpecialToken(
  values: readonly number[],
  maxLength: number,
): number[] {
  if (values.length <= maxLength) {
    return [...values];
  }
  if (maxLength === 1) {
    return [values[values.length - 1]];
  }
  return [...values.slice(0, maxLength - 1), values[values.length - 1]];
}

export class LocalTokenizer {
  readonly #tokenizer: Tokenizer;
  readonly #maxLength: number;

  private constructor(tokenizer: Tokenizer, maxLength: number) {
    this.#tokenizer = tokenizer;
    this.#maxLength = maxLength;
  }

  static async create(pack: ModelPack): Promise<LocalTokenizer> {
    const { directory, max_length: maxLength } = pack.manifest.tokenizer;
    const tokenizerJsonPath = `${directory}/tokenizer.json`;
    const tokenizerConfigPath = `${directory}/tokenizer_config.json`;
    try {
      const [tokenizerJson, tokenizerConfig] = await Promise.all([
        pack.read(tokenizerJsonPath),
        pack.read(tokenizerConfigPath),
      ]);
      return new LocalTokenizer(
        new Tokenizer(
          parseJson(tokenizerJson, tokenizerJsonPath),
          parseJson(tokenizerConfig, tokenizerConfigPath),
        ),
        maxLength,
      );
    } finally {
      for (const path of Object.keys(pack.manifest.files)) {
        if (path.startsWith(`${directory}/`)) {
          pack.release(path);
        }
      }
    }
  }

  async encode(prompt: string): Promise<EncodedPrompt> {
    const encoded = this.#tokenizer.encode(prompt, {
      add_special_tokens: true,
      return_token_type_ids: true,
    });
    const inputIds = truncateWithFinalSpecialToken(
      encoded.ids,
      this.#maxLength,
    );
    const attentionMask = truncateWithFinalSpecialToken(
      encoded.attention_mask,
      this.#maxLength,
    );
    const tokenTypeIds = truncateWithFinalSpecialToken(
      encoded.token_type_ids ?? new Array(encoded.ids.length).fill(0),
      this.#maxLength,
    );
    if (
      inputIds.length === 0 ||
      inputIds.length > this.#maxLength ||
      attentionMask.length !== inputIds.length ||
      tokenTypeIds.length !== inputIds.length
    ) {
      throw new RangeError("Tokenizer produced invalid sequence dimensions");
    }
    return {
      inputIds: toBigInt(inputIds),
      attentionMask: toBigInt(attentionMask),
      tokenTypeIds: toBigInt(tokenTypeIds),
      sequenceLength: inputIds.length,
    };
  }

  async dispose(): Promise<void> {
    // Tokenizers.js is pure JavaScript and owns no native/GPU resources.
  }
}

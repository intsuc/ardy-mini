// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { ModelPack } from "./model-pack";
import { LocalTokenizer } from "./tokenizer";

const tokenRecord = (id: number, content: string) => ({
  id,
  content,
  single_word: false,
  lstrip: false,
  rstrip: false,
  normalized: false,
  special: true,
});

function miniLmTokenizerJson(): object {
  return {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
      tokenRecord(0, "[PAD]"),
      tokenRecord(100, "[UNK]"),
      tokenRecord(101, "[CLS]"),
      tokenRecord(102, "[SEP]"),
      tokenRecord(103, "[MASK]"),
    ],
    normalizer: {
      type: "BertNormalizer",
      clean_text: true,
      handle_chinese_chars: true,
      strip_accents: null,
      lowercase: true,
    },
    pre_tokenizer: { type: "BertPreTokenizer" },
    post_processor: {
      type: "TemplateProcessing",
      single: [
        { SpecialToken: { id: "[CLS]", type_id: 0 } },
        { Sequence: { id: "A", type_id: 0 } },
        { SpecialToken: { id: "[SEP]", type_id: 0 } },
      ],
      pair: [],
      special_tokens: {
        "[CLS]": { id: "[CLS]", ids: [101], tokens: ["[CLS]"] },
        "[SEP]": { id: "[SEP]", ids: [102], tokens: ["[SEP]"] },
      },
    },
    decoder: { type: "WordPiece", prefix: "##", cleanup: true },
    model: {
      type: "WordPiece",
      unk_token: "[UNK]",
      continuing_subword_prefix: "##",
      max_input_chars_per_word: 100,
      vocab: {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
        a: 1037,
        person: 2711,
        forward: 2830,
        walks: 7365,
        ".": 1012,
      },
    },
  };
}

function pack(maxLength = 128): ModelPack {
  const encoder = new TextEncoder();
  const entries = new Map<string, Uint8Array>([
    [
      "tokenizer/tokenizer.json",
      encoder.encode(JSON.stringify(miniLmTokenizerJson())),
    ],
    [
      "tokenizer/tokenizer_config.json",
      encoder.encode(
        JSON.stringify({
          tokenizer_class: "BertTokenizer",
          do_lower_case: true,
          cls_token: "[CLS]",
          sep_token: "[SEP]",
          unk_token: "[UNK]",
          pad_token: "[PAD]",
        }),
      ),
    ],
  ]);
  return {
    manifest: {
      tokenizer: { directory: "tokenizer", max_length: maxLength },
      files: Object.fromEntries([...entries].map(([path]) => [path, {}])),
    },
    read: async (path: string) => {
      const bytes = entries.get(path);
      if (bytes === undefined) throw new Error(`missing ${path}`);
      return bytes;
    },
    release: (path: string) => {
      entries.delete(path);
    },
  } as unknown as ModelPack;
}

describe("LocalTokenizer", () => {
  it("matches the all-MiniLM-L6-v2 golden IDs and masks", async () => {
    const tokenizer = await LocalTokenizer.create(pack());
    const output = await tokenizer.encode("A person walks forward.");
    expect([...output.inputIds]).toEqual([
      101n,
      1037n,
      2711n,
      7365n,
      2830n,
      1012n,
      102n,
    ]);
    expect([...output.attentionMask]).toEqual(new Array(7).fill(1n));
    expect([...output.tokenTypeIds]).toEqual(new Array(7).fill(0n));
  });

  it("right-truncates content while retaining MiniLM's final SEP token", async () => {
    const tokenizer = await LocalTokenizer.create(pack(5));
    const output = await tokenizer.encode("A person walks forward.");
    expect([...output.inputIds]).toEqual([101n, 1037n, 2711n, 7365n, 102n]);
  });
});

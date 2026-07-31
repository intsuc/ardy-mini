// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  deriveModelTermsUrl,
  resolveHostedModelFamilyBaseUrl,
  resolveModelTermsUrl,
} from "./deployment-config";

const PAGE_URL = "https://intsuc-ardy-mini.hf.space/";
const REVISION = "0123456789abcdef0123456789abcdef01234567";
const FAMILY_URL =
  `https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser/resolve/${REVISION}/`;

describe("hosted model deployment configuration", () => {
  it("prefers the runtime Static Space variable and normalizes a directory", () => {
    expect(
      resolveHostedModelFamilyBaseUrl({
        buildValue: "https://example.com/build/",
        pageUrl: PAGE_URL,
        spaceValue: FAMILY_URL.slice(0, -1),
      }),
    ).toBe(FAMILY_URL);
  });

  it("uses the Vite build value outside a configured Static Space", () => {
    expect(
      resolveHostedModelFamilyBaseUrl({
        buildValue: FAMILY_URL,
        pageUrl: PAGE_URL,
      }),
    ).toBe(FAMILY_URL);
    expect(
      resolveHostedModelFamilyBaseUrl({ pageUrl: PAGE_URL }),
    ).toBeNull();
  });

  it("rejects credentials, mutable query parameters, and insecure hosts", () => {
    for (const configured of [
      "https://token@example.com/models/",
      "https://example.com/models/?revision=main",
      "http://example.com/models/",
    ]) {
      expect(() =>
        resolveHostedModelFamilyBaseUrl({
          pageUrl: PAGE_URL,
          spaceValue: configured,
        }),
      ).toThrow();
    }
  });

  it("derives terms at the same immutable Hub revision as the model", () => {
    const expected =
      `https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser/blob/${REVISION}/MODEL_TERMS.md`;
    expect(deriveModelTermsUrl(FAMILY_URL)).toBe(expected);
    expect(
      resolveModelTermsUrl({
        pageUrl: PAGE_URL,
        spaceValue: FAMILY_URL,
      }),
    ).toBe(expected);
  });

  it("allows an explicit immutable terms URL to override derivation", () => {
    const explicit =
      `https://huggingface.co/intsuc/model/blob/${REVISION}/MODEL_TERMS.md`;
    expect(
      resolveModelTermsUrl({
        pageUrl: PAGE_URL,
        spaceTermsValue: explicit,
        spaceValue: FAMILY_URL,
      }),
    ).toBe(explicit);
  });

  it("keeps the public terms reachable when deployment configuration is invalid", () => {
    expect(
      resolveModelTermsUrl({
        pageUrl: PAGE_URL,
        spaceTermsValue: "javascript:alert(1)",
        spaceValue: "http://example.com/model/",
      }),
    ).toBe(
      "https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser/blob/main/MODEL_TERMS.md",
    );
  });
});

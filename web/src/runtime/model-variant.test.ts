// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  modelVariantBaseUrl,
  preferredModelVariant,
} from "./model-variant";

describe("automatic browser model selection", () => {
  it("prefers the smaller model when the inference adapter reports native FP16 shaders", () => {
    expect(preferredModelVariant(true)).toBe("fp16");
  });

  it("uses the FP32 model when the inference adapter reports no native FP16 shaders", () => {
    expect(preferredModelVariant(false)).toBe("fp32");
  });

  it("resolves each variant beneath one model-family URL", () => {
    expect(
      modelVariantBaseUrl(
        "https://models.example/ardy/browser-v1",
        "fp32",
      ),
    ).toBe("https://models.example/ardy/browser-v1/fp32/");
  });
});

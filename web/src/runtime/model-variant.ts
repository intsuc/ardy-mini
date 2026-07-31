// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export type BrowserModelVariant = "fp16" | "fp32";

export function preferredModelVariant(
  shaderF16: boolean,
): BrowserModelVariant {
  return shaderF16 ? "fp16" : "fp32";
}

export function modelVariantBaseUrl(
  familyBaseUrl: string,
  variant: BrowserModelVariant,
): string {
  const normalizedFamilyUrl = familyBaseUrl.endsWith("/")
    ? familyBaseUrl
    : `${familyBaseUrl}/`;
  return new URL(`${variant}/`, normalizedFamilyUrl).href;
}

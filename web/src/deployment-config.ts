// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

const PUBLIC_MODEL_REPOSITORY_ID =
  "intsuc/Llama-3-ARDY-Mini-Core40-Browser";
const PUBLIC_MODEL_REPOSITORY_URL =
  `https://huggingface.co/${PUBLIC_MODEL_REPOSITORY_ID}`;
const DEFAULT_MODEL_TERMS_URL =
  `${PUBLIC_MODEL_REPOSITORY_URL}/blob/main/MODEL_TERMS.md`;

interface HuggingFaceStaticSpaceState {
  huggingface?: {
    variables?: Readonly<Record<string, unknown>>;
  };
}

export interface HostedModelUrlOptions {
  buildValue?: string;
  pageUrl: string;
  spaceValue?: string;
}

function nonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function staticSpaceVariable(name: string): string | null {
  const host = globalThis as typeof globalThis &
    HuggingFaceStaticSpaceState;
  return nonEmptyString(host.huggingface?.variables?.[name]);
}

function isLoopbackHost(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]"
  );
}

function normalizedHttpUrl(
  value: string,
  pageUrl: string,
  { directory }: { directory: boolean },
): URL {
  const url = new URL(value, pageUrl);
  if (
    url.protocol !== "https:" &&
    !(url.protocol === "http:" && isLoopbackHost(url.hostname))
  ) {
    throw new TypeError("Hosted model URLs must use HTTPS.");
  }
  if (url.username || url.password) {
    throw new TypeError("Hosted model URLs must not contain credentials.");
  }
  if (url.search || url.hash) {
    throw new TypeError(
      "Hosted model URLs must not contain a query or fragment.",
    );
  }
  if (directory && !url.pathname.endsWith("/")) {
    url.pathname = `${url.pathname}/`;
  }
  return url;
}

export function resolveHostedModelFamilyBaseUrl({
  buildValue,
  pageUrl,
  spaceValue,
}: HostedModelUrlOptions): string | null {
  const configured = nonEmptyString(spaceValue) ?? nonEmptyString(buildValue);
  if (!configured) return null;
  return normalizedHttpUrl(configured, pageUrl, { directory: true }).href;
}

export function deriveModelTermsUrl(
  familyBaseUrl: string,
): string | null {
  const url = new URL(familyBaseUrl);
  if (url.protocol !== "https:" || url.hostname !== "huggingface.co") {
    return null;
  }
  const match = /^\/(.+\/[^/]+)\/resolve\/([^/]+)\/$/u.exec(
    url.pathname,
  );
  if (!match) return null;
  const [, repository, revision] = match;
  return `${url.origin}/${repository}/blob/${revision}/MODEL_TERMS.md`;
}

export interface ModelTermsUrlOptions extends HostedModelUrlOptions {
  buildTermsValue?: string;
  spaceTermsValue?: string;
}

export function resolveModelTermsUrl({
  buildTermsValue,
  buildValue,
  pageUrl,
  spaceTermsValue,
  spaceValue,
}: ModelTermsUrlOptions): string {
  const explicit =
    nonEmptyString(spaceTermsValue) ?? nonEmptyString(buildTermsValue);
  if (explicit) {
    try {
      return normalizedHttpUrl(explicit, pageUrl, { directory: false }).href;
    } catch {
      return DEFAULT_MODEL_TERMS_URL;
    }
  }

  let familyBaseUrl: string | null = null;
  try {
    familyBaseUrl = resolveHostedModelFamilyBaseUrl({
      buildValue,
      pageUrl,
      spaceValue,
    });
  } catch {
    // Model startup reports invalid deployment configuration separately. Keep
    // the legal notice usable instead of failing React rendering as well.
  }
  return (
    (familyBaseUrl ? deriveModelTermsUrl(familyBaseUrl) : null) ??
    DEFAULT_MODEL_TERMS_URL
  );
}

export {
  DEFAULT_MODEL_TERMS_URL,
  PUBLIC_MODEL_REPOSITORY_ID,
  PUBLIC_MODEL_REPOSITORY_URL,
};

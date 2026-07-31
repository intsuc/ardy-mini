// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

interface CooperativeScheduler {
  yield?: () => Promise<void>;
}

/** Yield long browser-only work so rendering and input can run first. */
export function yieldToMainThread(): Promise<void> {
  const scheduler = (
    globalThis as typeof globalThis & {
      scheduler?: CooperativeScheduler;
    }
  ).scheduler;
  if (scheduler?.yield) return scheduler.yield();
  return new Promise((resolve) => globalThis.setTimeout(resolve, 0));
}

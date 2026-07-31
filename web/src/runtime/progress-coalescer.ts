// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import type { RuntimeProgress } from "./engine";

export const DOWNLOAD_PROGRESS_INTERVAL_MS = 100;

export type RuntimeProgressReporter = (progress: RuntimeProgress) => void;

/**
 * Coalesces high-frequency download updates before they cross the Worker
 * boundary. It is deliberately timer-free: a later download callback emits
 * the retained leading-edge update, while a stage transition synchronously
 * flushes the latest download state before the new stage.
 */
export function createRuntimeProgressCoalescer(
  emit: RuntimeProgressReporter,
  now: () => number = () => performance.now(),
  intervalMs = DOWNLOAD_PROGRESS_INTERVAL_MS,
): RuntimeProgressReporter {
  if (!Number.isFinite(intervalMs) || intervalMs < 0) {
    throw new RangeError("Progress interval must be a non-negative number");
  }

  let pendingDownload: RuntimeProgress | undefined;
  let lastDownloadEmission: number | undefined;

  const emitPendingDownload = (emittedAt: number): void => {
    if (pendingDownload === undefined) return;
    const progress = pendingDownload;
    pendingDownload = undefined;
    lastDownloadEmission = emittedAt;
    emit(progress);
  };

  return (progress) => {
    const currentTime = now();
    if (progress.stage !== "downloading-model") {
      emitPendingDownload(currentTime);
      emit(progress);
      return;
    }

    pendingDownload = progress;
    if (
      lastDownloadEmission === undefined ||
      currentTime - lastDownloadEmission >= intervalMs
    ) {
      emitPendingDownload(currentTime);
    }
  };
}

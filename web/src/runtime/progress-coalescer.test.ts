// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import type { RuntimeProgress } from "./engine";
import {
  createRuntimeProgressCoalescer,
  DOWNLOAD_PROGRESS_INTERVAL_MS,
} from "./progress-coalescer";

function progress(
  stage: RuntimeProgress["stage"],
  completed: number,
  total = 100,
): RuntimeProgress {
  return { stage, completed, total };
}

describe("runtime progress coalescing", () => {
  it("emits download progress on the leading edge at most once per interval", () => {
    let time = 0;
    const emitted: RuntimeProgress[] = [];
    const report = createRuntimeProgressCoalescer(
      (value) => emitted.push(value),
      () => time,
    );

    report(progress("downloading-model", 1));
    time = DOWNLOAD_PROGRESS_INTERVAL_MS / 4;
    report(progress("downloading-model", 20));
    time = DOWNLOAD_PROGRESS_INTERVAL_MS - 1;
    report(progress("downloading-model", 40));
    time = DOWNLOAD_PROGRESS_INTERVAL_MS;
    report(progress("downloading-model", 60));

    expect(emitted).toEqual([
      progress("downloading-model", 1),
      progress("downloading-model", 60),
    ]);
  });

  it("flushes exact terminal download progress before a new stage", () => {
    let time = 0;
    const emitted: RuntimeProgress[] = [];
    const report = createRuntimeProgressCoalescer(
      (value) => emitted.push(value),
      () => time,
    );

    report(progress("downloading-model", 1));
    time = 10;
    report(progress("downloading-model", 25));
    time = 20;
    report(progress("downloading-model", 100));
    time = 21;
    report(progress("loading-tokenizer", 0, 1));

    expect(emitted).toEqual([
      progress("downloading-model", 1),
      progress("downloading-model", 100),
      progress("loading-tokenizer", 0, 1),
    ]);
  });

  it("collapses a large same-tick chunk burst instead of queueing stale progress", () => {
    const emitted: RuntimeProgress[] = [];
    const report = createRuntimeProgressCoalescer(
      (value) => emitted.push(value),
      () => 0,
    );

    for (let completed = 1; completed <= 10_000; completed += 1) {
      report(progress("downloading-model", completed, 10_000));
    }
    report(progress("loading-tokenizer", 0, 1));

    expect(emitted).toEqual([
      progress("downloading-model", 1, 10_000),
      progress("downloading-model", 10_000, 10_000),
      progress("loading-tokenizer", 0, 1),
    ]);
  });

  it("emits every non-download update immediately", () => {
    const emitted: RuntimeProgress[] = [];
    const report = createRuntimeProgressCoalescer(
      (value) => emitted.push(value),
      () => 0,
    );

    report(progress("loading-tokenizer", 0, 1));
    report(progress("loading-tokenizer", 1, 1));
    report(progress("loading-sessions", 1, 3));
    report(progress("loading-sessions", 2, 3));

    expect(emitted).toEqual([
      progress("loading-tokenizer", 0, 1),
      progress("loading-tokenizer", 1, 1),
      progress("loading-sessions", 1, 3),
      progress("loading-sessions", 2, 3),
    ]);
  });

  it("does not duplicate a terminal update already emitted on an interval", () => {
    let time = 0;
    const emitted: RuntimeProgress[] = [];
    const report = createRuntimeProgressCoalescer(
      (value) => emitted.push(value),
      () => time,
    );

    report(progress("downloading-model", 1));
    time = DOWNLOAD_PROGRESS_INTERVAL_MS;
    report(progress("downloading-model", 100));
    time += 1;
    report(progress("verifying-model", 1, 5));

    expect(emitted).toEqual([
      progress("downloading-model", 1),
      progress("downloading-model", 100),
      progress("verifying-model", 1, 5),
    ]);
  });
});

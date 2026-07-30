// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";

import { BoundProgress } from "./control-bindings";
import {
  clearModelCacheAction,
  generationProgressControl,
  modelDownloadAction,
  modelUiControl,
  useModelUiState,
} from "../ui-control-store";

(globalThis as typeof globalThis & {
  IS_REACT_ACT_ENVIRONMENT: boolean;
}).IS_REACT_ACT_ENVIRONMENT = true;

describe("BoundProgress", () => {
  it("keeps Base UI visual and accessibility state synchronized", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);

    try {
      await act(async () => {
        root.render(
          <BoundProgress
            control={generationProgressControl}
            aria-label="Generation progress"
          />,
        );
      });

      await act(async () => {
        generationProgressControl.commit(42);
      });

      const progress = container.querySelector<HTMLElement>(
        "#generation-progressbar",
      );
      const indicator = progress?.querySelector<HTMLElement>(
        '[data-slot="progress-indicator"]',
      );

      expect(progress?.getAttribute("aria-valuenow")).toBe("42");
      expect(progress?.getAttribute("aria-valuetext")).toBe("42%");
      expect(progress?.hasAttribute("data-progressing")).toBe(true);
      expect(indicator?.style.width).toBe("42%");

      await act(async () => {
        generationProgressControl.commit(100);
      });

      expect(progress?.getAttribute("aria-valuenow")).toBe("100");
      expect(progress?.getAttribute("aria-valuetext")).toBe("100%");
      expect(progress?.hasAttribute("data-complete")).toBe(true);
      expect(progress?.hasAttribute("data-progressing")).toBe(false);
    } finally {
      await act(async () => {
        generationProgressControl.commit(0);
        root.unmount();
      });
    }
  });
});

function ModelStateProbe() {
  const state = useModelUiState();
  return (
    <output>
      {state.cache}:{state.runtime}:{state.cachedFiles}/{state.totalFiles}:
      {state.cachedBytes}/{state.totalBytes}:
      {String(state.downloadDialogOpen)}
    </output>
  );
}

describe("model UI control", () => {
  it("keeps cache progress, runtime state, and the download dialog coherent", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);

    try {
      modelUiControl.dispatch({ type: "reset" });
      await act(async () => {
        root.render(<ModelStateProbe />);
      });

      await act(async () => {
        modelUiControl.dispatch({
          type: "cache-missing",
          cachedFiles: 1,
          totalFiles: 4,
          cachedBytes: 250,
          totalBytes: 1_000,
          showDownloadPrompt: true,
        });
      });
      expect(container.textContent).toBe(
        "missing:idle:1/4:250/1000:true",
      );

      await act(async () => {
        modelUiControl.dispatch({ type: "download-started" });
        modelUiControl.dispatch({
          type: "download-progress",
          cachedFiles: 8,
          totalFiles: 4,
          cachedBytes: 2_000,
          totalBytes: 1_000,
        });
      });
      expect(container.textContent).toBe(
        "downloading:idle:4/4:1000/1000:false",
      );

      await act(async () => {
        modelUiControl.dispatch({ type: "verification-started" });
        modelUiControl.dispatch({
          type: "cache-ready",
          totalFiles: 4,
          totalBytes: 1_000,
        });
        modelUiControl.dispatch({ type: "runtime-ready" });
      });
      expect(container.textContent).toBe(
        "ready:ready:4/4:1000/1000:false",
      );

      await act(async () => {
        modelUiControl.dispatch({ type: "clear-started" });
        modelUiControl.dispatch({ type: "cache-cleared" });
      });
      expect(container.textContent).toBe(
        "missing:ready:0/4:0/1000:false",
      );
    } finally {
      await act(async () => {
        modelUiControl.dispatch({ type: "reset" });
        root.unmount();
      });
    }
  });

  it("ignores invalid prompt transitions and exposes explicit actions", () => {
    modelUiControl.dispatch({ type: "reset" });
    modelUiControl.dispatch({ type: "download-prompt-opened" });
    expect(modelUiControl.getSnapshot().downloadDialogOpen).toBe(false);

    let downloads = 0;
    let clears = 0;
    const stopDownload = modelDownloadAction.onTrigger(() => {
      downloads += 1;
    });
    const stopClear = clearModelCacheAction.onTrigger(() => {
      clears += 1;
    });

    try {
      modelDownloadAction.trigger();
      clearModelCacheAction.trigger();
      expect(downloads).toBe(1);
      expect(clears).toBe(1);
    } finally {
      stopDownload();
      stopClear();
      modelUiControl.dispatch({ type: "reset" });
    }
  });

  it("represents discovery failures and allows partial caches to clear", () => {
    modelUiControl.dispatch({ type: "reset" });
    modelUiControl.dispatch({
      type: "cache-error",
      operation: "download",
    });
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "error",
      errorOperation: "download",
    });

    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 2,
      totalFiles: 4,
      cachedBytes: 500,
      totalBytes: 1_000,
      showDownloadPrompt: false,
    });
    modelUiControl.dispatch({ type: "clear-started" });
    expect(modelUiControl.getSnapshot().cache).toBe("clearing");
    modelUiControl.dispatch({ type: "cache-cleared" });
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "missing",
      cachedFiles: 0,
      cachedBytes: 0,
    });

    modelUiControl.dispatch({ type: "reset" });
  });
});

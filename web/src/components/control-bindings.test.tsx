// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it } from "vitest";

import { BoundProgress } from "./control-bindings";
import { generationProgressControl } from "../ui-control-store";

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

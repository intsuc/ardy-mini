// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { createRoot } from "react-dom/client";

import { App } from "./app";
import "./index.css";
import "./style.css";

const container = document.getElementById("root");
if (!(container instanceof HTMLElement)) {
  throw new Error("Missing ARDY application root.");
}

const root = createRoot(container);
root.render(<App />);

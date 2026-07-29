// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";

import { App } from "./app";
import "./index.css";
import "./style.css";
import { bootstrap } from "./main";

const container = document.getElementById("root");
if (!(container instanceof HTMLElement)) {
  throw new Error("Missing ARDY application root.");
}

const root = createRoot(container);
flushSync(() => {
  root.render(<App />);
});
bootstrap();

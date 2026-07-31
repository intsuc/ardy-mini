// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

const crossOriginIsolationHeaders = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
};

const developmentModelUrl = "/models/ardy-minilm-core40-browser-v1/";
const defaultDevelopmentModelDirectory = new URL(
  "../artifacts/browser/ardy-minilm-core40-browser-v1/",
  import.meta.url,
).pathname;

const modelContentTypes: Readonly<Record<string, string>> = {
  ".gz": "application/gzip",
  ".json": "application/json; charset=utf-8",
};

type ModelFileRange =
  | { kind: "full" }
  | { kind: "partial"; start: number; end: number }
  | { kind: "unsatisfiable" };

function modelFileRange(
  header: string | string[] | undefined,
  size: number,
): ModelFileRange {
  if (header === undefined) return { kind: "full" };
  if (Array.isArray(header)) return { kind: "unsatisfiable" };
  const match = /^bytes=(\d+)-$/i.exec(header.trim());
  if (match === null) return { kind: "unsatisfiable" };
  const start = Number(match[1]);
  if (!Number.isSafeInteger(start) || start >= size) {
    return { kind: "unsatisfiable" };
  }
  return { kind: "partial", start, end: size - 1 };
}

function developmentModelFiles(): Plugin {
  return {
    name: "ardy-development-model-files",
    apply: "serve",
    configureServer(server) {
      const configuredDirectory =
        process.env.ARDY_BROWSER_MODEL_DIR ?? defaultDevelopmentModelDirectory;
      const modelDirectory = resolve(configuredDirectory);

      server.middlewares.use(developmentModelUrl, async (request, response, next) => {
        if (request.method !== "GET" && request.method !== "HEAD") {
          response.statusCode = 405;
          response.setHeader("Allow", "GET, HEAD");
          response.end();
          return;
        }

        try {
          const requestUrl = new URL(request.url ?? "", "http://localhost");
          const relativePath = decodeURIComponent(requestUrl.pathname).replace(
            /^\/+/,
            "",
          );
          if (
            relativePath.length === 0 ||
            relativePath.includes("\\") ||
            relativePath.split("/").some((part) => part === "..")
          ) {
            response.statusCode = 404;
            response.end();
            return;
          }

          const candidate = resolve(modelDirectory, relativePath);
          if (
            candidate !== modelDirectory &&
            !candidate.startsWith(`${modelDirectory}${sep}`)
          ) {
            response.statusCode = 404;
            response.end();
            return;
          }

          const [canonicalDirectory, canonicalFile] = await Promise.all([
            realpath(modelDirectory),
            realpath(candidate),
          ]);
          if (
            canonicalFile !== canonicalDirectory &&
            !canonicalFile.startsWith(`${canonicalDirectory}${sep}`)
          ) {
            response.statusCode = 404;
            response.end();
            return;
          }

          const fileStats = await stat(canonicalFile);
          if (!fileStats.isFile()) {
            response.statusCode = 404;
            response.end();
            return;
          }

          response.setHeader(
            "Content-Type",
            modelContentTypes[extname(canonicalFile)] ??
              "application/octet-stream",
          );
          response.setHeader("Cache-Control", "no-store");
          response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
          response.setHeader("Accept-Ranges", "bytes");
          const range = modelFileRange(
            request.method === "GET" ? request.headers.range : undefined,
            fileStats.size,
          );
          if (range.kind === "unsatisfiable") {
            response.statusCode = 416;
            response.setHeader("Content-Range", `bytes */${fileStats.size}`);
            response.setHeader("Content-Length", 0);
            response.end();
            return;
          }

          const start = range.kind === "partial" ? range.start : 0;
          const end =
            range.kind === "partial" ? range.end : fileStats.size - 1;
          response.statusCode = range.kind === "partial" ? 206 : 200;
          response.setHeader("Content-Length", end - start + 1);
          if (range.kind === "partial") {
            response.setHeader(
              "Content-Range",
              `bytes ${start}-${end}/${fileStats.size}`,
            );
          }
          if (request.method === "HEAD" || fileStats.size === 0) {
            response.end();
            return;
          }
          const stream =
            range.kind === "partial"
              ? createReadStream(canonicalFile, { start, end })
              : createReadStream(canonicalFile);
          stream.pipe(response);
        } catch (error) {
          const code =
            typeof error === "object" &&
            error !== null &&
            "code" in error
              ? error.code
              : undefined;
          if (code === "ENOENT" || code === "ENOTDIR") {
            response.statusCode = 404;
            response.end();
            return;
          }
          next(error);
        }
      });
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [developmentModelFiles(), react(), tailwindcss()],
  resolve: {
    alias: {
      "@": new URL("./src", import.meta.url).pathname,
    },
    conditions: ["onnxruntime-web-use-extern-wasm"],
  },
  build: {
    license: {
      fileName: "third-party-licenses.md",
    },
    target: "es2022",
    sourcemap: true,
  },
  server: {
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    restoreMocks: true,
  },
});

// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import { chmod, mkdir } from "node:fs/promises";
import { isIP } from "node:net";
import { hostname, networkInterfaces } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const webDirectory = fileURLToPath(new URL("../", import.meta.url));
const certificateDirectory = resolve(webDirectory, ".cert");
const certificatePath = resolve(
  certificateDirectory,
  "ardy-mini-dev.pem",
);
const privateKeyPath = resolve(
  certificateDirectory,
  "ardy-mini-dev-key.pem",
);

function runMkcert(args, { capture = false } = {}) {
  const result = spawnSync("mkcert", args, {
    encoding: "utf8",
    stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
  });
  if (result.error?.code === "ENOENT") {
    throw new Error(
      "mkcert is required. Install it from https://github.com/FiloSottile/mkcert and run this command again.",
    );
  }
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = capture ? result.stderr.trim() : "";
    throw new Error(
      `mkcert ${args[0] ?? ""} failed with exit code ${result.status}.${detail ? ` ${detail}` : ""}`,
    );
  }
  return capture ? result.stdout.trim() : "";
}

function normalizedAddress(address) {
  const scopeSeparator = address.indexOf("%");
  return scopeSeparator === -1
    ? address
    : address.slice(0, scopeSeparator);
}

function networkHosts() {
  const hosts = [];
  for (const entries of Object.values(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.internal) continue;
      const address = normalizedAddress(entry.address);
      if (
        entry.family === "IPv6" &&
        address.toLowerCase().startsWith("fe80:")
      ) {
        continue;
      }
      hosts.push(address);
    }
  }
  return hosts;
}

function extraHosts() {
  const hosts = (process.env.ARDY_DEV_HTTPS_HOSTS ?? "")
    .split(",")
    .map((host) => host.trim())
    .filter(Boolean);
  for (const host of hosts) {
    if (!isCertificateHost(host)) {
      throw new Error(
        `ARDY_DEV_HTTPS_HOSTS contains an invalid hostname or IP address: ${JSON.stringify(host)}`,
      );
    }
  }
  return hosts;
}

function isCertificateHost(value) {
  if (isIP(value) !== 0) return true;
  if (value.length === 0 || value.length > 253) return false;
  return value.split(".").every(
    (label) =>
      label.length > 0 &&
      label.length <= 63 &&
      /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/i.test(label),
  );
}

function certificateHosts() {
  const localHostname = hostname().trim();
  const hosts = new Set(["localhost", "127.0.0.1", "::1"]);
  if (isCertificateHost(localHostname)) {
    hosts.add(localHostname);
    if (!localHostname.includes(".")) {
      hosts.add(`${localHostname}.local`);
    }
  }
  for (const host of [...networkHosts(), ...extraHosts()]) {
    hosts.add(host);
  }
  return [...hosts];
}

function httpsUrl(host) {
  return host.includes(":")
    ? `https://[${host}]:5173/`
    : `https://${host}:5173/`;
}

async function main() {
  const hosts = certificateHosts();
  await mkdir(certificateDirectory, { recursive: true });

  runMkcert(["-install"]);
  runMkcert([
    "-cert-file",
    certificatePath,
    "-key-file",
    privateKeyPath,
    ...hosts,
  ]);
  if (process.platform !== "win32") {
    await chmod(privateKeyPath, 0o600);
  }

  const caRoot = runMkcert(["-CAROOT"], { capture: true });
  const lanHosts = networkHosts();
  console.log("\nARDY Mini HTTPS development certificate is ready.");
  console.log(`Certificate: ${certificatePath}`);
  console.log(`Private key: ${privateKeyPath}`);
  console.log(`Mobile root CA: ${resolve(caRoot, "rootCA.pem")}`);
  console.log("\nRun npm run dev, then open one of these URLs:");
  console.log(`  ${httpsUrl("localhost")}`);
  for (const host of lanHosts) {
    console.log(`  ${httpsUrl(host)}`);
  }
  if (lanHosts.length === 0) {
    console.log(
      "  No non-loopback network address was detected. Add one with ARDY_DEV_HTTPS_HOSTS and rerun setup.",
    );
  }
  console.log(
    "\nInstall only rootCA.pem on the test device. Never copy or share rootCA-key.pem.",
  );
}

main().catch((error) => {
  console.error(
    error instanceof Error ? error.message : String(error),
  );
  process.exitCode = 1;
});

// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

// Runs the parity check (parity.mjs) in Node.
//
// Usage: node examples/wasm/parity/node.mjs [PKG_DIR] [FIXTURES_DIR]
// PKG_DIR defaults to examples/wasm/pkg, built by examples/wasm/build.sh.

import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gzipSync } from "node:zlib";

import { runParity } from "./parity.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(process.argv[2] ?? join(here, "..", "pkg"));
const fixtures = resolve(process.argv[3] ?? join(here, "..", "..", "..", "tests", "fixtures"));

const { initSync, Transcriber, version } = await import(
  pathToFileURL(join(pkgDir, "basic_pitch_wasm.js")).href
);
const wasm = await readFile(join(pkgDir, "basic_pitch_wasm_bg.wasm"));
const t0 = performance.now();
initSync({ module: wasm });
console.log(`basic-pitch-rs ${version()}, Node ${process.version}, ${process.platform} ${process.arch}`);
console.log(
  `module: ${wasm.length} bytes (${gzipSync(wasm, { level: 9 }).length} gzipped), ` +
    `compiled in ${(performance.now() - t0).toFixed(0)} ms`,
);

const summary = await runParity({
  Transcriber,
  read: async (path) => new Uint8Array(await readFile(join(fixtures, path))),
  log: (line) => console.log(line),
});
process.exitCode = summary.failures ? 1 : 0;

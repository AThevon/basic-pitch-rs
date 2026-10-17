#!/usr/bin/env bash
# Copyright 2026 Adrien Thevon
# SPDX-License-Identifier: Apache-2.0
#
# Builds the WebAssembly module and its JavaScript bindings into examples/wasm/pkg.
#
# Requirements: the wasm32-unknown-unknown target (rustup target add
# wasm32-unknown-unknown) and wasm-bindgen-cli at the exact version of the
# wasm-bindgen crate in Cargo.lock (the script prints the install command if needed).
#
# The module uses WebAssembly SIMD (simd128), which tract's matrix kernels need to be
# fast; every current browser and Node supports it. Set SIMD=0 to build without it.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
profile="wasm-release"

target_dir="${CARGO_TARGET_DIR:-$root/target}"

expected="$(awk '$0 == "name = \"wasm-bindgen\"" { getline; gsub(/version = |"/, ""); print; exit }' "$root/Cargo.lock")"
installed="$(wasm-bindgen --version 2>/dev/null | awk '{ print $2 }' || true)"
if [ "$installed" != "$expected" ]; then
  echo "wasm-bindgen-cli $expected is required (found: ${installed:-none}):" >&2
  echo "  cargo install wasm-bindgen-cli --version $expected --locked" >&2
  exit 1
fi

rustflags="${RUSTFLAGS:-}"
if [ "${SIMD:-1}" != 0 ]; then
  rustflags="$rustflags -C target-feature=+simd128"
fi

RUSTFLAGS="$rustflags" cargo build --manifest-path "$root/Cargo.toml" --locked \
  -p basic-pitch-wasm --target wasm32-unknown-unknown --profile "$profile"

wasm-bindgen --target web --out-dir "$here/pkg" \
  "$target_dir/wasm32-unknown-unknown/$profile/basic_pitch_wasm.wasm"

wasm="$here/pkg/basic_pitch_wasm_bg.wasm"
echo "built $wasm: $(wc -c <"$wasm" | tr -d ' ') bytes, $(gzip -9 -c "$wasm" | wc -c | tr -d ' ') gzipped"

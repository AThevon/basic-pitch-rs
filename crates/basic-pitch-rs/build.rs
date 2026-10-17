// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Build script: only does something for the `ort` feature on macOS.
//!
//! The prebuilt ONNX Runtime library that `ort` links statically contains Objective-C++
//! code compiled with `@available` checks, which call `__isPlatformVersionAtLeast` from
//! clang's runtime library. rustc does not link that library, so a link with LTO (which
//! pulls every object of the archive) fails with an undefined symbol. Linking
//! `libclang_rt.osx.a` from the active Xcode toolchain resolves it.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_ORT").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
    {
        return;
    }
    let Some(dir) = clang_runtime_dir() else {
        println!(
            "cargo:warning=clang's runtime library was not found; linking ONNX Runtime with LTO may fail"
        );
        return;
    };
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}

fn clang_runtime_dir() -> Option<PathBuf> {
    let output = Command::new("xcrun")
        .args(["clang", "-print-resource-dir"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let resource_dir = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    let dir = resource_dir.join("lib").join("darwin");
    dir.join("libclang_rt.osx.a").is_file().then_some(dir)
}

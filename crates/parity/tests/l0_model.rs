// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L0: the bundled model is the official `nmp.onnx`, unmodified.

use basic_pitch::model::{MODEL_ONNX, MODEL_SHA256};
use parity::{Variant, meta};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn bundled_model_is_the_official_one() {
    assert_eq!(MODEL_ONNX.len(), 230_444);
    assert_eq!(hex(&Sha256::digest(MODEL_ONNX)), MODEL_SHA256);
    for variant in Variant::ALL {
        assert_eq!(meta(variant)["model_sha256"], MODEL_SHA256, "{variant:?}");
        assert_eq!(meta(variant)["model_backend"], "ONNX", "{variant:?}");
    }
}

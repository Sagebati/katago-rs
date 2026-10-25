//! Browser host of the ported KataGo analysis engine (plan §7), for packages/katago-web's Web Worker:
//! [`Engine`] runs `katago_core::command::analysis::analysis` (upstream's `MainCmds::analysis`) with the
//! browser's versions of what katago-cli takes from the process:
//! - stdin: lines the worker hands to [`Engine::send_lines`] (one call = one burst, all immediately
//!   available, like a block of lines written to stdin at once); [`Engine::close_input`] is end of input;
//! - stdout: a JS callback per response line (`cout << message << endl`);
//! - stderr and the log: a JS callback `(kind, text)` (no log files in the browser);
//! - clock: `performance.now()`; timed waits: `setTimeout` (cleared when the wait is dropped);
//! - executor: `wasm_bindgen_futures::spawn_local` (the JS microtask queue of the worker);
//! - NN backend: the networks compiled into katago-nn-burn on WebGPU (burn wgpu) or the wasm CPU (burn
//!   flex), with the `.bpk` weights passed in as bytes (fetched, cached and sha256-checked by the worker);
//! - config: built in memory, see [`config`].
//!
//! Plus a full-game review (`Engine.reviewGame`, M6): katago-ws's annotation (crate katago-annotate) over
//! this engine, returning the payload v3 JSON (docs/M6-report.md).
//!
//! Natively only [`config`] and [`sha256_hex`] compile (for their unit tests); the engine is wasm32-only.
//! Not a port of upstream code.

pub mod config;
#[cfg(target_arch = "wasm32")]
mod review;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
pub use web::*;

use sha2::Digest;
use wasm_bindgen::prelude::wasm_bindgen;

/// Lowercase hex SHA-256 of `bytes` (the worker checks downloaded and cached weights with it).
#[wasm_bindgen(js_name = sha256Hex)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  let digest = sha2::Sha256::digest(bytes);
  let mut s = String::with_capacity(64);
  for b in digest {
    s.push_str(&format!("{b:02x}"));
  }
  s
}

/// The networks compiled into this build (`network` option of [`Engine`]).
#[wasm_bindgen(js_name = compiledNetworks)]
pub fn compiled_networks() -> Vec<String> {
  katago_nn_burn::NetId::compiled().map(|net| net.name().to_string()).collect()
}

#[cfg(test)]
mod tests {
  #[test]
  fn sha256_of_abc() {
    assert_eq!(super::sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  }
}

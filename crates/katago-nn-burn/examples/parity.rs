//! Raw outputs from burn vs onnxruntime on the verify.py positions
//! (`fixtures/m0/<fixture>.*.f32`, written from `verify.py --dump-features`).
//! `human` (b18c384nbt-humanv0, feature `net-b18c384nbt-humanv0`): the same positions (its spatial and
//! global inputs are b18c384nbt's files) with `meta_input` = the rows of humanSLProfile rank_1k_15k
//! (`verify.py --human-sl-profiles rank_1k_15k --dump-features`, the npz arrays written as raw f32).
//!
//! cargo run --release -p katago-nn-burn --example parity [-F wgpu-fixed-kernels|rocm] -- <flex|wgpu|rocm> [b10|b18|human]
//!
//! In f32: f16 is checked against the Eigen reference by `just nn-parity` (tests/nn_parity.rs).

mod common;

use burn::prelude::*;
use katago_nn_burn::Network;

fn read(fixture: &str, name: &str) -> Vec<f32> {
    let path = format!("{}/../../fixtures/m0/{fixture}.{name}.f32", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn max_diff(name: &str, got: Tensor<2>, want: &[f32]) -> f32 {
    let got = got.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len(), "{name}: length");
    let d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let scale = want.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("  {name:<10} max |diff| {d:.3e}   (max |value| {scale:.2})");
    d
}

fn parity<N: Network>(backend: &str, bpk: &str, fixture: &str, input_fixture: &str) -> f32 {
    let device = common::device(backend);
    let model = N::from_bpk(std::fs::read(bpk).unwrap(), &device);
    let bin = read(input_fixture, "bin_input");
    let n = bin.len() / (22 * 19 * 19);
    let bin = Tensor::<4>::from_data(TensorData::new(bin, [n, 22, 19, 19]), &device);
    let global = Tensor::<2>::from_data(TensorData::new(read(input_fixture, "global_input"), [n, 19]), &device);
    let meta = (N::NUM_INPUT_META_CHANNELS > 0).then(|| {
        let c = N::NUM_INPUT_META_CHANNELS as usize;
        Tensor::<2>::from_data(TensorData::new(read(fixture, "meta_input"), [n, c]), &device)
    });
    let out = model.forward(bin, global, meta);
    println!("{} on {backend}: {n} positions, raw outputs vs onnxruntime", N::NAME);
    [
        max_diff("policy", out.policy.flatten::<2>(1, 2), &read(fixture, "policy")),
        max_diff("value", out.value, &read(fixture, "value")),
        max_diff("scorevalue", out.score, &read(fixture, "scorevalue")),
        max_diff("ownership", out.ownership.flatten::<2>(1, 3), &read(fixture, "ownership")),
    ]
    .into_iter()
    .fold(0.0, f32::max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "flex".into());
    let net = args.next().unwrap_or_else(|| "b10".into());
    let (fixture, input_fixture) = match net.as_str() {
        "b18" => ("b18c384nbt", "b18c384nbt"),
        "human" => ("b18c384nbt-humanv0", "b18c384nbt"),
        _ => ("b10c128", "b10c128"),
    };
    let worst = with_net!(net.as_str(), Net => |bpk| parity::<Net>(&backend, bpk, fixture, input_fixture));
    // Raw logits; KataGo's own fp16/fp32 backends disagree by more than this. GPUs reorder sums (Winograd, GEMMs).
    let tol = if backend == "flex" { 1e-4 } else { 2e-3 };
    println!("{}", if worst <= tol { "PASS" } else { "FAIL" });
    if worst > tol {
        std::process::exit(1);
    }
}

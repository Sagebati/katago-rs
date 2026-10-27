//! Forward-pass throughput of an imported net on one backend: evals/s with one readback per batch, as the engine
//! reads its batches (random inputs: the speed does not depend on them).
//!
//! cargo run --release -p katago-nn-burn --example bench -F <wgpu-fixed|rocm> -- <flex|wgpu|rocm> \
//!     <b6|b10|b18|human|b10c384tf|b10c512tf|tf3> [f16|f32] [batches...]
//!
//! The transformer nets need their feature too (`-F net-b10c384h6nbttflrs`, `net-b10c512h8nbt3tflrs-fson-silu-rsnh`,
//! `net-tf3-b11c768`).
//!
//! `just bench` runs it. `wgpu-fixed` is the browser's kernel set (no autotune); `-F wgpu` adds burn's
//! autotune. Batches default to 1 8 16 32 64.

mod common;

use std::time::Instant;

use burn::prelude::*;
use katago_nn_burn::Network;

fn bench<N: Network>(backend: &str, precision: &str, bpk: &str, batches: &[usize]) {
  let device = common::device(backend);
  let t = Instant::now();
  let model = N::from_bpk(std::fs::read(bpk).unwrap(), &device);
  println!("{} on {backend} ({precision}): load {:?}", N::NAME, t.elapsed());

  for &n in batches {
    // Feature 0 is the on-board mask; everything else random-ish but finite.
    let bin = Tensor::<4>::random([n, 22, 19, 19], burn::tensor::Distribution::Bernoulli(0.3), &device)
      .slice_assign(s![.., 0..1, .., ..], Tensor::ones([n, 1, 19, 19], &device));
    let global = Tensor::<2>::random([n, 19], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
    // Human-SL nets take SGF metadata rows too.
    let meta = (N::NUM_INPUT_META_CHANNELS > 0).then(|| Tensor::<2>::zeros([n, N::NUM_INPUT_META_CHANNELS as usize], &device));

    let run = || {
      let out = model.forward(bin.clone(), global.clone(), meta.clone());
      // One readback per batch, like the engine will do.
      let all = Tensor::cat(vec![out.policy.flatten::<2>(1, 2), out.value, out.score, out.ownership.flatten::<2>(1, 3)], 1);
      // Read the values: on rocm `into_data` returns before the GPU is done, so an unread readback lets the batches
      // queue up and times the queue instead of the GPU.
      std::hint::black_box(all.into_data().iter::<f32>().sum::<f32>())
    };
    // The first runs at a new shape compile kernels (and autotune with `-F wgpu`).
    let warm = Instant::now();
    run();
    let first = warm.elapsed();
    for _ in 0..5 {
      run();
    }
    let mut iters = 0;
    let t = Instant::now();
    while iters < 8 || t.elapsed().as_secs_f64() < 2.0 {
      run();
      iters += 1;
    }
    let dt = t.elapsed();
    let evals = (iters * n) as f64 / dt.as_secs_f64();
    println!(
      "{backend} batch {n:>3}: first {first:>10.2?}  {:>8.2?}/batch  {evals:>8.1} evals/s",
      dt / iters as u32
    );
  }
}

fn main() {
  let mut args: Vec<String> = std::env::args().skip(1).collect();
  let precision = common::take_precision(&mut args);
  let mut args = args.into_iter();
  let backend = args.next().unwrap_or_else(|| "flex".into());
  let net = args.next().unwrap_or_else(|| "b10".into());
  let batches: Vec<usize> = {
    let b: Vec<usize> = args.map(|a| a.parse().unwrap()).collect();
    if b.is_empty() { vec![1, 8, 16, 32, 64] } else { b }
  };
  with_net!(net.as_str(), Net => |bpk| bench::<Net>(&backend, precision, bpk, &batches));
}

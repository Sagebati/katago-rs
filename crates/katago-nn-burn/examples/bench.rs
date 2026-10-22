//! Forward-pass throughput of an imported net on one backend.
//!
//! cargo run --release -p katago-nn-burn --example bench [-F wgpu,net-b18c384nbt] -- <flex|wgpu> <b10|b18> [batches...]

mod common;

use std::time::Instant;

use burn::prelude::*;
use katago_nn_burn::Network;

fn bench<N: Network>(backend: &str, bpk: &str, batches: &[usize]) {
    let device = common::device(backend);
    let t = Instant::now();
    let model = N::from_bpk(std::fs::read(bpk).unwrap(), &device);
    println!("{} on {backend}: load {:?}", N::NAME, t.elapsed());

    for &n in batches {
        // Feature 0 is the on-board mask; everything else random-ish but finite.
        let bin = Tensor::<4>::random([n, 22, 19, 19], burn::tensor::Distribution::Bernoulli(0.3), &device)
            .slice_assign(s![.., 0..1, .., ..], Tensor::ones([n, 1, 19, 19], &device));
        let global = Tensor::<2>::random([n, 19], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);

        let run = || {
            let out = model.forward(bin.clone(), global.clone());
            // One readback per batch, like the engine will do.
            let all = Tensor::cat(
                vec![
                    out.policy.flatten::<2>(1, 2),
                    out.value,
                    out.score,
                    out.ownership.flatten::<2>(1, 3),
                ],
                1,
            );
            all.into_data()
        };
        // The first runs at a new shape compile kernels and autotune.
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
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "flex".into());
    let net = args.next().unwrap_or_else(|| "b10".into());
    let batches: Vec<usize> = {
        let b: Vec<usize> = args.map(|a| a.parse().unwrap()).collect();
        if b.is_empty() { vec![1, 8, 16, 32] } else { b }
    };
    with_net!(net.as_str(), Net => |bpk| bench::<Net>(&backend, bpk, &batches));
}

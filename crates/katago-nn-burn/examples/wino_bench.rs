//! Per-layer check and timing of the Winograd 3x3 conv (wino.rs) against `conv2d_same` on b18c384nbt's 3x3
//! shapes.
//!
//! cargo run --release -p katago-nn-burn --example wino_bench -F wgpu,fusion -- [batches...]

mod common;

use std::time::Instant;

use burn::backend::Dispatch;
use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use katago_nn_burn::conv::conv2d_im2col as conv2d_same;
use katago_nn_burn::wino::{WinogradOps, transform_filter_3x3};

/// (ic, oc, count in b18c384nbt)
const SHAPES: [(usize, usize, usize); 4] = [(192, 192, 62), (192, 128, 5), (192, 64, 5), (128, 192, 5)];

/// Convs queued per readback, so one sync is spread over many kernels.
const CHAIN: usize = 20;

/// Seconds per conv: `CHAIN` convs summed into one tensor, then one scalar readback.
fn time(f: impl Fn() -> Tensor<4>) -> f64 {
    let run = || {
        let mut acc = f();
        for _ in 1..CHAIN {
            acc = acc + f();
        }
        acc.slice(s![0..1, 0..1, 0..1, 0..1]).into_scalar::<f32>();
    };
    for _ in 0..2 {
        run();
    }
    let mut iters = 0;
    let t = Instant::now();
    while iters < 5 || t.elapsed().as_secs_f64() < 1.0 {
        run();
        iters += 1;
    }
    t.elapsed().as_secs_f64() / (iters * CHAIN) as f64
}

fn main() {
    let batches: Vec<usize> = {
        let b: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
        if b.is_empty() { vec![8, 16, 32, 64] } else { b }
    };
    let device = common::device("wgpu");
    println!("{:>10} {:>5} {:>10} {:>10} {:>7} {:>10}", "shape", "batch", "im2col ms", "wino ms", "speedup", "max |diff|");
    let mut total_ref = vec![0f64; batches.len()];
    let mut total_wino = vec![0f64; batches.len()];
    for &(ic, oc, count) in &SHAPES {
        let weight = Tensor::<4>::random([oc, ic, 3, 3], burn::tensor::Distribution::Normal(0.0, 0.05), &device);
        let mut conv = Conv2dConfig::new([ic, oc], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(&device);
        conv.weight = Param::from_tensor(weight.clone());
        let w_host: Vec<f32> = weight.into_data().try_into_vec::<f32>().unwrap();
        let filter_host = transform_filter_3x3(&w_host, oc, ic);
        let filter = Tensor::<1>::from_floats(filter_host.as_slice(), &device);

        for (bi, &n) in batches.iter().enumerate() {
            let x = Tensor::<4>::random([n, ic, 19, 19], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            let wino = || {
                Tensor::<4>::from_dispatch(Dispatch::wino_conv3x3(x.clone().into_dispatch(), filter.clone().into_dispatch(), oc))
            };
            let reference = conv2d_same(&conv, x.clone());
            let diff = (wino() - reference).abs().max().into_scalar::<f32>();
            let t_ref = time(|| conv2d_same(&conv, x.clone()));
            let t_wino = time(wino);
            total_ref[bi] += t_ref * count as f64;
            total_wino[bi] += t_wino * count as f64;
            println!(
                "{:>10} {n:>5} {:>10.3} {:>10.3} {:>6.2}x {diff:>10.2e}",
                format!("{ic}->{oc}"),
                t_ref * 1e3,
                t_wino * 1e3,
                t_ref / t_wino
            );
        }
    }
    println!("\nAll 77 3x3 convs of b18c384nbt, per batch (per-conv time x count):");
    for (bi, &n) in batches.iter().enumerate() {
        println!(
            "batch {n:>3}: im2col {:>8.1} ms  wino {:>8.1} ms  ({:.2}x)",
            total_ref[bi] * 1e3,
            total_wino[bi] * 1e3,
            total_ref[bi] / total_wino[bi]
        );
    }
}

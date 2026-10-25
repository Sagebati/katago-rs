//! Per-layer check and timing of wino.rs's convs (Winograd 3x3, NCHW 1x1) against `conv2d_im2col` on
//! b18c384nbt's trunk shapes.
//!
//! cargo run --release -p katago-nn-burn --example wino_bench -F wgpu,fusion -- [batches...]

mod common;

use std::time::Instant;

use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;
use katago_nn_burn::conv::conv2d_im2col;
use katago_nn_burn::wino::{self, ConvOps};
use burn::backend::Dispatch;

fn device_of(conv: &burn::nn::conv::Conv2d) -> Device {
    conv.weight.val().device()
}

/// (kernel, ic, oc, count in b18c384nbt)
const SHAPES: [(usize, usize, usize, usize); 6] =
    [(3, 192, 192, 62), (3, 192, 128, 5), (3, 192, 64, 5), (3, 128, 192, 5), (1, 384, 192, 18), (1, 192, 384, 18)];

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
    println!("{:>12} {:>5} {:>10} {:>10} {:>7} {:>10}", "shape", "batch", "im2col ms", "new ms", "speedup", "max |diff|");
    let mut total_ref = [vec![0f64; batches.len()], vec![0f64; batches.len()]];
    let mut total_new = [vec![0f64; batches.len()], vec![0f64; batches.len()]];
    for &(k, ic, oc, count) in &SHAPES {
        let weight = Tensor::<4>::random([oc, ic, k, k], burn::tensor::Distribution::Normal(0.0, 0.05), &device);
        let mut conv = Conv2dConfig::new([ic, oc], [k, k])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(&device);
        conv.weight = Param::from_tensor(weight.clone());
        let kind = usize::from(k == 1);
        if k == 3 {
            // The device filter transform against the host reference.
            let host = wino::transform_filter_3x3(&weight.clone().into_data().try_into_vec::<f32>().unwrap(), oc, ic);
            let device = Tensor::<1>::from_dispatch(Dispatch::wino_filter_3x3(weight.into_dispatch(), false));
            let diff = (device - Tensor::<1>::from_floats(host.as_slice(), &device_of(&conv)))
                .abs()
                .max()
                .into_scalar::<f32>();
            println!("{:>12} filter transform: max |device - host| {diff:.2e}", format!("{k}x{k} {ic}->{oc}"));
            assert!(diff < 1e-6, "device filter transform differs from the host one");
        }

        for (bi, &n) in batches.iter().enumerate() {
            let x = Tensor::<4>::random([n, ic, 19, 19], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            let new = || if k == 3 { wino::conv3x3(&conv, x.clone()) } else { wino::conv1x1(&conv, x.clone()) };
            let reference = conv2d_im2col(&conv, x.clone());
            let diff = (new() - reference).abs().max().into_scalar::<f32>();
            let t_ref = time(|| conv2d_im2col(&conv, x.clone()));
            let t_new = time(new);
            total_ref[kind][bi] += t_ref * count as f64;
            total_new[kind][bi] += t_new * count as f64;
            println!(
                "{:>12} {n:>5} {:>10.3} {:>10.3} {:>6.2}x {diff:>10.2e}",
                format!("{k}x{k} {ic}->{oc}"),
                t_ref * 1e3,
                t_new * 1e3,
                t_ref / t_new
            );
        }
    }
    for (kind, label) in ["77 3x3 convs", "36 trunk 1x1 convs"].iter().enumerate() {
        println!("\n{label} of b18c384nbt, per batch (per-conv time x count):");
        for (bi, &n) in batches.iter().enumerate() {
            println!(
                "batch {n:>3}: im2col {:>8.1} ms  new {:>8.1} ms  ({:.2}x)",
                total_ref[kind][bi] * 1e3,
                total_new[kind][bi] * 1e3,
                total_ref[kind][bi] / total_new[kind][bi]
            );
        }
    }
}

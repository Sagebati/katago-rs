//! Isolates op costs on one backend: a 3x3 128->128 conv on 19x19, a 1x1 conv,
//! an elementwise chain, and a matmul of equal FLOPs.

mod common;

use std::time::Instant;

use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::prelude::*;

fn time(label: &str, flops: f64, mut f: impl FnMut() -> TensorData) {
    for _ in 0..5 {
        f();
    }
    let mut iters = 0;
    let t = Instant::now();
    while iters < 10 || t.elapsed().as_secs_f64() < 1.5 {
        f();
        iters += 1;
    }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    println!("{label:<34} {:>9.3} ms  {:>8.1} GFLOP/s", per * 1e3, flops / per / 1e9);
}

fn main() {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "flex".into());
    let n: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(32);
    let device = common::device(&backend);
    let x = Tensor::<4>::random([n, 128, 19, 19], burn::tensor::Distribution::Default, &device);
    let hw = 361.0;

    let conv3 = Conv2dConfig::new([128, 128], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(&device);
    time("conv 3x3 128->128", 2.0 * n as f64 * hw * 128.0 * 128.0 * 9.0, || {
        conv3.forward(x.clone()).sum().into_data()
    });
    // im2col by hand: pad once, take the 9 shifted windows, one matmul.
    let w = Tensor::<2>::random([128 * 9, 128], burn::tensor::Distribution::Default, &device);
    let im2col = |x: Tensor<4>| {
        let [n, c, h, wd] = x.dims();
        let p = x.pad((1, 1, 1, 1), burn::tensor::ops::PadMode::Constant(0.0));
        let mut cols = Vec::with_capacity(9);
        for dy in 0..3 {
            for dx in 0..3 {
                cols.push(p.clone().slice(s![.., .., dy..dy + h, dx..dx + wd]));
            }
        }
        // [n, 9*c, h, w] -> [n*h*w, 9*c]
        let cols = Tensor::cat(cols, 1).permute([0, 2, 3, 1]).reshape([n * h * wd, 9 * c]);
        cols.matmul(w.clone()).reshape([n, h, wd, 128]).permute([0, 3, 1, 2])
    };
    time("conv 3x3 via im2col+matmul", 2.0 * n as f64 * hw * 128.0 * 128.0 * 9.0, || {
        im2col(x.clone()).sum().into_data()
    });
    // NHWC-resident variant: activations stay [n*h*w, c], no permutes around the matmul.
    let xn = x.clone().permute([0, 2, 3, 1]);
    let im2col_nhwc = |x: Tensor<4>| {
        let [n, h, wd, c] = x.dims();
        let p = x.pad((0, 0, 1, 1), burn::tensor::ops::PadMode::Constant(0.0));
        let p = p.permute([0, 1, 3, 2]).pad((1, 1, 0, 0), burn::tensor::ops::PadMode::Constant(0.0)).permute([0, 1, 3, 2]);
        let mut cols = Vec::with_capacity(9);
        for dy in 0..3 {
            for dx in 0..3 {
                cols.push(p.clone().slice(s![.., dy..dy + h, dx..dx + wd, ..]));
            }
        }
        Tensor::cat(cols, 3).reshape([n * h * wd, 9 * c]).matmul(w.clone())
    };
    time("conv 3x3 im2col NHWC", 2.0 * n as f64 * hw * 128.0 * 128.0 * 9.0, || {
        im2col_nhwc(xn.clone()).sum().into_data()
    });
    let conv1 = Conv2dConfig::new([128, 128], [1, 1]).with_bias(false).init(&device);
    time("conv 1x1 128->128", 2.0 * n as f64 * hw * 128.0 * 128.0, || {
        conv1.forward(x.clone()).sum().into_data()
    });
    // Same contraction as the 3x3 conv, expressed as a plain matmul (im2col shape).
    let a = Tensor::<2>::random([n * 361, 128 * 9], burn::tensor::Distribution::Default, &device);
    let b = Tensor::<2>::random([128 * 9, 128], burn::tensor::Distribution::Default, &device);
    time("matmul [n*361,1152]x[1152,128]", 2.0 * n as f64 * hw * 128.0 * 128.0 * 9.0, || {
        a.clone().matmul(b.clone()).sum().into_data()
    });
    let m = Tensor::<4>::ones([n, 1, 19, 19], &device);
    let c = Tensor::<4>::ones([1, 128, 1, 1], &device);
    time("x*c + c, relu, *mask", 4.0 * n as f64 * hw * 128.0, || {
        burn::tensor::activation::relu(x.clone().mul(c.clone()).add(c.clone())).mul(m.clone()).sum().into_data()
    });
}

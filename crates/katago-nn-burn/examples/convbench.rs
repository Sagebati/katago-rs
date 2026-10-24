//! Isolates op costs on one backend: a 3x3 conv on 19x19 (128 channels, or the third argument: 384 for
//! b18c384nbt), a 1x1 conv, an elementwise chain, a global-pooling reduction, and a matmul of equal FLOPs.
//!
//! cargo run --release -p katago-nn-burn --example convbench [-F wgpu] -- <flex|wgpu> [batch=32] [channels=128]

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
    let ch: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(128);
    let device = common::device(&backend);
    let x = Tensor::<4>::random([n, ch, 19, 19], burn::tensor::Distribution::Default, &device);
    let hw = 361.0;

    let conv3 = Conv2dConfig::new([ch, ch], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(&device);
    time("conv 3x3 c->c", 2.0 * n as f64 * hw * ch as f64 * ch as f64 * 9.0, || {
        conv3.forward(x.clone()).sum().into_data()
    });
    // im2col by hand: pad once, take the 9 shifted windows, one matmul.
    let w = Tensor::<2>::random([ch * 9, ch], burn::tensor::Distribution::Default, &device);
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
        cols.matmul(w.clone()).reshape([n, h, wd, ch]).permute([0, 3, 1, 2])
    };
    time("conv 3x3 via im2col+matmul", 2.0 * n as f64 * hw * ch as f64 * ch as f64 * 9.0, || {
        im2col(x.clone()).sum().into_data()
    });
    // The im2col lowering stage by stage (each timing includes the stages before it).
    {
        let stage = |upto: usize| {
            let x = x.clone();
            let w = w.clone();
            move || {
                let [n, c, h, wd] = x.dims();
                let p = x.clone().pad((1, 1, 1, 1), burn::tensor::ops::PadMode::Constant(0.0));
                if upto == 0 {
                    return p.sum().into_data();
                }
                let mut cols = Vec::with_capacity(9);
                for dy in 0..3 {
                    for dx in 0..3 {
                        cols.push(p.clone().slice(s![.., .., dy..dy + h, dx..dx + wd]));
                    }
                }
                let cat = Tensor::cat(cols, 1);
                if upto == 1 {
                    return cat.sum().into_data();
                }
                let cols = cat.permute([0, 2, 3, 1]).reshape([n * h * wd, 9 * c]);
                if upto == 2 {
                    return cols.sum().into_data();
                }
                cols.matmul(w.clone()).sum().into_data()
            }
        };
        time("  im2col: pad", 0.0, stage(0));
        time("  im2col: + 9 slices + cat", 0.0, stage(1));
        time("  im2col: + permute + reshape", 0.0, stage(2));
        time("  im2col: + matmul", 0.0, stage(3));
    }
    let conv1 = Conv2dConfig::new([ch, ch], [1, 1]).with_bias(false).init(&device);
    time("conv 1x1 c->c", 2.0 * n as f64 * hw * ch as f64 * ch as f64, || {
        conv1.forward(x.clone()).sum().into_data()
    });
    // Same contraction as the 3x3 conv, expressed as a plain matmul (im2col shape).
    let a = Tensor::<2>::random([n * 361, ch * 9], burn::tensor::Distribution::Default, &device);
    let b = Tensor::<2>::random([ch * 9, ch], burn::tensor::Distribution::Default, &device);
    time("matmul [n*361,9c]x[9c,c]", 2.0 * n as f64 * hw * ch as f64 * ch as f64 * 9.0, || {
        a.clone().matmul(b.clone()).sum().into_data()
    });
    let m = Tensor::<4>::ones([n, 1, 19, 19], &device);
    let c = Tensor::<4>::ones([1, ch, 1, 1], &device);
    time("x*c + c, relu, *mask", 4.0 * n as f64 * hw * ch as f64, || {
        burn::tensor::activation::relu(x.clone().mul(c.clone()).add(c.clone())).mul(m.clone()).sum().into_data()
    });
    // KataGo's global pooling: mean and max over the board, per channel.
    time("mean+max over 19x19 (gpool)", 2.0 * n as f64 * hw * ch as f64, || {
        let flat = x.clone().reshape([n, ch, 361]);
        Tensor::cat(vec![flat.clone().mean_dim(2), flat.max_dim(2)], 2).into_data()
    });
    // The elementwise chain of one residual block (norm, mish) on its own: Mish = x * tanh(softplus(x)).
    time("mish (x*tanh(softplus x))", 8.0 * n as f64 * hw * ch as f64, || {
        let sp = burn::tensor::activation::softplus(x.clone(), 1.0);
        x.clone().mul(sp.tanh()).sum().into_data()
    });
}

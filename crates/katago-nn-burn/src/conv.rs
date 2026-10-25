//! Convolution lowering used by the generated models (see `SameConv` in build.rs).

use burn::nn::conv::Conv2d;
use burn::prelude::*;
use burn::tensor::ops::PadMode;

/// Stride-1 "same" convolution without bias.
///
/// On GPUs 3x3 kernels go through the Winograd convolution of wino.rs (7-12x faster than im2col + matmul per
/// layer without autotune on an RX 6800 XT) and the trunk's 1x1 kernels through its NCHW GEMM (no permutes).
/// Every other kernel size is lowered to (im2col +) one matmul: on wgpu this is ~7x faster than burn's direct
/// conv kernel for KataGo's 19x19 shapes with 3x3 kernels, and for 1x1 kernels too, which matter in
/// b18c384nbt's bottleneck blocks: without autotune (the browser build, `wgpu-fixed-kernels`) burn's 1x1
/// conv is its direct kernel, ~10x slower than the equivalent matmul, and was 78% of the forward pass
/// (docs/M7-report.md). The CPU backend, where burn's own conv is faster, uses it for every size.
pub fn conv2d_same(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    if is_cpu(&x.device()) {
        return conv.forward(x);
    }
    #[cfg(feature = "wgpu-fixed-kernels")]
    {
        let [oc, ic, kh, kw] = conv.weight.dims();
        if (kh, kw) == (3, 3) {
            return crate::wino::conv3x3(conv, x);
        }
        if (kh, kw) == (1, 1) && crate::wino::conv1x1_fits(oc, ic) {
            return crate::wino::conv1x1(conv, x);
        }
    }
    conv2d_im2col(conv, x)
}

/// `conv2d_same(conv, mish(x * scale + bias) * mask)`: KataGo's batch norm + activation + board mask before a conv
/// (build.rs `fuse_bn_act` rewrites the generated chains into this). On GPUs the prologue runs inside the conv's
/// input loads (Winograd transform, 1x1 GEMM) instead of as a separate elementwise kernel; elsewhere it is the
/// original op sequence.
pub fn conv2d_bn_act(conv: &Conv2d, x: Tensor<4>, scale: Tensor<4>, bias: Tensor<4>, mask: Tensor<4>) -> Tensor<4> {
    #[cfg(feature = "wgpu-fixed-kernels")]
    if !is_cpu(&x.device()) {
        let [oc, ic, kh, kw] = conv.weight.dims();
        if (kh, kw) == (3, 3) {
            return crate::wino::conv3x3_bn_act(conv, x, scale, bias, mask);
        }
        if (kh, kw) == (1, 1) && crate::wino::conv1x1_fits(oc, ic) {
            return crate::wino::conv1x1_bn_act(conv, x, scale, bias, mask);
        }
    }
    let a = x.mul(scale).add(bias);
    let sp = burn::tensor::activation::softplus(a.clone(), 1.0);
    let m = a.mul(sp.tanh()).mul(mask);
    conv2d_same(conv, m)
}

/// [`conv2d_same`] as (im2col +) one matmul, for every kernel size on a GPU.
pub fn conv2d_im2col(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let weight = conv.weight.val();
    let [oc, ic, kh, kw] = weight.dims();
    let [n, _, h, w] = x.dims();
    let (ph, pw) = (kh / 2, kw / 2);
    let cols = if kh == 1 && kw == 1 {
        x
    } else {
        let padded = x.pad((pw, pw, ph, ph), PadMode::Constant(0.0));
        // Column (dy*kw + dx)*ic + c holds input channel c shifted by (dy, dx).
        let mut windows = Vec::with_capacity(kh * kw);
        for dy in 0..kh {
            for dx in 0..kw {
                windows.push(padded.clone().slice(s![.., .., dy..dy + h, dx..dx + w]));
            }
        }
        Tensor::cat(windows, 1)
    };
    let cols = cols.permute([0, 2, 3, 1]).reshape([n * h * w, kh * kw * ic]);
    let weight = weight.permute([2, 3, 1, 0]).reshape([kh * kw * ic, oc]);
    cols.matmul(weight)
        .reshape([n, h, w, oc])
        .permute([0, 3, 1, 2])
}

fn is_cpu(device: &Device) -> bool {
    #[cfg(feature = "flex")]
    if matches!(device.as_dispatch(), burn::backend::DispatchDevice::Flex(_)) {
        return true;
    }
    let _ = device;
    false
}

/// ReduceSum over the last axis of `[N, C, L]` (build.rs `SpatialReduce`): one workgroup per row on GPUs.
pub fn spatial_sum(x: Tensor<3>) -> Tensor<2> {
    #[cfg(feature = "wgpu-fixed-kernels")]
    if !is_cpu(&x.device()) {
        return crate::wino::reduce_last(x, false);
    }
    x.sum_dim(2usize).squeeze_dims::<2usize>(&[2])
}

/// ReduceMax over the last axis of `[N, C, L]` (build.rs `SpatialReduce`).
pub fn spatial_max(x: Tensor<3>) -> Tensor<2> {
    #[cfg(feature = "wgpu-fixed-kernels")]
    if !is_cpu(&x.device()) {
        return crate::wino::reduce_last(x, true);
    }
    x.max_dim(2usize).squeeze_dims::<2usize>(&[2])
}

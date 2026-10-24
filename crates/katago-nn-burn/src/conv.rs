//! Convolution lowering used by the generated models (see `SameConv` in build.rs).

use burn::nn::conv::Conv2d;
use burn::prelude::*;
use burn::tensor::ops::PadMode;

/// Stride-1 "same" convolution without bias.
///
/// On GPUs every kernel size is lowered to (im2col +) one matmul: on wgpu this is ~7x faster than burn's direct
/// conv kernel for KataGo's 19x19 shapes with 3x3 kernels, and for 1x1 kernels too, which matter in
/// b18c384nbt's bottleneck blocks: without autotune (the browser build, `wgpu-fixed-kernels`) burn's 1x1
/// conv is its direct kernel, ~10x slower than the equivalent matmul, and was 78% of the forward pass
/// (docs/M7-report.md). The CPU backend, where burn's own conv is faster, uses it for every size.
pub fn conv2d_same(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    if is_cpu(&x.device()) {
        return conv.forward(x);
    }
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

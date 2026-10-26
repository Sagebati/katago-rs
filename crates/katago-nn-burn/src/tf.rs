//! Transformer-block lowering used by the generated models: build.rs (`fuse_tf`) routes their linears, per-token
//! RMSNorms and attention here. On GPUs they run the kernels of `tfk`; elsewhere (and for shapes those kernels do
//! not take) the original op sequence.

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::ops::AttentionModuleOptions;

#[cfg(feature = "gpu-kernels")]
use crate::is_cpu;

/// `lin.forward(x)`: one token-major GEMM over all the leading axes on GPUs (instead of burn's batched matmul with a
/// broadcast weight).
pub fn linear<const D: usize>(lin: &Linear, x: Tensor<D>) -> Tensor<D> {
  #[cfg(feature = "gpu-kernels")]
  if D >= 2 && !is_cpu(&x.device()) {
    let dims = x.dims();
    let k = dims[D - 1];
    let rows = dims[..D - 1].iter().product::<usize>();
    let w = lin.weight.val();
    let mut out_dims = dims;
    out_dims[D - 1] = w.dims()[1];
    return crate::tfk::linear(x.reshape([rows, k]), w, lin.bias.as_ref().map(|b| b.val())).reshape(out_dims);
  }
  lin.forward(x)
}

/// A transformer block's RMSNorm of the `[N, S, C]` tokens (bin2onnx `transformer_rmsnorm`):
/// `x * (1 / sqrt(sum(x^2) / divisor + eps)) * gamma * mask`, with `divisor`, `eps` the graph's 1-element constants,
/// `gamma` `[C]` and the `[N, S, 1]` token mask. One kernel per norm on GPUs.
pub fn rms_norm(x: Tensor<3>, divisor: Tensor<1>, eps: Tensor<1>, gamma: Tensor<1>, mask: Tensor<3>) -> Tensor<3> {
  #[cfg(feature = "gpu-kernels")]
  if !is_cpu(&x.device()) {
    let [n, s, c] = x.dims();
    return crate::tfk::rms_norm(x.reshape([n * s, c]), gamma, mask.reshape([n * s]), divisor, eps).reshape([n, s, c]);
  }
  let ss = x.clone().mul(x.clone()).sum_dim(2usize);
  let den = ss
    .div(divisor.unsqueeze_dims(&[0isize, 1isize]))
    .add(eps.unsqueeze_dims(&[0isize, 1isize]))
    .sqrt();
  let rms = den.recip();
  x.mul(rms).mul(gamma.unsqueeze_dims(&[0isize, 1isize])).mul(mask)
}

/// `burn::tensor::module::attention` with the graph's arguments. The nets' form (no boolean mask, an additive
/// `[N, 1, 1, S]` key bias for the off-board points, no softcap, not causal) runs as one fused kernel on GPUs.
pub fn attention(
  q: Tensor<4>,
  k: Tensor<4>,
  v: Tensor<4>,
  mask: Option<Tensor<4, Bool>>,
  bias: Option<Tensor<4>>,
  options: AttentionModuleOptions,
) -> Tensor<4> {
  #[cfg(feature = "gpu-kernels")]
  if !is_cpu(&q.device())
    && let (None, Some(b)) = (&mask, &bias)
    && options.softcap.is_none()
    && !options.is_causal
  {
    let [n, _, _, s] = b.dims();
    if b.dims() == [n, 1, 1, s] && q.dims()[0] == n && q.dims()[2] == s {
      let dq = q.dims()[3];
      let scale = options.scale.unwrap_or(1.0 / (dq as f64).sqrt()) as f32;
      return crate::tfk::attention(q, k, v, bias.unwrap().reshape([n, s]), scale);
    }
  }
  burn::tensor::module::attention(q, k, v, mask, bias, options)
}

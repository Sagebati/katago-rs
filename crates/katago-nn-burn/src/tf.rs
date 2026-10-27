//! Transformer-block lowering used by the generated models: build.rs (`fuse_tf`) routes their linears, per-token
//! RMSNorms and attention here. On GPUs they run the kernels of `tfk`; elsewhere (and for shapes those kernels do
//! not take) the original op sequence.

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::s;

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
    let mut w = lin.weight.val();
    let mut out_dims = dims;
    out_dims[D - 1] = w.dims()[1];
    // f16 nets (crate::wino::half): an f16 copy of the weight, for the packed-f16 GEMM (widths a multiple of 4), from a
    // few boards on (its 128-row tiles leave a single board too few workgroups; the f32 GEMM is faster there). So an
    // f16 net's outputs for a position differ slightly between batches below and above HALF_MIN_ROWS (within the f16
    // tolerances; one forward pass is consistent, all its token linears have the same rows).
    if crate::wino::half() && w.dims()[1].is_multiple_of(4) && rows >= crate::tfk::HALF_MIN_ROWS {
      w = crate::tfk::half_weight(lin.weight.id, w);
    }
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

/// burn-onnx's attention block with an additive key bias (the off-board mask): `softmax(q kᵗ * scale + bias) v`
/// per head, and 0 for a query row whose bias is -inf for every key. The nets' form (a `[N, 1, 1, S]` bias, no
/// softcap, not causal) runs as one fused kernel on GPUs.
pub fn attention_masked(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: Tensor<4>, options: AttentionModuleOptions) -> Tensor<4> {
  #[cfg(feature = "gpu-kernels")]
  if !is_cpu(&q.device()) && options.softcap.is_none() && !options.is_causal {
    let [n, _, _, s] = bias.dims();
    if bias.dims() == [n, 1, 1, s] && q.dims()[0] == n && q.dims()[2] == s {
      let dq = q.dims()[3];
      let scale = options.scale.unwrap_or(1.0 / (dq as f64).sqrt()) as f32;
      return crate::tfk::attention(q, k, v, bias.reshape([n, s]), scale);
    }
  }
  let hidden_rows = bias.clone().max_dim(3).equal_elem(f32::NEG_INFINITY);
  let out = burn::tensor::module::attention(q, k, v, None, Some(bias), options);
  let dims = out.dims();
  out.mask_fill(hidden_rows.expand(dims), 0.0)
}

/// A transformer block's attention from its projections to its merged heads (bin2onnx
/// `transformer_attention_block`, build.rs `fuse_tf`): the `[N, S, H * dq]` queries and `[N, S, Hkv * dq | dv]` keys and
/// values split into heads, the rotary embedding of q and k (`rope`: the q and k cos/sin tables, `[1, 1 | H, S, dq / 2]`
/// and `[1, 1 | Hkv, S, dq / 2]`), [`attention_masked`] with the `[N, 1, 1, S]` key bias, and the heads merged back into
/// `[N, S, H * dv]`. On GPUs one kernel, which reads and writes the token layout directly.
#[allow(clippy::too_many_arguments)]
pub fn attention_tokens(
  q: Tensor<3>,
  k: Tensor<3>,
  v: Tensor<3>,
  rope: Option<[Tensor<4>; 4]>,
  bias: Tensor<4>,
  heads: usize,
  kv_heads: usize,
  dq: usize,
  dv: usize,
  scale: f64,
) -> Tensor<3> {
  let [n, s, _] = q.dims();
  #[cfg(feature = "gpu-kernels")]
  if !is_cpu(&q.device()) && bias.dims() == [n, 1, 1, s] {
    let rope = rope.map(|tables| tables.map(|t| t.reshape([-1])));
    return crate::tfk::attention_tokens(q, k, v, rope, bias.reshape([n, s]), heads, kv_heads, dq, dv, scale as f32);
  }
  let split = |t: Tensor<3>, h: usize, d: usize| t.reshape([n, s, h, d]).permute([0, 2, 1, 3]);
  let (mut qh, mut kh, vh) = (split(q, heads, dq), split(k, kv_heads, dq), split(v, kv_heads, dv));
  if let Some([cos_q, sin_q, cos_k, sin_k]) = rope {
    let half = dq / 2;
    let rotate = |x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>| {
      let x0 = x.clone().slice(s![.., .., .., 0..half]);
      let x1 = x.slice(s![.., .., .., half..dq]);
      let r0 = x0.clone().mul(cos.clone()).sub(x1.clone().mul(sin.clone()));
      let r1 = x0.mul(sin).add(x1.mul(cos));
      Tensor::cat(vec![r0, r1], 3)
    };
    qh = rotate(qh, cos_q, sin_q);
    kh = rotate(kh, cos_k, sin_k);
  }
  // GQA: each kv head serves `heads / kv_heads` query heads (burn-onnx's attention block expands them the same way).
  let groups = heads / kv_heads;
  let (kh, vh) = if groups > 1 {
    let expand = |t: Tensor<4>, d: usize| t.unsqueeze_dim::<5>(2).expand([n, kv_heads, groups, s, d]).reshape([n, heads, s, d]);
    (expand(kh, dq), expand(vh, dv))
  } else {
    (kh, vh)
  };
  let options = AttentionModuleOptions {
    scale: Some(scale),
    softcap: None,
    is_causal: false,
  };
  attention_masked(qh, kh, vh, bias, options).permute([0, 2, 1, 3]).reshape([n, s, heads * dv])
}

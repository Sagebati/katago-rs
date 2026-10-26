//! Transformer-block kernels on cubecl (the nets with `hasAnyTransformerBlocks`): the projections and FFN linears as
//! one token-major GEMM, the per-token RMSNorm in one pass, and softmax attention over the board fused into one kernel
//! (no `[S, S]` scores in memory), in place of burn's batched matmul, generic reduce and naive attention.
//!
//! [`kernels`] holds the cubecl kernels, [`ext`] the burn backend extension that launches them ([`TfOps`]), this
//! module the entry points `crate::tf` calls.

mod ext;
mod kernels;

use burn::backend::Dispatch;
use burn::tensor::Tensor;

pub use ext::TfOps;

/// `x · w (+ bias)` for `x` `[M, K]`, `w` `[K, N]`.
pub fn linear(x: Tensor<2>, w: Tensor<2>, bias: Option<Tensor<1>>) -> Tensor<2> {
  let has_bias = bias.is_some();
  // Without a bias the kernel never reads it: any tensor will do.
  let bias = bias.unwrap_or_else(|| w.clone().reshape([-1]));
  Tensor::<2>::from_dispatch(Dispatch::tf_linear(x.into_dispatch(), w.into_dispatch(), bias.into_dispatch(), has_bias))
}

/// [`TfOps::tf_rms_norm`] of `[R, C]` rows.
pub fn rms_norm(x: Tensor<2>, gamma: Tensor<1>, mask: Tensor<1>, divisor: Tensor<1>, eps: Tensor<1>) -> Tensor<2> {
  Tensor::<2>::from_dispatch(Dispatch::tf_rms_norm(
    x.into_dispatch(),
    gamma.into_dispatch(),
    mask.into_dispatch(),
    divisor.into_dispatch(),
    eps.into_dispatch(),
  ))
}

/// [`TfOps::tf_attention`].
pub fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: Tensor<2>, scale: f32) -> Tensor<4> {
  Tensor::<4>::from_dispatch(Dispatch::tf_attention(
    q.into_dispatch(),
    k.into_dispatch(),
    v.into_dispatch(),
    bias.into_dispatch(),
    scale.to_bits(),
  ))
}

//! Transformer-block kernels on cubecl (the nets with `hasAnyTransformerBlocks`): the projections and FFN linears as
//! one token-major GEMM, the per-token RMSNorm in one pass, and softmax attention over the board fused into one kernel
//! (no `[S, S]` scores in memory), in place of burn's batched matmul, generic reduce and naive attention.
//!
//! [`kernels`] holds the cubecl kernels, [`ext`] the burn backend extension that launches them ([`TfOps`]), this
//! module the entry points `crate::tf` calls.

mod ext;
mod kernels;

use std::cell::RefCell;
use std::collections::HashMap;

use burn::backend::Dispatch;
use burn::module::ParamId;
use burn::tensor::{DType, Device, Tensor};

pub use ext::TfOps;

/// Rows from which an f16 net's linears use the packed-f16 GEMM (crate::tf::linear).
pub const HALF_MIN_ROWS: usize = kernels::LIN_TM_ROWS;

thread_local! {
  /// f16 copies of the linears' weights, per device and weight parameter (as wino's transformed filters).
  static HALF_WEIGHTS: RefCell<Vec<(Device, HashMap<ParamId, Tensor<2>>)>> = const { RefCell::new(Vec::new()) };
}

/// The f16 copy of a linear's `[K, N]` weight (made once per device, with device ops only).
pub fn half_weight(id: ParamId, w: Tensor<2>) -> Tensor<2> {
  let device = w.device();
  HALF_WEIGHTS.with(|cache| {
    let mut cache = cache.borrow_mut();
    let i = match cache.iter().position(|(d, _)| *d == device) {
      Some(i) => i,
      None => {
        cache.push((device, HashMap::new()));
        cache.len() - 1
      }
    };
    cache[i].1.entry(id).or_insert_with(|| w.cast(DType::F16)).clone()
  })
}

/// `x · w (+ bias)` for `x` `[M, K]`, `w` `[K, N]` (f32, or f16 for the packed-f16 GEMM).
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

/// [`TfOps::tf_attention_tokens`].
#[allow(clippy::too_many_arguments)]
pub fn attention_tokens(
  q: Tensor<3>,
  k: Tensor<3>,
  v: Tensor<3>,
  rope: Option<[Tensor<1>; 4]>,
  bias: Tensor<2>,
  heads: usize,
  kv_heads: usize,
  dq: usize,
  dv: usize,
  scale: f32,
) -> Tensor<3> {
  let has_rope = rope.is_some();
  // Without RoPE the kernel never reads the tables: any tensor will do.
  let [cq, sq, ck, sk] = rope.unwrap_or_else(|| std::array::from_fn(|_| bias.clone().reshape([-1])));
  Tensor::<3>::from_dispatch(Dispatch::tf_attention_tokens(
    q.into_dispatch(),
    k.into_dispatch(),
    v.into_dispatch(),
    cq.into_dispatch(),
    sq.into_dispatch(),
    ck.into_dispatch(),
    sk.into_dispatch(),
    bias.into_dispatch(),
    heads,
    kv_heads,
    dq,
    dv,
    scale.to_bits(),
    has_rope,
  ))
}

//! The burn backend extension ([`TfOps`]) that launches the kernels of [`super::kernels`] on cubecl backends, with
//! its Fusion routing.

use burn::backend::{Backend, backend_extension, tensor::FloatTensor};
use burn::tensor::{DType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use super::kernels::*;

/// Output shape of [`TfOps::tf_linear`]: `[M, N]` for `x` `[M, K]` and `w` `[K, N]`.
pub fn linear_shape(x: &Shape, w: &Shape) -> Shape {
  Shape::new([x[0], w[1]])
}

/// Output shape of [`TfOps::tf_attention`]: the queries' `[N, H, S]` with the values' head size.
pub fn attention_shape(q: &Shape, v: &Shape) -> Shape {
  Shape::new([q[0], q[1], q[2], v[3]])
}

#[allow(clippy::too_many_arguments)]
#[backend_extension(Cube, Fusion)]
pub trait TfOps: Backend {
  /// `x · w (+ bias)` for `x` `[M, K]`, `w` `[K, N]`, `bias` `[N]` (read only with `has_bias`).
  #[fusion(dtype = x, shape = linear_shape(x, w))]
  fn tf_linear(x: FloatTensor<Self>, w: FloatTensor<Self>, bias: FloatTensor<Self>, has_bias: bool) -> FloatTensor<Self>;

  /// RMSNorm of each row of `x` `[R, C]` times `gamma` `[C]` and `mask` `[R]` (`rms_norm_kernel`); `divisor` and `eps`
  /// are 1-element tensors.
  #[fusion(dtype = x, shape = x.clone())]
  fn tf_rms_norm(
    x: FloatTensor<Self>,
    gamma: FloatTensor<Self>,
    mask: FloatTensor<Self>,
    divisor: FloatTensor<Self>,
    eps: FloatTensor<Self>,
  ) -> FloatTensor<Self>;

  /// `softmax(q kᵗ * scale + bias) v` per head for `q` `[N, H, S, dq]`, `k` `[N, Hkv, S, dq]`, `v` `[N, Hkv, S, dv]`
  /// and the `[N, S]` key bias; `scale_bits` is the f32 scale's bits.
  #[fusion(dtype = q, shape = attention_shape(q, v))]
  fn tf_attention(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, bias: FloatTensor<Self>, scale_bits: u32) -> FloatTensor<Self>;
}

fn contiguous(t: CubeTensor) -> CubeTensor {
  if t.is_contiguous() { t } else { into_contiguous(t) }
}

fn len(t: &CubeTensor) -> usize {
  t.meta.shape().num_elements()
}

/// Rows per grid dimension (65535 workgroups each).
const GRID_X: usize = 32768;

impl TfOps for CubeBackend {
  fn tf_linear(x: CubeTensor, w: CubeTensor, bias: CubeTensor, has_bias: bool) -> CubeTensor {
    let x = contiguous(x);
    let w = contiguous(w);
    let bias = contiguous(bias);
    let (m, k) = (x.meta.shape()[0], x.meta.shape()[1]);
    let (wk, n) = (w.meta.shape()[0], w.meta.shape()[1]);
    assert_eq!(k, wk, "tf_linear: x is [{m}, {k}], w [{wk}, {n}]");
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::new([m, n]), x.dtype);
    macro_rules! launch {
      ($f:ty) => {
        // SAFETY: contiguous buffers of the lengths given; the kernel bounds-checks every edge.
        unsafe {
          linear_gemm::launch::<$f>(
            &x.client,
            CubeCount::Static(m.div_ceil(LIN_M) as u32, n.div_ceil(LIN_N) as u32, 1),
            CubeDim::new_2d(LIN_T as u32, LIN_T as u32),
            BufferArg::from_raw_parts(x.handle.clone(), len(&x)),
            BufferArg::from_raw_parts(w.handle.clone(), len(&w)),
            BufferArg::from_raw_parts(bias.handle.clone(), len(&bias)),
            BufferArg::from_raw_parts(out.handle.clone(), m * n),
            m as u32,
            n as u32,
            k as u32,
            has_bias,
          )
        }
      };
    }
    match x.dtype {
      DType::F32 => launch!(f32),
      DType::F16 => launch!(half::f16),
      other => panic!("tf_linear: unsupported dtype {other:?}"),
    }
    out
  }

  fn tf_rms_norm(x: CubeTensor, gamma: CubeTensor, mask: CubeTensor, divisor: CubeTensor, eps: CubeTensor) -> CubeTensor {
    let x = contiguous(x);
    let (gamma, mask, divisor, eps) = (contiguous(gamma), contiguous(mask), contiguous(divisor), contiguous(eps));
    let (rows, c) = (x.meta.shape()[0], x.meta.shape()[1]);
    assert_eq!(len(&gamma), c, "tf_rms_norm: gamma");
    assert_eq!(len(&mask), rows, "tf_rms_norm: mask");
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::new([rows, c]), x.dtype);
    let (gx, gy) = (rows.min(GRID_X), rows.div_ceil(GRID_X));
    assert!(gy == 1 || rows.is_multiple_of(GRID_X), "tf_rms_norm: {rows} rows");
    macro_rules! launch {
      ($f:ty) => {
        // SAFETY: contiguous `[rows, c]` input and output; one workgroup per row.
        unsafe {
          rms_norm_kernel::launch::<$f>(
            &x.client,
            CubeCount::Static(gx as u32, gy as u32, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(x.handle.clone(), rows * c),
            BufferArg::from_raw_parts(gamma.handle.clone(), c),
            BufferArg::from_raw_parts(mask.handle.clone(), rows),
            BufferArg::from_raw_parts(divisor.handle.clone(), 1),
            BufferArg::from_raw_parts(eps.handle.clone(), 1),
            BufferArg::from_raw_parts(out.handle.clone(), rows * c),
            c as u32,
            gx as u32,
          )
        }
      };
    }
    match x.dtype {
      DType::F32 => launch!(f32),
      DType::F16 => launch!(half::f16),
      other => panic!("tf_rms_norm: unsupported dtype {other:?}"),
    }
    out
  }

  fn tf_attention(q: CubeTensor, k: CubeTensor, v: CubeTensor, bias: CubeTensor, scale_bits: u32) -> CubeTensor {
    let (q, k, v, bias) = (contiguous(q), contiguous(k), contiguous(v), contiguous(bias));
    let qd = q.meta.shape().to_vec();
    let kd = k.meta.shape().to_vec();
    let vd = v.meta.shape().to_vec();
    let (n, heads, seq, dq) = (qd[0], qd[1], qd[2], qd[3]);
    let (kv_heads, dv) = (kd[1], vd[3]);
    assert!(
      kd == [n, kv_heads, seq, dq] && vd[..3] == [n, kv_heads, seq],
      "tf_attention: q {qd:?}, k {kd:?}, v {vd:?}"
    );
    assert!(heads.is_multiple_of(kv_heads), "tf_attention: {heads} heads, {kv_heads} kv heads");
    assert_eq!(len(&bias), n * seq, "tf_attention: bias");
    let out = empty_device_contiguous_dtype(q.client.clone(), q.device.clone(), Shape::new([n, heads, seq, dv]), q.dtype);
    let scale = f32::from_bits(scale_bits);
    macro_rules! launch {
      ($f:ty) => {
        // SAFETY: contiguous buffers of the shapes asserted above; the kernel bounds-checks the sequence.
        unsafe {
          attention_kernel::launch::<$f>(
            &q.client,
            CubeCount::Static(seq.div_ceil(ATT_Q) as u32, heads as u32, n as u32),
            CubeDim::new_1d(ATT_Q as u32),
            BufferArg::from_raw_parts(q.handle.clone(), len(&q)),
            BufferArg::from_raw_parts(k.handle.clone(), len(&k)),
            BufferArg::from_raw_parts(v.handle.clone(), len(&v)),
            BufferArg::from_raw_parts(bias.handle.clone(), n * seq),
            BufferArg::from_raw_parts(out.handle.clone(), n * heads * seq * dv),
            seq as u32,
            heads as u32,
            (heads / kv_heads) as u32,
            scale,
            dq,
            dv,
          )
        }
      };
    }
    match q.dtype {
      DType::F32 => launch!(f32),
      DType::F16 => launch!(half::f16),
      other => panic!("tf_attention: unsupported dtype {other:?}"),
    }
    out
  }
}

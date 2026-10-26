//! The burn backend extension ([`ConvOps`]) that launches the kernels of [`super::kernels`] on cubecl
//! backends, with its Fusion routing.

use burn::backend::{Backend, backend_extension, tensor::FloatTensor};
use burn::tensor::{DType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use super::kernels::*;

/// Output shape of [`ConvOps::reduce_last`]: the input without its last axis.
pub fn reduce_last_shape(x: &Shape) -> Shape {
  Shape::new([x[0], x[1]])
}

/// Output shape of [`ConvOps::wino_conv3x3`] and [`ConvOps::conv1x1`]: `[N, oc, H, W]` for an input `[N, ic, H, W]`.
pub fn wino_out_shape(x: &Shape, oc: &usize) -> Shape {
  Shape::new([x[0], *oc, x[2], x[3]])
}

/// Output dtype of [`ConvOps::wino_filter_3x3`].
pub fn wino_filter_dtype(weight: DType, half: &bool) -> DType {
  if *half { DType::F16 } else { weight }
}

/// Output shape of [`ConvOps::wino_filter_3x3`] for a `[oc, ic, 3, 3]` weight.
pub fn wino_filter_shape(weight: &Shape) -> Shape {
  Shape::new([TILE_POINTS * round_up(weight[1], GEMM_K) * round_up(weight[0], GEMM_N)])
}

#[backend_extension(Cube, Fusion)]
pub trait ConvOps: Backend {
  /// Stride-1 "same" 3x3 convolution, no bias. `filter` is [`transform_filter_3x3`]'s output for `oc` output
  /// channels.
  #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
  fn wino_conv3x3(x: FloatTensor<Self>, filter: FloatTensor<Self>, oc: usize) -> FloatTensor<Self>;

  /// [`transform_filter_3x3`] on the device: `[oc, ic, 3, 3]` weight to the `[36, icPad, ocPad]` filter.
  /// The filter is stored in f16 with `half` (the conv then transforms and multiplies in f16).
  #[fusion(dtype = wino_filter_dtype(weight, half), shape = wino_filter_shape(weight))]
  fn wino_filter_3x3(weight: FloatTensor<Self>, half: bool) -> FloatTensor<Self>;

  /// Sum (or max, with `max`) of a `[N, C, L]` tensor over its last axis.
  #[fusion(dtype = x, shape = reduce_last_shape(x))]
  fn reduce_last(x: FloatTensor<Self>, max: bool) -> FloatTensor<Self>;

  /// 1x1 convolution, no bias. `wt` is the `[ic, oc]` transposed weight; `ic % GEMM_K == 0`.
  #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
  fn conv1x1(x: FloatTensor<Self>, wt: FloatTensor<Self>, oc: usize) -> FloatTensor<Self>;

  /// [`ConvOps::conv1x1`] of `mish(x * scale + bias) * mask`: `scale`, `bias` `[ic]`, `mask` `[N * H * W]`.
  #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
  fn conv1x1_bn_act(
    x: FloatTensor<Self>,
    wt: FloatTensor<Self>,
    scale: FloatTensor<Self>,
    bias: FloatTensor<Self>,
    mask: FloatTensor<Self>,
    oc: usize,
  ) -> FloatTensor<Self>;

  /// [`ConvOps::wino_conv3x3`] of `mish(x * scale + bias) * mask` (as [`ConvOps::conv1x1_bn_act`]).
  #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
  fn wino_conv3x3_bn_act(
    x: FloatTensor<Self>,
    filter: FloatTensor<Self>,
    scale: FloatTensor<Self>,
    bias: FloatTensor<Self>,
    mask: FloatTensor<Self>,
    oc: usize,
  ) -> FloatTensor<Self>;
}

fn contiguous(t: CubeTensor) -> CubeTensor {
  if t.is_contiguous() { t } else { into_contiguous(t) }
}

impl ConvOps for CubeBackend {
  fn reduce_last(x: CubeTensor, max: bool) -> CubeTensor {
    let x = contiguous(x);
    let dims = x.meta.shape().to_vec();
    let (rows, len) = (dims[0] * dims[1], dims[2]);
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::new([dims[0], dims[1]]), x.dtype);
    macro_rules! launch {
      ($f:ty) => {
        // SAFETY: contiguous `[rows, len]` input and `[rows]` output; one workgroup per row.
        unsafe {
          reduce_last_kernel::launch::<$f>(
            &x.client,
            CubeCount::Static(dims[1] as u32, dims[0] as u32, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(x.handle.clone(), rows * len),
            BufferArg::from_raw_parts(out.handle.clone(), rows),
            len as u32,
            dims[1] as u32,
            max,
          )
        }
      };
    }
    match x.dtype {
      DType::F32 => launch!(f32),
      DType::F16 => launch!(half::f16),
      other => panic!("reduce_last: unsupported dtype {other:?}"),
    }
    out
  }

  fn wino_filter_3x3(weight: CubeTensor, half: bool) -> CubeTensor {
    let weight = contiguous(weight);
    let dims = weight.meta.shape().to_vec();
    let (oc, ic) = (dims[0], dims[1]);
    assert_eq!(&dims[2..], &[3, 3], "wino_filter_3x3: not a 3x3 weight");
    let (oc_pad, ic_pad) = (round_up(oc, GEMM_N), round_up(ic, GEMM_K));
    let len = TILE_POINTS * ic_pad * oc_pad;
    let out = empty_device_contiguous_dtype(
      weight.client.clone(),
      weight.device.clone(),
      Shape::new([len]),
      wino_filter_dtype(weight.dtype, &half),
    );
    macro_rules! launch {
      ($f:ty, $s:ty) => {
        // SAFETY: contiguous buffers of the lengths given; threads past the padded sizes return.
        unsafe {
          wino_filter::launch::<$f, $s>(
            &weight.client,
            CubeCount::Static(oc_pad.div_ceil(64) as u32, ic_pad as u32, 1),
            CubeDim::new_2d(64, 1),
            BufferArg::from_raw_parts(weight.handle.clone(), oc * ic * 9),
            BufferArg::from_raw_parts(out.handle.clone(), len),
            oc as u32,
            ic as u32,
            oc_pad as u32,
            ic_pad as u32,
          )
        }
      };
    }
    match (weight.dtype, out.dtype) {
      (DType::F32, DType::F32) => launch!(f32, f32),
      (DType::F32, DType::F16) => launch!(f32, half::f16),
      (DType::F16, DType::F16) => launch!(half::f16, half::f16),
      other => panic!("wino_filter_3x3: unsupported dtypes {other:?}"),
    }
    out
  }

  fn conv1x1(x: CubeTensor, wt: CubeTensor, oc: usize) -> CubeTensor {
    conv1x1_impl(x, wt, oc, None)
  }

  fn conv1x1_bn_act(x: CubeTensor, wt: CubeTensor, scale: CubeTensor, bias: CubeTensor, mask: CubeTensor, oc: usize) -> CubeTensor {
    conv1x1_impl(x, wt, oc, Some(Prologue::new(scale, bias, mask)))
  }

  fn wino_conv3x3(x: CubeTensor, filter: CubeTensor, oc: usize) -> CubeTensor {
    wino_conv3x3_impl(x, filter, oc, None)
  }

  fn wino_conv3x3_bn_act(x: CubeTensor, filter: CubeTensor, scale: CubeTensor, bias: CubeTensor, mask: CubeTensor, oc: usize) -> CubeTensor {
    wino_conv3x3_impl(x, filter, oc, Some(Prologue::new(scale, bias, mask)))
  }
}

/// The `mish(x * scale + bias) * mask` that KataGo applies before every conv (batch norm, activation, board mask),
/// fused into the conv's input loads: `scale`, `bias` per input channel, `mask` per board point of each row.
struct Prologue {
  scale: CubeTensor,
  bias: CubeTensor,
  mask: CubeTensor,
}

impl Prologue {
  fn new(scale: CubeTensor, bias: CubeTensor, mask: CubeTensor) -> Self {
    Prologue {
      scale: contiguous(scale),
      bias: contiguous(bias),
      mask: contiguous(mask),
    }
  }

  /// (scale, bias, mask) buffer arguments, or 1-element placeholders on `x` without a prologue.
  fn args(pro: &Option<Prologue>, x: &CubeTensor) -> [(cubecl::server::Handle, usize); 3] {
    let len = |t: &CubeTensor| t.meta.shape().num_elements();
    match pro {
      Some(p) => [
        (p.scale.handle.clone(), len(&p.scale)),
        (p.bias.handle.clone(), len(&p.bias)),
        (p.mask.handle.clone(), len(&p.mask)),
      ],
      None => [(x.handle.clone(), 1), (x.handle.clone(), 1), (x.handle.clone(), 1)],
    }
  }
}

/// [`ConvOps::conv1x1`], with the `mish(x * scale + bias) * mask` prologue of `pro` applied to its input.
fn conv1x1_impl(x: CubeTensor, wt: CubeTensor, oc: usize, pro: Option<Prologue>) -> CubeTensor {
  let x = contiguous(x);
  let wt = contiguous(wt);
  let dims = x.meta.shape().to_vec();
  let (n, ic, h, w) = (dims[0], dims[1], dims[2], dims[3]);
  assert!(ic.is_multiple_of(GEMM_K), "conv1x1: ic {ic} not a multiple of {GEMM_K}");
  assert_eq!(wt.meta.shape().num_elements(), ic * oc, "conv1x1 weight size");
  let dtype = x.dtype;
  let output = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::new([n, oc, h, w]), dtype);
  let m = h * w;
  let (x_len, w_len, o_len) = (x.meta.shape().num_elements(), ic * oc, n * oc * m);
  let pargs = Prologue::args(&pro, &x);
  macro_rules! launch {
    ($f:ty) => {
      // SAFETY: contiguous buffers of the lengths given; the kernel bounds-checks m and needs ic, oc
      // padded, as asserted above.
      unsafe {
        conv1x1_gemm::launch::<$f>(
          &x.client,
          CubeCount::Static(m.div_ceil(64) as u32, oc.div_ceil(GEMM_N) as u32, n as u32),
          CubeDim::new_2d(GEMM_T as u32, GEMM_T as u32),
          BufferArg::from_raw_parts(x.handle.clone(), x_len),
          BufferArg::from_raw_parts(wt.handle.clone(), w_len),
          BufferArg::from_raw_parts(pargs[0].0.clone(), pargs[0].1),
          BufferArg::from_raw_parts(pargs[1].0.clone(), pargs[1].1),
          BufferArg::from_raw_parts(pargs[2].0.clone(), pargs[2].1),
          BufferArg::from_raw_parts(output.handle.clone(), o_len),
          m as u32,
          oc as u32,
          ic as u32,
          pro.is_some(),
        )
      }
    };
  }
  match dtype {
    DType::F32 => launch!(f32),
    DType::F16 => launch!(half::f16),
    other => panic!("conv1x1: unsupported dtype {other:?}"),
  }
  output
}

/// [`ConvOps::wino_conv3x3`], with the prologue of `pro` applied to its input in the transform.
fn wino_conv3x3_impl(x: CubeTensor, filter: CubeTensor, oc: usize, pro: Option<Prologue>) -> CubeTensor {
  let x = contiguous(x);
  let filter = contiguous(filter);
  let dims = x.meta.shape().to_vec();
  let (n, ic, h, w) = (dims[0], dims[1], dims[2], dims[3]);
  let dtype = x.dtype;
  let client = x.client.clone();
  let device = x.device.clone();

  let tiles_x = w.div_ceil(OUT_TILE);
  let tiles_y = h.div_ceil(OUT_TILE);
  let nt = n * tiles_x * tiles_y;
  let nt_pad = round_up(nt, GEMM_M);
  let ic_pad = round_up(ic, GEMM_K);
  let oc_pad = round_up(oc, GEMM_N);
  assert_eq!(filter.meta.shape().num_elements(), TILE_POINTS * ic_pad * oc_pad, "wino filter size");

  // Transformed inputs, the GEMM and its output are in the filter's dtype (f16 when the filter is f16).
  let store = filter.dtype;
  let transformed = empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([TILE_POINTS * ic_pad * nt_pad]), store);
  // f16: the vectorized GEMM (packed f16 math: +20% end to end on an RX 6800 XT), its result written in f32 (as
  // cheap, and b18's ownership error 2.3e-2 -> 1.3e-2 in nn_parity). f32: the scalar GEMM (the vector one is 2-3%
  // slower there).
  let vector = store == DType::F16;
  let gemm_store = if vector { DType::F32 } else { store };
  let gemm_out = empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([TILE_POINTS * oc_pad * nt_pad]), gemm_store);
  let output = empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([n, oc, h, w]), dtype);

  let len = |t: &CubeTensor| t.meta.shape().num_elements();

  let pargs = Prologue::args(&pro, &x);
  macro_rules! launch_all {
    ($f:ty, $s:ty, $o:ty) => {{
      let x_len = len(&x);
      let t_len = len(&transformed);
      let g_len = len(&gemm_out);
      let f_len = len(&filter);
      let o_len = len(&output);
      // SAFETY: every buffer is contiguous and as long as the length given; the kernels index within
      // the padded sizes they are given.
      unsafe {
        wino_transform::launch::<$f, $s>(
          &client,
          CubeCount::Static((nt_pad / 64) as u32, ic_pad as u32, 1),
          CubeDim::new_2d(64, 1),
          BufferArg::from_raw_parts(x.handle.clone(), x_len),
          BufferArg::from_raw_parts(pargs[0].0.clone(), pargs[0].1),
          BufferArg::from_raw_parts(pargs[1].0.clone(), pargs[1].1),
          BufferArg::from_raw_parts(pargs[2].0.clone(), pargs[2].1),
          BufferArg::from_raw_parts(transformed.handle.clone(), t_len),
          n as u32,
          w as u32,
          h as u32,
          tiles_x as u32,
          tiles_y as u32,
          ic as u32,
          ic_pad as u32,
          nt_pad as u32,
          pro.is_some(),
        );
        if vector {
          wino_gemm_v::launch::<$s, $o>(
            &client,
            CubeCount::Static((nt_pad / GEMM_M) as u32, (oc_pad / GEMM_N) as u32, TILE_POINTS as u32),
            CubeDim::new_2d(GEMM_T as u32, GEMM_T as u32),
            4,
            BufferArg::from_raw_parts(transformed.handle.clone(), t_len / 4),
            BufferArg::from_raw_parts(filter.handle.clone(), f_len / 4),
            BufferArg::from_raw_parts(gemm_out.handle.clone(), g_len / 4),
            nt_pad as u32,
            oc_pad as u32,
            ic_pad as u32,
          );
        } else {
          wino_gemm::launch::<$s>(
            &client,
            CubeCount::Static((nt_pad / GEMM_M) as u32, (oc_pad / GEMM_N) as u32, TILE_POINTS as u32),
            CubeDim::new_2d(GEMM_T as u32, GEMM_T as u32),
            BufferArg::from_raw_parts(transformed.handle.clone(), t_len),
            BufferArg::from_raw_parts(filter.handle.clone(), f_len),
            BufferArg::from_raw_parts(gemm_out.handle.clone(), g_len),
            nt_pad as u32,
            oc_pad as u32,
            ic_pad as u32,
          );
        }
        let per_row = oc * tiles_x * tiles_y;
        wino_untransform::launch::<$o, $f>(
          &client,
          CubeCount::Static(per_row.div_ceil(64) as u32, n as u32, 1),
          CubeDim::new_1d(64),
          BufferArg::from_raw_parts(gemm_out.handle.clone(), g_len),
          BufferArg::from_raw_parts(output.handle.clone(), o_len),
          n as u32,
          w as u32,
          h as u32,
          tiles_x as u32,
          tiles_y as u32,
          oc as u32,
          oc_pad as u32,
          nt_pad as u32,
        );
      }
    }};
  }
  match (dtype, store, gemm_store) {
    (DType::F32, DType::F32, DType::F32) => launch_all!(f32, f32, f32),
    (DType::F32, DType::F16, DType::F32) => launch_all!(f32, half::f16, f32),
    other => panic!("wino_conv3x3: unsupported dtypes {other:?}"),
  }
  output
}

//! Winograd F(4x4, 3x3) convolution on cubecl, following KataGo's OpenCL backend
//! (upstream cpp/neuralnet/openclkernels.cpp `winogradTransformNCHW`, `winogradBNActTransformNCHW`,
//! `winogradUntransformNCHW`, and the 3x3 branch of `ConvLayer` in openclbackend.cpp): the input is cut into
//! 6x6 tiles (offset -1, stride 4), each tile transformed (B^T d B), one GEMM per tile point multiplies the
//! transformed inputs by the transformed filters (G g G^T, done once on the host), and each 6x6 result is
//! untransformed (A^T m A) into a 4x4 output tile.
//!
//! Layouts are upstream's: transformed inputs `[36, icPad, ntPad]`, filters `[36, icPad, ocPad]`, GEMM output
//! `[36, ocPad, ntPad]`, where `nt = N * tilesY * tilesX` is padded to `GEMM_M`, `ic` to `GEMM_K` and `oc` to
//! `GEMM_N`. The transforms are upstream's active ones (points 0, ±1, ±2), not the commented-out low-error set.
//!
//! [`kernels`] holds the cubecl kernels, [`ext`] the burn backend extension that launches them ([`ConvOps`]),
//! this module the per-conv entry points `conv.rs` calls and the process-wide precision switch.

mod ext;
mod kernels;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::backend::Dispatch;
use burn::module::ParamId;
use burn::nn::conv::Conv2d;
use burn::tensor::{Device, Tensor};

pub use ext::ConvOps;
#[doc(hidden)]
pub use kernels::{GEMM_K, GEMM_M, GEMM_N, IN_TILE, OUT_TILE, TILE_POINTS, transform_filter_3x3};

/// One device's transformed filters, by (weight parameter, f16).
type DeviceFilters = (Device, HashMap<(ParamId, bool), Tensor<1>>);

/// The default precision of the 3x3 convs ([`set_half`]): what a network runs in unless its forward pass is
/// inside [`with_half`]. Process-wide.
static HALF: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The precision of the forward pass being built on this thread ([`with_half`]), over [`HALF`].
    static FORWARD_HALF: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Transformed filters, built on a conv's first call with device ops only: per device (burn's Device is not
    /// Hash, and a thread uses one or two), by weight parameter and precision. burn-onnx fixes a net's ParamIds at
    /// build time, so every load of the same net shares them (and its weights: the same `.bpk`).
    static FILTERS: RefCell<Vec<DeviceFilters>> = const { RefCell::new(Vec::new()) };
}

/// Run the 3x3 convs in f16 from now on, by default (the device must support f16; filters already built keep their
/// dtype, so call this before the first forward pass). A network with its own precision builds its forward pass
/// inside [`with_half`] instead.
pub fn set_half(half: bool) {
    // (filters are cached per precision, so switching never reuses the other precision's)
    HALF.store(half, Ordering::Relaxed);
}

/// The cached device-side form of `conv`'s weight (kept for the thread's lifetime: one entry per conv of each
/// net, precision and device used), built by `make` from the `[oc, ic, kh, kw]` weight with lazy
/// device ops only (wasm cannot read a tensor back synchronously).
fn cached_filter(conv: &Conv2d, make: impl FnOnce(Tensor<4>) -> Tensor<1>) -> Tensor<1> {
    let w = conv.weight.val();
    let device = w.device();
    let key = (conv.weight.id, half());
    FILTERS.with(|filters| {
        let mut filters = filters.borrow_mut();
        let i = match filters.iter().position(|(d, _)| *d == device) {
            Some(i) => i,
            None => {
                filters.push((device, HashMap::new()));
                filters.len() - 1
            }
        };
        filters[i].1.entry(key).or_insert_with(|| make(w)).clone()
    })
}

/// Whether [`conv1x1`] takes this 1x1 conv: every one of KataGo's nets but a 19- or 22-channel input.
pub fn conv1x1_fits(_oc: usize, ic: usize) -> bool {
    ic.is_multiple_of(GEMM_K)
}

/// `conv2d_same` for a 1x1 kernel with [`conv1x1_fits`] on a cubecl device.
pub fn conv1x1(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let [oc, ic, _, _] = conv.weight.dims();
    let wt = cached_filter(conv, |w| w.reshape([oc, ic]).swap_dims(0, 1).reshape([ic * oc]));
    Tensor::<4>::from_dispatch(Dispatch::conv1x1(x.into_dispatch(), wt.into_dispatch(), oc))
}

/// `conv2d_bn_act` (conv.rs) for a 1x1 kernel with [`conv1x1_fits`] on a cubecl device.
pub fn conv1x1_bn_act(conv: &Conv2d, x: Tensor<4>, scale: Tensor<4>, bias: Tensor<4>, mask: Tensor<4>) -> Tensor<4> {
    let [oc, ic, _, _] = conv.weight.dims();
    let wt = cached_filter(conv, |w| w.reshape([oc, ic]).swap_dims(0, 1).reshape([ic * oc]));
    let (scale, bias, mask) = prologue_args(scale, bias, mask);
    Tensor::<4>::from_dispatch(Dispatch::conv1x1_bn_act(
        x.into_dispatch(),
        wt.into_dispatch(),
        scale.into_dispatch(),
        bias.into_dispatch(),
        mask.into_dispatch(),
        oc,
    ))
}

/// `conv2d_bn_act` (conv.rs) for a 3x3 kernel on a cubecl device.
pub fn conv3x3_bn_act(conv: &Conv2d, x: Tensor<4>, scale: Tensor<4>, bias: Tensor<4>, mask: Tensor<4>) -> Tensor<4> {
    let [oc, _, _, _] = conv.weight.dims();
    let filter = cached_filter(conv, |w| Tensor::<1>::from_dispatch(Dispatch::wino_filter_3x3(w.into_dispatch(), half())));
    let (scale, bias, mask) = prologue_args(scale, bias, mask);
    Tensor::<4>::from_dispatch(Dispatch::wino_conv3x3_bn_act(
        x.into_dispatch(),
        filter.into_dispatch(),
        scale.into_dispatch(),
        bias.into_dispatch(),
        mask.into_dispatch(),
        oc,
    ))
}

/// `[1, C, 1, 1]` scale and bias and the `[N, 1, H, W]` mask, flattened for the prologue kernels.
fn prologue_args(scale: Tensor<4>, bias: Tensor<4>, mask: Tensor<4>) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
    let c = scale.dims()[1];
    let points = mask.dims().iter().product::<usize>();
    (scale.reshape([c]), bias.reshape([c]), mask.reshape([points]))
}

/// `conv2d_same` for a 3x3 kernel on a cubecl device.
pub fn conv3x3(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let [oc, _, _, _] = conv.weight.dims();
    let filter = cached_filter(conv, |w| Tensor::<1>::from_dispatch(Dispatch::wino_filter_3x3(w.into_dispatch(), half())));
    Tensor::<4>::from_dispatch(Dispatch::wino_conv3x3(x.into_dispatch(), filter.into_dispatch(), oc))
}

/// [`ConvOps::reduce_last`] on a `[N, C, L]` tensor.
pub fn reduce_last(x: Tensor<3>, max: bool) -> Tensor<2> {
    Tensor::<2>::from_dispatch(Dispatch::reduce_last(x.into_dispatch(), max))
}

/// Whether the 3x3 convs built now run in f16: [`with_half`]'s precision inside it, else [`set_half`]'s.
pub fn half() -> bool {
    FORWARD_HALF.with(|h| h.get()).unwrap_or_else(|| HALF.load(Ordering::Relaxed))
}

/// The process default of [`set_half`], whatever [`with_half`] is in effect.
pub fn default_half() -> bool {
    HALF.load(Ordering::Relaxed)
}

/// Runs `f` (a network's forward pass) with its 3x3 convs in f16 or not. The precision is read while the graph is
/// built, which happens inside `f` on this thread (the kernels may run later on another), so each network of a
/// process can have its own.
pub fn with_half<R>(half: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORWARD_HALF.with(|h| h.set(self.0));
        }
    }
    let _restore = Restore(FORWARD_HALF.with(|h| h.replace(Some(half))));
    f()
}

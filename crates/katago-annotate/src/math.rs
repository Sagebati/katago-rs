//! KATAGO-ANNOTATE (new): the one transcendental of the annotation, `expf` (`move_accuracy`).
//!
//! katago-ws calls `f32::exp`, which natively is glibc's `expf`. On wasm32-unknown-unknown Rust's `f32::exp`
//! is compiler-builtins' libm (FreeBSD-derived), which differs from glibc in the last ulp for some inputs
//! (seen in `report.*.accuracy` of fixtures/katago-annotate/hand-9x9). There it is katago-core's port of
//! glibc's `__expf_fma` (`port::math::expf`, checked against glibc), so a payload built in the browser is
//! byte-identical to katago-ws's on an x86-64 server with FMA. Natively it stays `f32::exp`: the same libm as
//! katago-ws on the same machine.

/// `expf`, as glibc computes it.
#[cfg(target_arch = "wasm32")]
pub(crate) fn expf(x: f32) -> f32 {
    katago_core::port::math::expf(x)
}

/// `expf`, as glibc computes it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn expf(x: f32) -> f32 {
    x.exp()
}

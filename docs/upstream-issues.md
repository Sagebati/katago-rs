# Upstream issue drafts (not filed)

Found during M0. Review before filing; nothing has been posted.

## 1. cubecl-wgpu: WGSL kernels fail in Chrome — `extension 'subgroups' is not allowed`

**Repo:** tracel-ai/cubecl (cubecl-wgpu 0.11.2), possibly also gfx-rs/wgpu (30.0.1).

On `wasm32` with the WebGPU backend, matmul kernels (e.g.
`matmul_entry_lhs_f32_lhs_size_2_rhs_f32_rhs_size_2_acc_f32_acc_size_1`) fail shader creation in
Chromium 155 on an AMD RDNA2 adapter that *does* advertise `subgroups`:

```
1:8 extension 'subgroups' is not allowed in the current environment
1: enable subgroups;
```

then `[Invalid ShaderModule ...]` and the device is poisoned
(`cubecl-wgpu/src/compute/device_poison.rs:42`).

Cause, as far as we can tell:
- `compiler/wgsl/shader.rs` (~L199, `#[cfg(target_family = "wasm")]`) adds `enable subgroups;`
  whenever a kernel reads `SubgroupSize`/`SubgroupId`/`SubgroupInvocationId` builtins (here via
  `PlaneDim`), regardless of whether plane features were registered.
- The device is requested with `adapter.features()` (`backend/wgsl.rs`), but wgpu's WebGPU
  backend `FEATURES_MAPPING` (`wgpu/src/backend/webgpu.rs`) has no entry for `subgroups`, so
  the browser device never gets the feature, and `Plane::Ops` is (correctly) not registered.

So kernels that don't use plane *ops* still emit the directive because they read the plane size.

Workaround (page-level): patch `GPUAdapter.prototype.requestDevice` to add `"subgroups"` to
`requiredFeatures` when `adapter.features.has("subgroups")`.

Possible fixes: only emit the directive when the device has the feature (and lower `PlaneDim`
to a constant/uniform otherwise), or have wgpu map `Features::SUBGROUP` ↔ `"subgroups"` on web.

Repro: burn 0.22.1 + `webgpu` feature, any model with a matmul, `Device::wgpu_options().init_async()`
in Chrome with WebGPU enabled.

## 2. macerator 0.5.1: build fails on wasm32 with `+relaxed-simd`

**Repo:** wingertge/macerator (pulled in by burn-flex `simd`).

`RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --target wasm32-unknown-unknown`:

```
error[E0053]: method `run_vectorized` has an incompatible type for trait
   --> macerator-0.5.1/src/backend/wasm32.rs:176:9
    |  fn run_vectorized<F: NullaryFnOnce>(f: F) -> F::Output {
    |  ^ expected unsafe fn, found safe fn
note: type in trait
   --> macerator-0.5.1/src/backend/wasm32.rs:207:5
    |  unsafe fn run_vectorized<F: NullaryFnOnce>(f: F) -> F::Output;
```

The `Simd128Relaxed` impl of `Simd128Run` (only compiled with `relaxed-simd`) declares the
method safe while the trait declares it `unsafe`. Marking the impl method `unsafe fn` should fix it.

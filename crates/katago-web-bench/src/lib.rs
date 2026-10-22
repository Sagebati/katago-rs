//! Browser side of the M0 spike: load the b10c128 weights, check raw outputs
//! against onnxruntime fixtures, and time forward passes per batch size.

use burn::prelude::*;
use burn::tensor::Bytes;
use katago_nn_burn::b10c128::Model;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn now_ms() -> f64 {
    js_sys::Reflect::get(&js_sys::global(), &"performance".into())
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
        .map(|p| p.now())
        .unwrap_or_else(js_sys::Date::now)
}

fn log(s: &str) {
    web_sys::console::log_1(&s.into());
}

thread_local! {
    // cubecl registers one server per device; initialise WebGPU once per page.
    static WEBGPU: std::cell::RefCell<Option<Device>> = const { std::cell::RefCell::new(None) };
}

async fn device(backend: &str) -> Result<Device, JsError> {
    match backend {
        "webgpu" => {
            if let Some(d) = WEBGPU.with_borrow(|d| d.clone()) {
                return Ok(d);
            }
            let d = Device::wgpu_options()
                .init_async()
                .await
                .map_err(|e| JsError::new(&format!("WebGPU init: {e:?}")))?;
            WEBGPU.set(Some(d.clone()));
            Ok(d)
        }
        "flex" => Ok(Device::flex()),
        other => Err(JsError::new(&format!("unknown backend {other}"))),
    }
}

async fn read_all(t: Tensor<2>) -> Vec<f32> {
    t.into_data_async().await.expect("readback").try_to_vec::<f32>().expect("f32")
}

/// Max |burn - onnxruntime| over the raw outputs, per output.
#[wasm_bindgen]
pub async fn parity(
    backend: String,
    bpk: Vec<u8>,
    bin: Vec<f32>,
    global: Vec<f32>,
    policy: Vec<f32>,
    value: Vec<f32>,
    score: Vec<f32>,
    ownership: Vec<f32>,
) -> Result<String, JsError> {
    log(&format!("parity[{backend}]: device init"));
    let device = device(&backend).await?;
    log("parity: loading weights");
    let model = Model::from_bytes(Bytes::from_bytes_vec(bpk), &device);
    log("parity: weights loaded");
    let n = bin.len() / (22 * 19 * 19);
    let bin = Tensor::<4>::from_data(TensorData::new(bin, [n, 22, 19, 19]), &device);
    let global = Tensor::<2>::from_data(TensorData::new(global, [n, 19]), &device);
    log("parity: forward");
    let (p, v, s, o) = model.forward(bin, global);
    log("parity: forward queued");
    let mut out = format!("{backend}: {n} positions, raw outputs vs onnxruntime\n");
    let mut worst = 0f32;
    for (name, got, want) in [
        ("policy", p.flatten::<2>(1, 2), policy),
        ("value", v, value),
        ("scorevalue", s, score),
        ("ownership", o.flatten::<2>(1, 3), ownership),
    ] {
        log(&format!("parity: reading {name}"));
        let got = read_all(got).await;
        let d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        worst = worst.max(d);
        out += &format!("  {name:<10} max |diff| {d:.3e}\n");
    }
    let tol = if backend == "webgpu" { 2e-3 } else { 1e-4 };
    out += if worst <= tol { "PASS\n" } else { "FAIL\n" };
    Ok(out)
}

/// Forward-pass throughput; one readback per batch, like the engine.
#[wasm_bindgen]
pub async fn bench(backend: String, bpk: Vec<u8>, batches: Vec<u32>, seconds: f64) -> Result<String, JsError> {
    let t0 = now_ms();
    let device = device(&backend).await?;
    let model = Model::from_bytes(Bytes::from_bytes_vec(bpk), &device);
    let mut out = format!("{backend}: init+load {:.0} ms\n", now_ms() - t0);
    for n in batches {
        let n = n as usize;
        let bin = Tensor::<4>::random([n, 22, 19, 19], burn::tensor::Distribution::Bernoulli(0.3), &device)
            .slice_assign(s![.., 0..1, .., ..], Tensor::ones([n, 1, 19, 19], &device));
        let global = Tensor::<2>::random([n, 19], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
        let run = || async {
            let (p, v, s, o) = model.forward(bin.clone(), global.clone());
            let all = Tensor::cat(vec![p.flatten::<2>(1, 2), v, s, o.flatten::<2>(1, 3)], 1);
            read_all(all).await
        };
        let t = now_ms();
        run().await;
        let first = now_ms() - t;
        for _ in 0..3 {
            run().await;
        }
        let (mut iters, t) = (0u32, now_ms());
        while iters < 4 || now_ms() - t < seconds * 1e3 {
            run().await;
            iters += 1;
        }
        let per = (now_ms() - t) / iters as f64;
        out += &format!(
            "{backend} batch {n:>3}: first {first:>8.0} ms  {per:>8.2} ms/batch  {:>8.1} evals/s\n",
            n as f64 * 1e3 / per
        );
    }
    Ok(out)
}

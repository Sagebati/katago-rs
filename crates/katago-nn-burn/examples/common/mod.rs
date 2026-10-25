use burn::prelude::*;

pub fn device(name: &str) -> Device {
    match name {
        #[cfg(feature = "flex")]
        "flex" => burn::backend::FlexDevice::default().into(),
        // `wgpu-fixed-kernels` alone is the browser build's setup (no autotune), `wgpu` adds autotune.
        #[cfg(any(feature = "wgpu", feature = "wgpu-fixed-kernels"))]
        "wgpu" => {
            wino_switches();
            burn::backend::WgpuDevice::default().into()
        }
        #[cfg(feature = "rocm")]
        "rocm" => {
            wino_switches();
            burn::backend::RocmDevice::default().into()
        }
        other => panic!("backend {other} not compiled in"),
    }
}

/// Experiment switches for wino.rs (examples only: the library reads no environment).
#[cfg(any(feature = "wgpu", feature = "wgpu-fixed-kernels"))]
fn wino_switches() {
    if std::env::var_os("WINO_HALF").is_some() {
        katago_nn_burn::wino::set_half(true);
    }
    if std::env::var_os("WINO_GEMM_OUT_F32").is_some() {
        katago_nn_burn::wino::set_gemm_out_f32(true);
    }
    if std::env::var_os("WINO_SCALAR_GEMM").is_some() {
        katago_nn_burn::wino::set_vector_gemm(false);
    }
}

/// Runs `$body` with `$net` bound to the network type named by `$name` (b10 | b18 | human).
#[macro_export]
macro_rules! with_net {
    ($name:expr, $net:ident => $body:expr) => {
        match $name {
            #[cfg(feature = "net-b10c128")]
            "b10" => {
                type $net = katago_nn_burn::b10c128::Model;
                const BPK: &str = katago_nn_burn::bpk_path!("g170e-b10c128");
                $body(BPK)
            }
            #[cfg(feature = "net-b18c384nbt")]
            "b18" => {
                type $net = katago_nn_burn::b18c384nbt::Model;
                const BPK: &str = katago_nn_burn::bpk_path!("kata1-b18c384nbt");
                $body(BPK)
            }
            #[cfg(feature = "net-b18c384nbt-humanv0")]
            "human" => {
                type $net = katago_nn_burn::b18c384nbt_humanv0::Model;
                const BPK: &str = katago_nn_burn::bpk_path!("b18c384nbt-humanv0");
                $body(BPK)
            }
            other => panic!("network {other} not compiled in (b10 | b18 | human)"),
        }
    };
}

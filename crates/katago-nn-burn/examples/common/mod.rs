use burn::prelude::*;

/// The device of a native backend (`flex`, `wgpu`, `rocm`); panics when it is not compiled in. `wgpu` needs
/// `wgpu-fixed` (no autotune, as the browser) or `wgpu` (with burn's autotune).
pub fn device(name: &str) -> Device {
    katago_nn_burn::device(name).unwrap_or_else(|| panic!("backend {name} not compiled in (flex | wgpu | rocm)"))
}

/// Takes a precision argument (`f16` or `f32`) out of `args`: f16 runs the 3x3 convs in f16 on a GPU backend, as
/// `useFP16 = true` does in the engine. f32 when absent.
#[allow(dead_code)]
pub fn take_precision(args: &mut Vec<String>) -> &'static str {
    let precision = match args.iter().position(|a| a == "f16" || a == "f32") {
        Some(i) => args.remove(i),
        None => "f32".to_string(),
    };
    #[cfg(feature = "gpu-kernels")]
    katago_nn_burn::wino::set_half(precision == "f16");
    #[cfg(not(feature = "gpu-kernels"))]
    assert_eq!(precision, "f32", "f16 needs a GPU backend (wgpu-fixed or rocm)");
    if precision == "f16" { "f16" } else { "f32" }
}

/// Runs `$body` with `$net` bound to the network type named by `$name` (b6 | b10 | b18 | human) and its `.bpk`.
#[macro_export]
macro_rules! with_net {
    ($name:expr, $net:ident => $body:expr) => {
        match $name {
            #[cfg(feature = "net-b6c96")]
            "b6" => {
                type $net = katago_nn_burn::b6c96::Model;
                $body(katago_nn_burn::b6c96::BPK_PATH)
            }
            #[cfg(feature = "net-b10c128")]
            "b10" => {
                type $net = katago_nn_burn::b10c128::Model;
                $body(katago_nn_burn::b10c128::BPK_PATH)
            }
            #[cfg(feature = "net-b18c384nbt")]
            "b18" => {
                type $net = katago_nn_burn::b18c384nbt::Model;
                $body(katago_nn_burn::b18c384nbt::BPK_PATH)
            }
            #[cfg(feature = "net-b18c384nbt-humanv0")]
            "human" => {
                type $net = katago_nn_burn::b18c384nbt_humanv0::Model;
                $body(katago_nn_burn::b18c384nbt_humanv0::BPK_PATH)
            }
            other => panic!("network {other} not compiled in (b6 | b10 | b18 | human)"),
        }
    };
}

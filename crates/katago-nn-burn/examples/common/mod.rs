use burn::prelude::*;

pub fn device(name: &str) -> Device {
    match name {
        #[cfg(feature = "flex")]
        "flex" => burn::backend::FlexDevice::default().into(),
        #[cfg(feature = "wgpu")]
        "wgpu" => burn::backend::WgpuDevice::default().into(),
        other => panic!("backend {other} not compiled in"),
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

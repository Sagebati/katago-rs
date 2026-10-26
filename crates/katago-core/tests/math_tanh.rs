//! port::math::{tanh, atanh} (CORE-MATH's correctly rounded cr_tanh / cr_atanh, math/tanh.rs and
//! math/atanh.rs) against glibc 2.44, which implements both with the same code: replays
//! fixtures/upstream-v1.18.2/unit/oracle/math_tanh.txt, made by math_tanh.cpp next to it, line by
//! line and bit-exact. Runs natively and on wasm32 (`just test-wasm`), where the std
//! `f64::tanh` / `f64::atanh` are not glibc's. Natively it also compares directly with the host
//! libm on freshly generated inputs.

#[macro_use]
mod common;

use common::Cases;
use common::fixture;
use katago_core::port::math;

/// NaN results are canonicalised in the fixture (payloads differ between x86 and wasm).
fn bits(d: f64) -> u64 {
    if d.is_nan() {
        0x7FF8000000000000
    } else {
        d.to_bits()
    }
}
fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("hex in math_tanh.txt")
}

/// Replays the lines starting with `tag` ("T " or "A "): `<tag> <x bits> <result bits>`.
fn replay(tag: &str, name: &'static str, f: fn(f64) -> f64) {
    let text = fixture("oracle", "math_tanh.txt");
    let mut cases = Cases::new(name);
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0usize;
    for line in text.lines().filter(|l| l.starts_with(tag)) {
        let t: Vec<&str> = line.split(' ').collect();
        assert_eq!(t.len(), 3, "{}", line);
        let x = f64::from_bits(hex(t[1]));
        let got = bits(f(x));
        cases.ran();
        n += 1;
        if got != hex(t[2]) {
            bad.push(format!("{} -> rust {:016X}", line, got));
        }
    }
    assert!(n >= 1, "no {} lines in math_tanh.txt", tag);
    assert!(
        bad.is_empty(),
        "{} of {} cases differ from glibc:\n{}",
        bad.len(),
        n,
        bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
}

wasm_test! {
fn tanh_vs_glibc_fixture() {
    replay("T ", "tanh_vs_glibc_fixture", math::tanh);
}
}

wasm_test! {
fn atanh_vs_glibc_fixture() {
    replay("A ", "atanh_vs_glibc_fixture", math::atanh);
}
}

wasm_test! {
/// `tanh(atanh(w) + s*o)`, the form the caller (searchmirror.cpp) uses, on a grid: the fixture
/// only has random draws of it; this checks the C-visible invariants that follow from correct
/// rounding (odd symmetry, |result| < 1 or = 1 only when saturated, exact fixed points) on every
/// target.
fn tanh_atanh_symmetry_and_limits() {
    let mut cases = Cases::new("tanh_atanh_symmetry_and_limits");
    let mut z = 12345u64;
    for _ in 0..20000 {
        z = z
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let w = ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        let a = math::atanh(w);
        assert_eq!(
            math::atanh(-w).to_bits(),
            (-a).to_bits(),
            "atanh odd at {:e}",
            w
        );
        let t = math::tanh(a + 0.3);
        assert_eq!(
            math::tanh(-(a + 0.3)).to_bits(),
            (-t).to_bits(),
            "tanh odd at {:e}",
            a + 0.3
        );
        assert!(t.abs() <= 1.0, "tanh out of range at {:e}", a);
        cases.ran();
    }
    assert_eq!(math::tanh(0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(math::tanh(-0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(math::atanh(-0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(math::tanh(f64::INFINITY), 1.0);
    assert_eq!(math::tanh(f64::NEG_INFINITY), -1.0);
    assert_eq!(math::atanh(1.0), f64::INFINITY);
    assert_eq!(math::atanh(-1.0), f64::NEG_INFINITY);
    assert!(
        math::atanh(1.5).is_nan() && math::atanh(-2.0).is_nan() && math::atanh(f64::NAN).is_nan()
    );
    assert!(math::tanh(f64::NAN).is_nan());
}
}

/// Direct comparison with the host libm (glibc 2.44 on the fixture machine): fresh pseudo-random
/// inputs over the whole exponent range plus the ranges the caller uses. The 10^8-input runs
/// were done outside the test suite (see port/math.rs).
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn tanh_atanh_vs_host_glibc() {
    unsafe extern "C" {
        fn tanh(x: f64) -> f64;
        fn atanh(x: f64) -> f64;
    }
    let mut z = 99u64;
    let mut sm = move || {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        let mut v = z;
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
        v ^ (v >> 31)
    };
    let u = |b: u64| (b >> 11) as f64 / (1u64 << 53) as f64;
    let n: u64 = 1 << 18;
    let (mut bad_t, mut bad_a) = (0, 0);
    for i in 0..n {
        let (xt, xa) = match i % 4 {
            0 => (f64::from_bits(sm()), f64::from_bits(sm())),
            1 => (u(sm()) * 50.0 - 25.0, u(sm()) * 2.0 - 1.0),
            2 => ((u(sm()) - 0.5) * 0.5, (u(sm()) - 0.5) * 0.5),
            _ => (u(sm()) * 8.0 - 4.0, 1.0 - u(sm()) * 1e-3),
        };
        // SAFETY: plain libm calls.
        let (gt, ga) = unsafe { (tanh(xt), atanh(xa)) };
        bad_t += (bits(gt) != bits(math::tanh(xt))) as i32;
        bad_a += (bits(ga) != bits(math::atanh(xa))) as i32;
    }
    assert_eq!(
        (bad_t, bad_a),
        (0, 0),
        "port::math::{{tanh, atanh}} differ from host glibc"
    );
    eprintln!("tanh_atanh_vs_host_glibc: {} inputs each", n);
}

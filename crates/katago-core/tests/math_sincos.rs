//! port::math::{sin, cos} (CORE-MATH's correctly rounded cr_sincos, math/sincos.rs) against the
//! C cr_sincos, itself checked against MPFR: replays
//! fixtures/upstream-v1.16.5/unit/oracle/math_sincos.txt, made by math_sincos.cpp next to it,
//! line by line and bit-exact. Runs natively and on wasm32 (`just test-wasm`), where the std
//! `f64::sin` / `f64::cos` are not glibc's. Natively it also checks that the float casts of
//! sgfmetadata.cpp fillMetadataRow's `(float)cos(x)` / `(float)sin(x)` agree with the host glibc
//! (whose double results are not correctly rounded, so they are only checked to be within 1 ulp).

// upstream's twopi literal (sgfmetadata.cpp) kept verbatim
#![allow(clippy::approx_constant, clippy::excessive_precision)]

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
    u64::from_str_radix(s, 16).expect("hex in math_sincos.txt")
}

wasm_test! {
/// Replays the `S <x bits> <sin(x) bits> <cos(x) bits>` lines.
fn sincos_vs_core_math_fixture() {
    let text = fixture("oracle", "math_sincos.txt");
    let mut cases = Cases::new("sincos_vs_core_math_fixture");
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0usize;
    for line in text.lines().filter(|l| l.starts_with("S ")) {
        let t: Vec<&str> = line.split(' ').collect();
        assert_eq!(t.len(), 4, "{}", line);
        let x = f64::from_bits(hex(t[1]));
        let (s, c) = (bits(math::sin(x)), bits(math::cos(x)));
        let (s2, c2) = math::cr_sincos(x);
        assert_eq!(
            (bits(s2), bits(c2)),
            (s, c),
            "cr_sincos vs sin/cos at {}",
            line
        );
        cases.ran();
        n += 1;
        if s != hex(t[2]) || c != hex(t[3]) {
            bad.push(format!("{} -> rust {:016X} {:016X}", line, s, c));
        }
    }
    assert!(n >= 1, "no S lines in math_sincos.txt");
    assert!(
        bad.is_empty(),
        "{} of {} cases differ from CORE-MATH:\n{}",
        bad.len(),
        n,
        bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
}
}

wasm_test! {
/// Symmetries and special values that follow from correct rounding, on every target.
fn sincos_symmetry_and_specials() {
    let mut cases = Cases::new("sincos_symmetry_and_specials");
    let mut z = 12345u64;
    for _ in 0..20000 {
        z = z
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = f64::from_bits(z >> 2); // positive, exponents up to ~2^511
        let (s, c) = math::cr_sincos(x);
        let (sn, cn) = math::cr_sincos(-x);
        assert_eq!(bits(sn), bits(-s), "sin odd at {:e}", x);
        assert_eq!(bits(cn), bits(c), "cos even at {:e}", x);
        assert!(s.abs() <= 1.0 && c.abs() <= 1.0, "out of range at {:e}", x);
        cases.ran();
    }
    assert_eq!(math::sin(0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(math::sin(-0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(math::cos(0.0), 1.0);
    assert_eq!(math::cos(-0.0), 1.0);
    let tiny = f64::from_bits(1);
    assert_eq!(math::sin(tiny).to_bits(), tiny.to_bits());
    for x in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(math::sin(x).is_nan() && math::cos(x).is_nan());
    }
}
}

/// sgfmetadata.cpp fillMetadataRow computes `(float)cos(x)` / `(float)sin(x)` with
/// x = (daysDifference / period) * twopi. glibc 2.44's sin/cos are not correctly rounded, but
/// after the cast to float they agree with the correctly rounded value on these inputs: checked
/// here against the host libm (glibc on the fixture machine) for every d in [-80000, 80000] and
/// two of the 32 periods each (the full 5.1M grid also agreed, outside the suite).
#[cfg(all(not(target_arch = "wasm32"), target_os = "linux", target_env = "gnu"))]
#[test]
fn fill_metadata_row_float_cast_vs_host_glibc() {
    unsafe extern "C" {
        fn sin(x: f64) -> f64;
        fn cos(x: f64) -> f64;
    }
    let mut periods = [0.0f64; 32];
    let mut period = 7.0;
    let factor = math::powf(80000.0, 1.0 / (32 - 1) as f64);
    for p in periods.iter_mut() {
        *p = period;
        period *= factor;
    }
    let twopi = 6.283185307179586476925;
    let mut cases = Cases::new("fill_metadata_row_float_cast_vs_host_glibc");
    let (mut bad, mut double_diffs) = (Vec::new(), 0usize);
    for d in -80000i32..=80000 {
        let a = (d as u32 & 31) as usize;
        for i in [a, 31 - a] {
            let days_difference = d as f64;
            let num_revolutions = days_difference / periods[i];
            let x = num_revolutions * twopi;
            // SAFETY: plain libm calls.
            let (gs, gc) = unsafe { (sin(x), cos(x)) };
            let (s, c) = (math::sin(x), math::cos(x));
            for (ours, glibc) in [(s, gs), (c, gc)] {
                let ulps = (ours.to_bits() as i64).wrapping_sub(glibc.to_bits() as i64);
                assert!(
                    ulps.abs() <= 1,
                    "more than 1 ulp from glibc at d={} i={}",
                    d,
                    i
                );
                double_diffs += (ulps != 0) as usize;
            }
            if (s as f32).to_bits() != (gs as f32).to_bits()
                || (c as f32).to_bits() != (gc as f32).to_bits()
            {
                bad.push(format!("d={} i={} x={:e}", d, i, x));
            }
            cases.ran();
        }
    }
    eprintln!(
        "fill_metadata_row_float_cast_vs_host_glibc: {} double results differ from glibc by 1 ulp \
         (expected: glibc is not correctly rounded), {} after the float cast",
        double_diffs,
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} float casts differ from glibc:\n{}",
        bad.len(),
        bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
}

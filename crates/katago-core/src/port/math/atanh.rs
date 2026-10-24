//! C `atanh` (double): CORE-MATH's correctly rounded `cr_atanh`, which is what glibc 2.44 uses
//! for `atanh`, so `port::math::atanh` is bit-identical to glibc on every target.
//!
//! Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/,
//! https://gitlab.inria.fr/core-math/core-math), src/binary64/atanh/atanh.c at commit
//! 0ce0cbc15935e24c656280e5a46c34e89f301b7a (2026-10-20). Correctly rounded.
//!
//! atanh.c: Copyright (c) 2023-2025 Alexei Sibidanov <sibid@uvic.ca>.
//! This file is part of the CORE-MATH project (https://core-math.gitlabpages.inria.fr/).
//! Under the MIT license:
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//!
//! Port notes (same conventions as math/atan.rs):
//! - `__builtin_fma(a, b, c)` -> `a.mul_add(b, c)` (a correctly rounded fma on every target);
//!   every other `a*b+c` stays two roundings, as in a `-ffp-contract=off` C build (Rust never
//!   contracts). The result is correctly rounded either way, so it does not depend on the CPU
//!   or on the C compiler's contraction choices.
//! - Hex float literals keep their C spelling through `hf!` (math/lgamma.rs; const-evaluated,
//!   exact). The `b64u64_u` unions are `to_bits`/`from_bits`; `__builtin_copysign`/`fabs` are
//!   `f64::copysign`/`f64::abs` (exact bit operations). Unsigned C arithmetic that can wrap is
//!   `wrapping_*`.
//! - Out-parameters `double *l` -> `&mut f64`, in C argument order. The C function-local
//!   `static const` tables are module-level statics.
//! - `CORE_MATH_SUPPORT_ERRNO` (errno = ERANGE on underflow) and the floating-point exception
//!   flags are not ported: C++ callers never read them; the returned values are unchanged.

#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::too_many_arguments
)]

use super::lgamma::hexf;

const MANT_MASK: u64 = !0u64 >> 12; // ~(uint64_t)0 >> 12

#[inline(always)]
fn fasttwosum(x: f64, y: f64, e: &mut f64) -> f64 {
    let s = x + y;
    let z = s - x;
    *e = y - z;
    s
}

#[inline(always)]
fn fasttwosub(x: f64, y: f64, e: &mut f64) -> f64 {
    let s = x - y;
    let z = x - s;
    *e = z - y;
    s
}

#[inline(always)]
fn adddd(xh: f64, xl: f64, ch: f64, cl: f64, l: &mut f64) -> f64 {
    let s = xh + ch;
    let d = s - xh;
    *l = ((ch - d) + (xh + (d - s))) + (xl + cl);
    s
}

#[inline(always)]
fn muldd_acc(xh: f64, xl: f64, ch: f64, cl: f64, l: &mut f64) -> f64 {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let mut ahhl = ch.mul_add(xh, -ahhh);
    ahhl += alhh + ahlh;
    fasttwosum(ahhh, ahhl, l)
}

/// Same as `muldd_acc`, but without the `fasttwosum` normalization.
#[inline(always)]
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64, l: &mut f64) -> f64 {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = ch.mul_add(xh, -ahhh);
    *l = ahhl + (alhh + ahlh);
    ahhh
}

#[inline(always)]
fn mulddd(xh: f64, xl: f64, ch: f64, l: &mut f64) -> f64 {
    let hh = xh * ch;
    *l = ch.mul_add(xh, -hh) + xl * ch;
    hh
}

#[inline(always)]
fn polydd(xh: f64, xl: f64, n: usize, c: &[[f64; 2]], l: &mut f64) -> f64 {
    let mut i = n - 1;
    let mut cl = 0.0;
    let mut ch = fasttwosum(c[i][0], *l, &mut cl);
    cl += c[i][1];
    while i > 0 {
        i -= 1;
        ch = muldd_acc(xh, xl, ch, cl, &mut cl);
        let mut tl = 0.0;
        let th = fasttwosum(c[i][0], ch, &mut tl);
        ch = th;
        cl += tl + c[i][1];
    }
    *l = cl;
    ch
}

#[rustfmt::skip]
static ZERO_CH: [[f64; 2]; 13] = [
  [hf!("0x1.5555555555555p-2"), hf!("0x1.5555555555555p-56")],
  [hf!("0x1.999999999999ap-3"), hf!("-0x1.999999999611cp-57")],
  [hf!("0x1.2492492492492p-3"), hf!("0x1.2492490f76b25p-57")],
  [hf!("0x1.c71c71c71c71cp-4"), hf!("0x1.c71cd5c38a112p-58")],
  [hf!("0x1.745d1745d1746p-4"), hf!("-0x1.7556c4165f4cap-59")],
  [hf!("0x1.3b13b13b13b14p-4"), hf!("-0x1.b893c3b36052ep-59")],
  [hf!("0x1.1111111111105p-4"), hf!("0x1.4e1afd723ed1fp-59")],
  [hf!("0x1.e1e1e1e1e2678p-5"), hf!("-0x1.f86ea96fb1435p-59")],
  [hf!("0x1.af286bc9f90ccp-5"), hf!("0x1.1e51a6e54fde9p-60")],
  [hf!("0x1.8618618c779b6p-5"), hf!("-0x1.ab913de95c3bfp-61")],
  [hf!("0x1.642c84aa383ebp-5"), hf!("0x1.632e747641b12p-59")],
  [hf!("0x1.47ae2d205013cp-5"), hf!("-0x1.0c9617e7bcff2p-60")],
  [hf!("0x1.2f664d60473f9p-5"), hf!("0x1.3adb3e2b7f35ep-61")],
];

#[rustfmt::skip]
static ZERO_CL: [f64; 5] = [
  hf!("0x1.1a9a91fd692afp-5"), hf!("0x1.06dfbb35e7f44p-5"), hf!("0x1.037bed4d7588fp-5"),
  hf!("0x1.5aca6d6d720d6p-6"), hf!("0x1.99ea5700d53a5p-5"),
];

#[rustfmt::skip]
static SMALL_C: [f64; 9] = [
  hf!("0x1.999999999999ap-3"), hf!("0x1.2492492492244p-3"), hf!("0x1.c71c71c79715fp-4"),
  hf!("0x1.745d16f777723p-4"), hf!("0x1.3b13ca4174634p-4"), hf!("0x1.110c9724989bdp-4"),
  hf!("0x1.e2d17608a5b2ep-5"), hf!("0x1.a0b56308cba0bp-5"), hf!("0x1.fb6341208ad2ep-5"),
];

#[rustfmt::skip]
static B: [(u16, i16); 32] = [
  (301, 27565), (7189, 24786), (13383, 22167), (18923, 19696), (23845, 17361), (28184, 15150),
  (31969, 13054), (35231, 11064), (37996, 9173), (40288, 7372), (42129, 5657), (43542, 4020),
  (44546, 2457), (45160, 962), (45399, -468), (45281, -1838), (44821, -3151), (44032, -4412),
  (42929, -5622), (41522, -6786), (39825, -7905), (37848, -8982), (35602, -10020), (33097, -11020),
  (30341, -11985), (27345, -12916), (24115, -13816), (20661, -14685), (16989, -15526), (13107, -16339),
  (9022, -17126), (4740, -17889),
];

#[rustfmt::skip]
static R1: [f64; 33] = [
  hf!("0x1p+0"), hf!("0x1.f5076p-1"), hf!("0x1.ea4bp-1"), hf!("0x1.dfc98p-1"), hf!("0x1.d5818p-1"),
  hf!("0x1.cb72p-1"), hf!("0x1.c199cp-1"), hf!("0x1.b7f76p-1"), hf!("0x1.ae8ap-1"), hf!("0x1.a5504p-1"),
  hf!("0x1.9c492p-1"), hf!("0x1.93738p-1"), hf!("0x1.8ace6p-1"), hf!("0x1.8258ap-1"), hf!("0x1.7a114p-1"),
  hf!("0x1.71f76p-1"), hf!("0x1.6a09ep-1"), hf!("0x1.6247ep-1"), hf!("0x1.5ab08p-1"), hf!("0x1.5342cp-1"),
  hf!("0x1.4bfdap-1"), hf!("0x1.44e08p-1"), hf!("0x1.3dea6p-1"), hf!("0x1.371a8p-1"), hf!("0x1.306fep-1"),
  hf!("0x1.29e9ep-1"), hf!("0x1.2387ap-1"), hf!("0x1.1d488p-1"), hf!("0x1.172b8p-1"), hf!("0x1.11302p-1"),
  hf!("0x1.0b558p-1"), hf!("0x1.059bp-1"), hf!("0x1p-1"),
];

#[rustfmt::skip]
static R2: [f64; 33] = [
  hf!("0x1p+0"), hf!("0x1.ffa74p-1"), hf!("0x1.ff4eap-1"), hf!("0x1.fef62p-1"), hf!("0x1.fe9dap-1"),
  hf!("0x1.fe452p-1"), hf!("0x1.fdeccp-1"), hf!("0x1.fd946p-1"), hf!("0x1.fd3c2p-1"), hf!("0x1.fce3ep-1"),
  hf!("0x1.fc8bcp-1"), hf!("0x1.fc33ap-1"), hf!("0x1.fbdbap-1"), hf!("0x1.fb83ap-1"), hf!("0x1.fb2bcp-1"),
  hf!("0x1.fad3ep-1"), hf!("0x1.fa7c2p-1"), hf!("0x1.fa246p-1"), hf!("0x1.f9ccap-1"), hf!("0x1.f975p-1"),
  hf!("0x1.f91d8p-1"), hf!("0x1.f8c6p-1"), hf!("0x1.f86e8p-1"), hf!("0x1.f8172p-1"), hf!("0x1.f7bfep-1"),
  hf!("0x1.f768ap-1"), hf!("0x1.f7116p-1"), hf!("0x1.f6ba4p-1"), hf!("0x1.f6632p-1"), hf!("0x1.f60c2p-1"),
  hf!("0x1.f5b52p-1"), hf!("0x1.f55e4p-1"), hf!("0x1.f5076p-1"),
];

#[rustfmt::skip]
static L1: [[f64; 2]; 33] = [
  [hf!("0x0p+0"), hf!("0x0p+0")],
  [hf!("-0x1.532c1269e2038p-27"), hf!("0x1.62e5p-7")],
  [hf!("0x1.ce42d81b54e84p-27"), hf!("0x1.62e3cp-6")],
  [hf!("-0x1.25826f815ec3dp-26"), hf!("0x1.0a2acp-5")],
  [hf!("0x1.0db1b1e7cee11p-26"), hf!("0x1.62e4ap-5")],
  [hf!("-0x1.1f3a8c6c95003p-26"), hf!("0x1.bb9dcp-5")],
  [hf!("-0x1.774cd4fb8c30dp-26"), hf!("0x1.0a2b2p-4")],
  [hf!("0x1.452e56c030a0ap-29"), hf!("0x1.3687fp-4")],
  [hf!("0x1.6b63c4966a79ap-28"), hf!("0x1.62e41p-4")],
  [hf!("-0x1.b20a21ccb525ep-28"), hf!("0x1.8f40ap-4")],
  [hf!("0x1.4006cfb3d8f85p-26"), hf!("0x1.bb9d1p-4")],
  [hf!("-0x1.cdb026b310c41p-26"), hf!("0x1.e7f9bp-4")],
  [hf!("-0x1.69124fdc0f16dp-26"), hf!("0x1.0a2b08p-3")],
  [hf!("-0x1.084656cdc2727p-26"), hf!("0x1.205958p-3")],
  [hf!("-0x1.376fa8b0357fdp-26"), hf!("0x1.3687cp-3")],
  [hf!("0x1.e56ae55a47b4ap-28"), hf!("0x1.4cb5e8p-3")],
  [hf!("0x1.070ff8834eeb4p-26"), hf!("0x1.62e44p-3")],
  [hf!("0x1.623516109f4fep-26"), hf!("0x1.79129p-3")],
  [hf!("-0x1.ec656b95fbdacp-29"), hf!("0x1.8f40bp-3")],
  [hf!("0x1.f0ca2e729f51p-28"), hf!("0x1.a56ed8p-3")],
  [hf!("-0x1.7d260a858354ap-26"), hf!("0x1.bb9d68p-3")],
  [hf!("0x1.e7279075503d3p-27"), hf!("0x1.d1cb9p-3")],
  [hf!("0x1.39e1a0a503873p-27"), hf!("0x1.e7f9dp-3")],
  [hf!("0x1.cd86d7b87c3d6p-26"), hf!("0x1.fe27d8p-3")],
  [hf!("0x1.060ab88de341ep-26"), hf!("0x1.0a2b24p-2")],
  [hf!("0x1.20a860d3f939p-28"), hf!("0x1.154244p-2")],
  [hf!("-0x1.dacee95fc2f1p-27"), hf!("0x1.205974p-2")],
  [hf!("0x1.45de3a86e0acap-26"), hf!("0x1.2b707p-2")],
  [hf!("0x1.c164cbfb991afp-27"), hf!("0x1.3687bp-2")],
  [hf!("0x1.d3f66b24225efp-26"), hf!("0x1.419ec4p-2")],
  [hf!("0x1.fc023efa144bap-26"), hf!("0x1.4cb5f8p-2")],
  [hf!("0x1.086a8af6f26cp-28"), hf!("0x1.57cd28p-2")],
  [hf!("-0x1.05c610ca86c39p-30"), hf!("0x1.62e43p-2")],
];

#[rustfmt::skip]
static L2: [[f64; 2]; 33] = [
  [hf!("0x0p+0"), hf!("0x0p+0")],
  [hf!("-0x1.37e152a129e4ep-28"), hf!("0x1.632p-12")],
  [hf!("-0x1.3f6c916b8be9cp-26"), hf!("0x1.63p-11")],
  [hf!("0x1.20505936739d5p-26"), hf!("0x1.0a24p-10")],
  [hf!("-0x1.23e2e8cb541bap-26"), hf!("0x1.62dcp-10")],
  [hf!("-0x1.acb7983ac4f5ep-32"), hf!("0x1.bbap-10")],
  [hf!("0x1.6f7c7689c63aep-28"), hf!("0x1.0a2ap-9")],
  [hf!("0x1.f5ca695b4c58bp-30"), hf!("0x1.368cp-9")],
  [hf!("-0x1.c6c18bd953226p-27"), hf!("0x1.62e6p-9")],
  [hf!("0x1.7a516c34846bdp-26"), hf!("0x1.8f46p-9")],
  [hf!("-0x1.f3b83dd8b853p-27"), hf!("0x1.bbap-9")],
  [hf!("-0x1.c3459046e4e57p-31"), hf!("0x1.e8p-9")],
  [hf!("0x1.b5c7e34cb79f6p-38"), hf!("0x1.0a2cp-8")],
  [hf!("-0x1.2487e9af9a692p-27"), hf!("0x1.205cp-8")],
  [hf!("0x1.f21bbc4ad79cep-26"), hf!("0x1.3687p-8")],
  [hf!("-0x1.550ffc857b731p-29"), hf!("0x1.4cb7p-8")],
  [hf!("0x1.87458ec1b7b34p-27"), hf!("0x1.62e2p-8")],
  [hf!("0x1.103d4fe83ee81p-26"), hf!("0x1.7911p-8")],
  [hf!("0x1.810483d3b398cp-27"), hf!("0x1.8f44p-8")],
  [hf!("-0x1.2085cb340608ep-27"), hf!("0x1.a573p-8")],
  [hf!("0x1.12698a119c42fp-26"), hf!("0x1.bb9dp-8")],
  [hf!("-0x1.edb8c172b4c33p-26"), hf!("0x1.d1ccp-8")],
  [hf!("-0x1.8b55b87a5e238p-26"), hf!("0x1.e7fep-8")],
  [hf!("0x1.be5e17763f78ap-26"), hf!("0x1.fe2bp-8")],
  [hf!("-0x1.c2d496790073ep-30"), hf!("0x1.0a2a8p-7")],
  [hf!("0x1.6542f523abeecp-26"), hf!("0x1.1541p-7")],
  [hf!("-0x1.b7fdbe5b193f8p-26"), hf!("0x1.205ap-7")],
  [hf!("0x1.fa4d42fe30c7cp-26"), hf!("0x1.2b7p-7")],
  [hf!("0x1.0d46ad04adc86p-26"), hf!("0x1.36888p-7")],
  [hf!("-0x1.1c22d02d17c4cp-26"), hf!("0x1.419fp-7")],
  [hf!("0x1.a7d1e330dcccep-30"), hf!("0x1.4cb7p-7")],
  [hf!("0x1.187025e656ba3p-31"), hf!("0x1.57cdp-7")],
  [hf!("-0x1.532c1269e2038p-27"), hf!("0x1.62e5p-7")],
];

#[rustfmt::skip]
static C: [f64; 5] = [
  hf!("-0x1p+0"), hf!("0x1.555555555553p+0"), hf!("-0x1.fffffffffffap+0"), hf!("0x1.99999e33a6366p+1"),
  hf!("-0x1.555559ef9525fp+2"),
];

#[rustfmt::skip]
static DB: [[f64; 3]; 2] = [
  [hf!("0x1.2dbb7b1c91363p-2"), hf!("0x1.36f33d51c264dp-2"), hf!("0x1p-56")],
  [hf!("0x1.c493dc899e4a5p-2"), hf!("0x1.e611aa58ab608p-2"), hf!("-0x1p-56")],
];

#[rustfmt::skip]
static RT1: [f64; 17] = [
  hf!("0x1p+0"), hf!("0x1.ea4afap-1"), hf!("0x1.d5818ep-1"), hf!("0x1.c199bep-1"), hf!("0x1.ae89f98p-1"),
  hf!("0x1.9c4918p-1"), hf!("0x1.8ace54p-1"), hf!("0x1.7a1147p-1"), hf!("0x1.6a09e68p-1"),
  hf!("0x1.5ab07ep-1"), hf!("0x1.4bfdad8p-1"), hf!("0x1.3dea65p-1"), hf!("0x1.306fe08p-1"),
  hf!("0x1.2387a7p-1"), hf!("0x1.172b84p-1"), hf!("0x1.0b5587p-1"), hf!("0x1p-1"),
];

#[rustfmt::skip]
static RT2: [f64; 16] = [
  hf!("0x1p+0"), hf!("0x1.fe9d968p-1"), hf!("0x1.fd3c228p-1"), hf!("0x1.fbdba38p-1"), hf!("0x1.fa7c18p-1"),
  hf!("0x1.f91d8p-1"), hf!("0x1.f7bfdbp-1"), hf!("0x1.f663278p-1"), hf!("0x1.f507658p-1"),
  hf!("0x1.f3ac948p-1"), hf!("0x1.f252b38p-1"), hf!("0x1.f0f9c2p-1"), hf!("0x1.efa1bfp-1"),
  hf!("0x1.ee4aaap-1"), hf!("0x1.ecf483p-1"), hf!("0x1.eb9f488p-1"),
];

#[rustfmt::skip]
static RT3: [f64; 16] = [
  hf!("0x1p+0"), hf!("0x1.ffe9d2p-1"), hf!("0x1.ffd3a58p-1"), hf!("0x1.ffbd798p-1"), hf!("0x1.ffa74e8p-1"),
  hf!("0x1.ff91248p-1"), hf!("0x1.ff7afb8p-1"), hf!("0x1.ff64d38p-1"), hf!("0x1.ff4eac8p-1"),
  hf!("0x1.ff38868p-1"), hf!("0x1.ff22618p-1"), hf!("0x1.ff0c3dp-1"), hf!("0x1.fef61ap-1"),
  hf!("0x1.fedff78p-1"), hf!("0x1.fec9d68p-1"), hf!("0x1.feb3b6p-1"),
];

#[rustfmt::skip]
static RT4: [f64; 16] = [
  hf!("0x1p+0"), hf!("0x1.fffe9dp-1"), hf!("0x1.fffd3ap-1"), hf!("0x1.fffbd78p-1"), hf!("0x1.fffa748p-1"),
  hf!("0x1.fff9118p-1"), hf!("0x1.fff7ae8p-1"), hf!("0x1.fff64cp-1"), hf!("0x1.fff4e9p-1"),
  hf!("0x1.fff386p-1"), hf!("0x1.fff2238p-1"), hf!("0x1.fff0c08p-1"), hf!("0x1.ffef5d8p-1"),
  hf!("0x1.ffedfa8p-1"), hf!("0x1.ffec98p-1"), hf!("0x1.ffeb35p-1"),
];

#[rustfmt::skip]
static LL: [[[f64; 3]; 17]; 4] = [
  [
    [hf!("0x0p+0"), hf!("0x0p+0"), hf!("0x0p+0")],
    [hf!("0x1.62e432b24p-6"), hf!("-0x1.745af34bb54b8p-42"), hf!("-0x1.17e3ec05cde7p-97")],
    [hf!("0x1.62e42e4a8p-5"), hf!("0x1.111a4eadf312p-44"), hf!("0x1.cff3027abb119p-93")],
    [hf!("0x1.0a2b233f1p-4"), hf!("-0x1.88ac4ec78af8p-42"), hf!("0x1.4fa087ca75dfdp-93")],
    [hf!("0x1.62e43056cp-4"), hf!("0x1.6bd65e8b0b7p-46"), hf!("-0x1.b18e160362c24p-95")],
    [hf!("0x1.bb9d3cbd6p-4"), hf!("0x1.de14aa55ec2bp-42"), hf!("-0x1.c6ac3f1862a6bp-94")],
    [hf!("0x1.0a2b244dap-3"), hf!("0x1.94def487fea7p-42"), hf!("-0x1.dead1a4581acfp-94")],
    [hf!("0x1.3687aa9b78p-3"), hf!("0x1.9cec9a50db22p-43"), hf!("0x1.34a70684f8e0ep-93")],
    [hf!("0x1.62e42fabap-3"), hf!("-0x1.d69047a3aebp-44"), hf!("-0x1.4e061f79144e2p-95")],
    [hf!("0x1.8f40b56d28p-3"), hf!("0x1.de7d755fd2e2p-42"), hf!("0x1.bdc7ecf001489p-94")],
    [hf!("0x1.bb9d3b61fp-3"), hf!("0x1.c14f1445b12p-46"), hf!("0x1.a1d78cbdc5b58p-93")],
    [hf!("0x1.e7f9c11f08p-3"), hf!("-0x1.6e3e0000dae7p-43"), hf!("0x1.6a4559fadde98p-94")],
    [hf!("0x1.0a2b242ec4p-2"), hf!("0x1.bb7cf852a5fe8p-42"), hf!("0x1.a6aef11ee43bdp-93")],
    [hf!("0x1.205966c764p-2"), hf!("0x1.ad3a5f214294p-45"), hf!("0x1.5cc344fa10652p-93")],
    [hf!("0x1.3687a98aacp-2"), hf!("0x1.1623671842fp-45"), hf!("-0x1.0b428fe1f9e43p-94")],
    [hf!("0x1.4cb5ec93f4p-2"), hf!("0x1.3d50980ea513p-42"), hf!("0x1.67f0ea083b1c4p-93")],
    [hf!("0x1.62e42fefa4p-2"), hf!("-0x1.8432a1b0e264p-44"), hf!("0x1.803f2f6af40f3p-93")],
  ],
  [
    [hf!("0x0p+0"), hf!("0x0p+0"), hf!("0x0p+0")],
    [hf!("0x1.62e462b4p-10"), hf!("0x1.061d003b97318p-42"), hf!("0x1.d7faee66a2e1ep-93")],
    [hf!("0x1.62e44c92p-9"), hf!("0x1.95a7bff5e239p-42"), hf!("-0x1.f7e788a87135p-95")],
    [hf!("0x1.0a2b1e33p-8"), hf!("0x1.2a3a1a65aa3ap-43"), hf!("-0x1.54599c9605442p-93")],
    [hf!("0x1.62e4367cp-8"), hf!("-0x1.4a995b6d9ddcp-45"), hf!("-0x1.56bb79b254f33p-100")],
    [hf!("0x1.bb9d449ap-8"), hf!("0x1.8a119c42e9bcp-42"), hf!("-0x1.8ecf7d8d661f1p-93")],
    [hf!("0x1.0a2b1f19p-7"), hf!("0x1.8863771bd10a8p-42"), hf!("0x1.e9731de7f0155p-94")],
    [hf!("0x1.3687ad11p-7"), hf!("0x1.e026a347ca1c8p-42"), hf!("0x1.fadc62522444dp-97")],
    [hf!("0x1.62e436f28p-7"), hf!("0x1.25b84f71b70b8p-42"), hf!("-0x1.fcb3f98612d27p-96")],
    [hf!("0x1.8f40b7b38p-7"), hf!("-0x1.62a0a4fd4758p-43"), hf!("0x1.3cb3c35d9f6a1p-93")],
    [hf!("0x1.bb9d3abbp-7"), hf!("-0x1.0ec48f94d786p-42"), hf!("-0x1.6b47d410e4cc7p-93")],
    [hf!("0x1.e7f9bb23p-7"), hf!("0x1.e4415cbc97ap-43"), hf!("-0x1.3729fdb677231p-93")],
    [hf!("0x1.0a2b22478p-6"), hf!("-0x1.cb73f4505b03p-42"), hf!("-0x1.1b3b3a3bc370ap-93")],
    [hf!("0x1.2059691e8p-6"), hf!("-0x1.abcc3412f264p-43"), hf!("-0x1.fe6e998e48673p-95")],
    [hf!("0x1.3687a768p-6"), hf!("-0x1.43901e5c97a9p-42"), hf!("0x1.b54cdd52a5d88p-96")],
    [hf!("0x1.4cb5eb5d8p-6"), hf!("-0x1.8f106f00f13b8p-42"), hf!("-0x1.8f793f5fce148p-93")],
    [hf!("0x1.62e432b24p-6"), hf!("-0x1.745af34bb54b8p-42"), hf!("-0x1.17e3ec05cde7p-97")],
  ],
  [
    [hf!("0x0p+0"), hf!("0x0p+0"), hf!("0x0p+0")],
    [hf!("0x1.62e7bp-14"), hf!("-0x1.868625640a68p-44"), hf!("-0x1.34bf0db910f65p-93")],
    [hf!("0x1.62e35f6p-13"), hf!("-0x1.2ee3d96b696ap-43"), hf!("0x1.a2948cd558655p-94")],
    [hf!("0x1.0a2b4b2p-12"), hf!("0x1.53edbcf1165p-47"), hf!("-0x1.cfc26ccf6d0e4p-97")],
    [hf!("0x1.62e4be1p-12"), hf!("0x1.783e334614p-52"), hf!("-0x1.04b96da30e63ap-93")],
    [hf!("0x1.bb9e085p-12"), hf!("-0x1.60785f20acb2p-43"), hf!("-0x1.f33369bf7dff1p-96")],
    [hf!("0x1.0a2b94dp-11"), hf!("0x1.fd4b3a273353p-42"), hf!("-0x1.685a35575eff1p-96")],
    [hf!("0x1.368810f8p-11"), hf!("0x1.7ded26dc813p-47"), hf!("-0x1.4c4d1abca79bfp-96")],
    [hf!("0x1.62e47878p-11"), hf!("0x1.7d2bee9a1f63p-42"), hf!("0x1.860233b7ad13p-93")],
    [hf!("0x1.8f40cb48p-11"), hf!("-0x1.af034eaf471cp-42"), hf!("0x1.ae748822d57b7p-94")],
    [hf!("0x1.bb9d094p-11"), hf!("-0x1.7a223013a20fp-42"), hf!("-0x1.1e499087075b6p-93")],
    [hf!("0x1.e7fa32c8p-11"), hf!("-0x1.b2e67b1b59bdp-43"), hf!("-0x1.54a41eda30fa6p-93")],
    [hf!("0x1.0a2b237p-10"), hf!("-0x1.7ad97ff4ac7ap-44"), hf!("0x1.f932da91371ddp-93")],
    [hf!("0x1.2059a338p-10"), hf!("-0x1.96422d90df4p-44"), hf!("-0x1.90800fbbf2ed3p-94")],
    [hf!("0x1.36879824p-10"), hf!("0x1.0f9054001812p-44"), hf!("0x1.9567e01e48f9ap-93")],
    [hf!("0x1.4cb602cp-10"), hf!("-0x1.0d709a5ec0b5p-43"), hf!("0x1.253dfd44635d2p-94")],
    [hf!("0x1.62e462b4p-10"), hf!("0x1.061d003b97318p-42"), hf!("0x1.d7faee66a2e1ep-93")],
  ],
  [
    [hf!("0x0p+0"), hf!("0x0p+0"), hf!("0x0p+0")],
    [hf!("0x1.63007cp-18"), hf!("-0x1.db0e38e5aaaap-43"), hf!("0x1.259a7b94815b9p-93")],
    [hf!("0x1.6300f6p-17"), hf!("0x1.2b1c75580438p-44"), hf!("0x1.78cabba01e3e4p-93")],
    [hf!("0x1.0a2115p-16"), hf!("-0x1.5ff223730759p-42"), hf!("0x1.8074feacfe49dp-95")],
    [hf!("0x1.62e1ecp-16"), hf!("-0x1.85d6f6487ce4p-45"), hf!("0x1.05485074b9276p-93")],
    [hf!("0x1.bba301p-16"), hf!("-0x1.af5d58a7c921p-43"), hf!("-0x1.30a8c0fd2ff5fp-93")],
    [hf!("0x1.0a32298p-15"), hf!("0x1.590faa0883bdp-43"), hf!("0x1.95e9bda999947p-93")],
    [hf!("0x1.3682f1p-15"), hf!("0x1.f0224376efaf8p-42"), hf!("-0x1.5843c0db50d1p-93")],
    [hf!("0x1.62e3d8p-15"), hf!("-0x1.142c13daed4ap-43"), hf!("0x1.c68a61183ce87p-93")],
    [hf!("0x1.8f44dd8p-15"), hf!("-0x1.aa489f399931p-43"), hf!("0x1.11c5c376854eap-94")],
    [hf!("0x1.bb9601p-15"), hf!("0x1.9904d8b6a3638p-42"), hf!("0x1.8c89554493c8fp-93")],
    [hf!("0x1.e7f744p-15"), hf!("0x1.5785ddbe7cba8p-42"), hf!("0x1.e7ff3cde7d70cp-94")],
    [hf!("0x1.0a2c53p-14"), hf!("-0x1.6d9e8780d0d5p-43"), hf!("0x1.ad9c178106693p-94")],
    [hf!("0x1.205d134p-14"), hf!("-0x1.214a2e893fccp-43"), hf!("0x1.548a9500c9822p-93")],
    [hf!("0x1.3685e28p-14"), hf!("0x1.e23588646103p-43"), hf!("0x1.2a97b26da2d88p-94")],
    [hf!("0x1.4cb6c18p-14"), hf!("0x1.2b7cfcea9e0d8p-42"), hf!("-0x1.5095048a6b824p-93")],
    [hf!("0x1.62e7bp-14"), hf!("-0x1.868625640a68p-44"), hf!("-0x1.34bf0db910f65p-93")],
  ],
];

#[rustfmt::skip]
static REF_CH: [[f64; 2]; 3] = [
  [hf!("0x1p-1"), hf!("0x1.24b67ee516e3bp-111")],
  [hf!("-0x1p-2"), hf!("-0x1.932ce43199a8dp-110")],
  [hf!("0x1.5555555555555p-3"), hf!("0x1.55540c15cf91fp-57")],
];

#[rustfmt::skip]
static REF_CL: [f64; 3] = [
  hf!("-0x1p-3"), hf!("0x1.9999999a0754fp-4"), hf!("-0x1.55555555c3157p-4"),
];

#[cold]
#[inline(never)]
fn as_atanh_zero(x: f64) -> f64 {
    let x2 = x * x;
    let x2l = x.mul_add(x, -x2);
    let mut y2 = x2
        * (ZERO_CL[0]
            + x2 * (ZERO_CL[1] + x2 * (ZERO_CL[2] + x2 * (ZERO_CL[3] + x2 * ZERO_CL[4]))));
    let mut y1 = polydd(x2, x2l, 13, &ZERO_CH, &mut y2);
    y1 = mulddd(y1, y2, x, &mut y2);
    y1 = muldd_acc(x2, x2l, y1, y2, &mut y2);
    let y0 = fasttwosum(x, y1, &mut y1);
    y1 = fasttwosum(y1, y2, &mut y2);
    /* We have 22 failures (only for RNDN, 11 up to sign) if we disable this
    check, all with 53 or 54 identical bits after the round bit.
    For example x=0x1.cc7092205cfafp-10 we have y0=0x1.cc70b12857108p-10
    and y1=-0x1p-63. */
    let mut t = y1.to_bits();
    if t & MANT_MASK == 0 {
        let w = y2.to_bits();
        if (w ^ t) >> 63 != 0 {
            t = t.wrapping_sub(1);
        } else {
            t = t.wrapping_add(1);
        }
        y1 = f64::from_bits(t);
    }
    y0 + y1
}

#[cold]
#[inline(never)]
fn as_atanh_database(x: f64, mut f: f64) -> f64 {
    let ax = x.abs();
    let sgn = 1.0f64.copysign(x);
    let mut a: i32 = 0;
    let mut b: i32 = DB.len() as i32 - 1;
    let mut m: i32 = (a + b) / 2;
    while a <= b {
        // binary search
        let e = &DB[m as usize];
        if e[0] < ax {
            a = m + 1;
        } else if e[0] == ax {
            f = sgn * e[1] + sgn * e[2];
            break;
        } else {
            b = m - 1;
        }
        m = (a + b) / 2;
    }
    f
}

#[cold]
#[inline(never)]
fn as_atanh_refine(x: f64, zh: f64, zl: f64, a: f64) -> f64 {
    let mut t = zh.to_bits();
    let ex = (t >> 52) as i32;
    let e = ex - 0x3ff;
    t &= MANT_MASK;
    t |= 0x3ffu64 << 52;
    let ed = e as f64;
    let v = a - ed + hf!("0x1.00008p+0");
    let i = v.to_bits().wrapping_sub(0x3ffu64 << 52) >> (52 - 16);
    let i1 = (i >> 12) as usize;
    let i2 = ((i >> 8) & 0xf) as usize;
    let i3 = ((i >> 4) & 0xf) as usize;
    let i4 = (i & 0xf) as usize;
    let l20 = hf!("0x1.62e42fefa3ap-2");
    let l21 = hf!("-0x1.0ca86c3898dp-50");
    let l22 = hf!("0x1.f97b57a079ap-104");
    let el2 = l22 * ed;
    let el1 = l21 * ed;
    let el0 = l20 * ed;
    let mut ll = [0.0f64; 3];
    ll[0] = LL[0][i1][0] + LL[1][i2][0] + (LL[2][i3][0] + LL[3][i4][0]) + el0;
    ll[1] = LL[0][i1][1] + LL[1][i2][1] + (LL[2][i3][1] + LL[3][i4][1]);
    ll[2] = LL[0][i1][2] + LL[1][i2][2] + (LL[2][i3][2] + LL[3][i4][2]);
    let t12 = RT1[i1] * RT2[i2];
    let t34 = RT3[i3] * RT4[i4];
    let th = t12 * t34;
    let tl = t12.mul_add(t34, -th);
    let mut tf = f64::from_bits(t);
    let dh = th * tf;
    let dl = th.mul_add(tf, -dh);
    let mut sh = tl * tf;
    let mut sl = tl.mul_add(tf, -sh);
    let mut xl = 0.0;
    let mut xh = fasttwosum(dh - 1.0, dl, &mut xl);
    tf = zl;
    t = tf.to_bits().wrapping_sub((e as i64 as u64) << 52);
    tf = f64::from_bits(t);
    xl += th * tf;
    xh = adddd(xh, xl, sh, sl, &mut xl);
    sl = xh * (REF_CL[0] + xh * (REF_CL[1] + xh * REF_CL[2]));
    sh = polydd(xh, xl, 3, &REF_CH, &mut sl);
    sh = muldd_acc(xh, xl, sh, sl, &mut sl);
    sh = adddd(sh, sl, el1, el2, &mut sl);
    sh = adddd(sh, sl, ll[1], ll[2], &mut sl);
    let mut v2 = 0.0;
    let mut v0 = fasttwosum(ll[0], sh, &mut v2);
    let mut v1 = fasttwosum(v2, sl, &mut v2);
    let _ = v2;
    t = v1.to_bits();
    if t & MANT_MASK == 0 {
        // is v1 a power of 2?
        /* we enter here only for 5 values up to sign:
        * 0x1.110e96a6c2d96p-2: 49 identical bits after round bit
        * 0x1.2dbb7b1c91363p-2: 50 identical bits after round bit
        * 0x1.c493dc899e4a5p-2: 50 identical bits after round bit (only with FMA)
        * 0x1.dc3fe1b524821p-2: 49 identical bits after round bit (only without FMA)
        * 0x1.64279e7c7064bp-1: 50 identical bits after round bit (only without FMA)
        In all cases v1 and v2 are of same sign, thus we add sign(v1)*ulp(v1) to v1.
        */
        t = t.wrapping_add(1);
        v1 = f64::from_bits(t);
    }
    let t0 = v0.to_bits();
    let er: u64 = t.wrapping_add(1) & MANT_MASK;
    let de: u64 = ((t0 >> 52) & 0x7ff).wrapping_sub((t >> 52) & 0x7ff);
    v0 *= 1.0f64.copysign(x);
    v1 *= 1.0f64.copysign(x);
    let res = v0 + v1;
    if de > 104 || er < 3 {
        return as_atanh_database(x, res);
    }
    res
}

/// CORE-MATH `cr_atanh` (atanh.c): the correctly rounded inverse hyperbolic tangent (round to
/// nearest).
pub fn cr_atanh(x: f64) -> f64 {
    let ax = x.abs();
    let aix = ax.to_bits();
    if aix >= 0x3ff0000000000000 {
        // |x| >= 1
        if aix == 0x3ff0000000000000 {
            // |x| = 1: pole error (CORE_MATH_SUPPORT_ERRNO's errno = ERANGE is not ported)
            return 1.0f64.copysign(x) / 0.0;
        }
        if aix > 0x7ff0000000000000 {
            return x + x; // nan
        }
        // domain error (errno = EDOM is not ported): the C returns `__builtin_sqrt(-1.0)`, which on
        // x86-64 is the "real indefinite" qNaN (sign bit set); spelled out so that it is the same
        // bit pattern on every target (sqrt of a negative number is a NaN of unspecified sign on wasm).
        return f64::from_bits(0xfff8000000000000);
    }

    if aix < 0x3fd0000000000000 {
        // |x| < 0x1p-2
        // atanh(x) rounds to x to nearest for |x| < 0x1.d12ed0af1a27fp-27
        if aix < 0x3e4d12ed0af1a27f {
            // |x| < 0x1.d12ed0af1a27fp-27
            /* We have underflow exactly when 0 < |x| < 2^-1022:
            for RNDU, atanh(2^-1022-2^-1074) would round to 2^-1022-2^-1075
            with unbounded exponent range. (errno = ERANGE is not ported.) */
            return x.mul_add(hf!("0x1p-55"), x);
        }
        /* checked exhautively this branch (with and without FMA):
         * for 0x1.d12ed0af1a27fp-27 <= x < 2^-24
         */
        let x2 = x * x;
        /* the polynomial c(x) = c[0]+c[1]*x^2+...+c[8]*x^16 is a minimax
        approximation of (atanh(x) - x - x^3/3)/x^5, which yields for
        p(x) = x + (0x1.5555555555555p-2 + 0x1.5555555555555p-56) x^3 + x^5 c(x)
        an approximation of atanh(x) with relative error <= 2^-64.377
        on [0x1.d12ed0af1a27fp-27,1/4], cf atanh.sollya */
        let c = &SMALL_C;
        let dx2 = x.mul_add(x, -x2);
        let x4 = x2 * x2;
        let x3 = x2 * x;
        let x8 = x4 * x4;
        let dx3 = x2.mul_add(x, -x3) + dx2 * x;
        /* following the C standard, the following is equivalent to (a + b) + c,
        where a = c[0] + x2*c[1], b = x4*(...) and c = x8*(...) */
        let p = (c[0] + x2 * c[1])
            + x4 * (c[2] + x2 * c[3])
            + x8 * (((c[4] + x2 * c[5]) + x4 * (c[6] + x2 * c[7])) + x8 * c[8]);
        let t = x2.mul_add(p, hf!("0x1.5555555555555p-56"));
        let mut pl = 0.0;
        let mut ph = fasttwosum(hf!("0x1.5555555555555p-2"), t, &mut pl);
        ph = muldd(ph, pl, x3, dx3, &mut pl);
        let mut tl = 0.0;
        ph = fasttwosum(x, ph, &mut tl);
        pl += tl;
        let eps = x * (x4 * hf!("0x1.dp-53") + hf!("0x1p-103"));
        // revision 221543c fails with 0.737*eps and x=0x1.2f67d96be6eafp-9 (rndu)
        let lb = ph + (pl - eps);
        let ub = ph + (pl + eps);
        if lb == ub {
            return lb;
        }
        return as_atanh_zero(x);
    }

    /* We have performed an exhaustive search of branch |x| >= 0.25 on 221543c
    both with default CFLAGS (with FMA) and with CFLAGS="-O3 -march=x86-64"
    (no FMA). We found no failure, except the ones in atanh.wc which are fixed
    in a later commit. */
    let mut pl = 0.0;
    let ph = fasttwosum(1.0, ax, &mut pl);
    let mut ql = 0.0;
    let qh = fasttwosub(1.0, ax, &mut ql);
    let iqh = 1.0 / qh;
    let mut th = ph * iqh;
    let mut tl = ph.mul_add(iqh, -th) + (pl + ph * ((-qh).mul_add(iqh, 1.0) - ql * iqh)) * iqh;

    let mut t = th.to_bits();
    let ex = (t >> 52) as i32;
    let e = ex - 0x3ff;
    t &= MANT_MASK;
    let ed = e as f64;
    let i = (t >> (52 - 5)) as usize;
    let d = (t & (!0u64 >> 17)) as i64;
    let j = t
        .wrapping_add((B[i].0 as u64) << 33)
        .wrapping_add((B[i].1 as i64 * (d >> 16)) as u64)
        >> (52 - 10);
    t |= 0x3ffu64 << 52;
    let i1 = (j >> 5) as usize;
    let i2 = (j & 0x1f) as usize;
    let tf = f64::from_bits(t);
    let r = (0.5 * R1[i1]) * R2[i2];
    let dx = r.mul_add(tf, -0.5);
    let dx2 = dx * dx;
    let rx = r * tf;
    let dxl = r.mul_add(tf, -rx);
    let f = dx2 * ((C[0] + dx * C[1]) + dx2 * (C[2] + dx * C[3] + dx2 * C[4]));
    let l2h = hf!("0x1.62e42fefa3ap-2");
    let l2l = hf!("-0x1.0ca86c3898dp-50");
    let mut lh = (L1[i1][1] + L2[i2][1]) + l2h * ed;
    let mut ll = 0.0;
    lh = fasttwosum(lh, rx - 0.5, &mut ll);
    ll += l2l * ed + (L1[i1][0] + L2[i2][0]) + dxl + 0.5 * tl / th;
    ll += f;
    lh *= 1.0f64.copysign(x);
    ll *= 1.0f64.copysign(x);
    let eps = 38e-24 + dx2 * hf!("0x1p-49");
    // revision bbe3f8c fails with 0.989 * eps and x=0x1.ebf0bfefa727dp-1 (rndz)
    let lb = lh + (ll - eps);
    let ub = lh + (ll + eps);
    if lb == ub {
        return lb;
    }
    th = fasttwosum(th, tl, &mut tl);
    as_atanh_refine(x, th, tl, hf!("0x1.71547652b82fep+1") * (lh + ll).abs())
}

/// C `atanh(double)`, bit-identical to glibc 2.44 (CORE-MATH, correctly rounded) on every
/// target. Use instead of `f64::atanh`. errno and the exception flags are not set.
pub fn atanh(x: f64) -> f64 {
    cr_atanh(x)
}

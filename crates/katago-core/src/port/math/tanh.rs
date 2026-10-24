//! C `tanh` (double): CORE-MATH's correctly rounded `cr_tanh`, which is what glibc 2.44 uses
//! for `tanh`, so `port::math::tanh` is bit-identical to glibc on every target.
//!
//! Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/,
//! https://gitlab.inria.fr/core-math/core-math), src/binary64/tanh/tanh.c at commit
//! 0ce0cbc15935e24c656280e5a46c34e89f301b7a (2026-10-20). Correctly rounded.
//!
//! tanh.c: Copyright (c) 2023-2026 Alexei Sibidanov, Cyprien Peignier, Paul Zimmermann.
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
//!   `wrapping_*`. The x86-64-only SSE masking of the exponent shift (`_mm_and_pd`) is the same
//!   bit mask as the portable `#else` branch of the C, which is what is ported.
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

const MANT_MASK: u64 = !0u64 >> 12; // ~0ul >> 12

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
fn muldd_acc(xh: f64, xl: f64, ch: f64, cl: f64, l: &mut f64) -> f64 {
    let plh = xl * ch;
    let phl = xh * cl;
    let phh = xh * ch;
    let mut phh_rest = xh.mul_add(ch, -phh);
    phh_rest += phl + plh;
    fasttwosum(phh, phh_rest, l)
}

#[inline(always)]
fn mulddd_acc(xh: f64, xl: f64, c: f64, l: &mut f64) -> f64 {
    let pl = xl * c;
    let ph = xh * c;
    let mut ph_rest = xh.mul_add(c, -ph);
    ph_rest += pl;
    fasttwosum(ph, ph_rest, l)
}

/// At input, `l` is the approximation of the upper part of the polynomial (evaluated with
/// double arithmetic only).
#[inline(always)]
fn polydd(xh: f64, xl: f64, n: usize, c: &[[f64; 2]], l: &mut f64) -> f64 {
    let mut i = n - 1;
    let mut t = 0.0;
    let mut ch = fasttwosum(c[i][0], *l, &mut t);
    let mut cl = t + c[i][1];
    // ch + cl ~= c[i][0] + c[i][1] + *l
    while i > 0 {
        i -= 1;
        ch = muldd_acc(xh, xl, ch, cl, &mut cl);
        let mut tl = 0.0;
        let th = fasttwosum(c[i][0], ch, &mut tl);
        ch = th;
        cl = (cl + c[i][1]) + tl;
    }
    *l = cl;
    ch
}

#[rustfmt::skip]
static T0: [[f64; 2]; 64] = [
  [hf!("0x0p+0"), hf!("0x1p+0")],
  [hf!("-0x1.19083535b085ep-56"), hf!("0x1.02c9a3e778061p+0")],
  [hf!("0x1.d73e2a475b466p-55"), hf!("0x1.059b0d3158574p+0")],
  [hf!("0x1.186be4bb285p-57"), hf!("0x1.0874518759bc8p+0")],
  [hf!("0x1.8a62e4adc610ap-54"), hf!("0x1.0b5586cf9890fp+0")],
  [hf!("0x1.03a1727c57b52p-59"), hf!("0x1.0e3ec32d3d1a2p+0")],
  [hf!("-0x1.6c51039449b3ap-54"), hf!("0x1.11301d0125b51p+0")],
  [hf!("-0x1.32fbf9af1369ep-54"), hf!("0x1.1429aaea92dep+0")],
  [hf!("-0x1.19041b9d78a76p-55"), hf!("0x1.172b83c7d517bp+0")],
  [hf!("0x1.e5b4c7b4968e4p-55"), hf!("0x1.1a35beb6fcb75p+0")],
  [hf!("0x1.e016e00a2643cp-54"), hf!("0x1.1d4873168b9aap+0")],
  [hf!("0x1.dc775814a8494p-55"), hf!("0x1.2063b88628cd6p+0")],
  [hf!("0x1.9b07eb6c70572p-54"), hf!("0x1.2387a6e756238p+0")],
  [hf!("0x1.2bd339940e9dap-55"), hf!("0x1.26b4565e27cddp+0")],
  [hf!("0x1.612e8afad1256p-55"), hf!("0x1.29e9df51fdee1p+0")],
  [hf!("0x1.0024754db41d4p-54"), hf!("0x1.2d285a6e4030bp+0")],
  [hf!("0x1.6f46ad23182e4p-55"), hf!("0x1.306fe0a31b715p+0")],
  [hf!("0x1.32721843659a6p-54"), hf!("0x1.33c08b26416ffp+0")],
  [hf!("-0x1.63aeabf42eae2p-54"), hf!("0x1.371a7373aa9cbp+0")],
  [hf!("-0x1.5e436d661f5e2p-56"), hf!("0x1.3a7db34e59ff7p+0")],
  [hf!("0x1.ada0911f09ebcp-55"), hf!("0x1.3dea64c123422p+0")],
  [hf!("-0x1.ef3691c309278p-58"), hf!("0x1.4160a21f72e2ap+0")],
  [hf!("0x1.89b7a04ef80dp-59"), hf!("0x1.44e086061892dp+0")],
  [hf!("0x1.3c1a3b69062fp-56"), hf!("0x1.486a2b5c13cdp+0")],
  [hf!("0x1.d4397afec42e2p-56"), hf!("0x1.4bfdad5362a27p+0")],
  [hf!("-0x1.4b309d25957e4p-54"), hf!("0x1.4f9b2769d2ca7p+0")],
  [hf!("-0x1.07abe1db13cacp-55"), hf!("0x1.5342b569d4f82p+0")],
  [hf!("0x1.9bb2c011d93acp-54"), hf!("0x1.56f4736b527dap+0")],
  [hf!("0x1.6324c054647acp-54"), hf!("0x1.5ab07dd485429p+0")],
  [hf!("0x1.ba6f93080e65ep-54"), hf!("0x1.5e76f15ad2148p+0")],
  [hf!("-0x1.383c17e40b496p-54"), hf!("0x1.6247eb03a5585p+0")],
  [hf!("-0x1.bb60987591c34p-54"), hf!("0x1.6623882552225p+0")],
  [hf!("-0x1.bdd3413b26456p-54"), hf!("0x1.6a09e667f3bcdp+0")],
  [hf!("-0x1.bbe3a683c88aap-57"), hf!("0x1.6dfb23c651a2fp+0")],
  [hf!("-0x1.16e4786887a9ap-55"), hf!("0x1.71f75e8ec5f74p+0")],
  [hf!("-0x1.0245957316dd4p-54"), hf!("0x1.75feb564267c9p+0")],
  [hf!("-0x1.41577ee04993p-55"), hf!("0x1.7a11473eb0187p+0")],
  [hf!("0x1.05d02ba15797ep-56"), hf!("0x1.7e2f336cf4e62p+0")],
  [hf!("-0x1.d4c1dd41532d8p-54"), hf!("0x1.82589994cce13p+0")],
  [hf!("-0x1.fc6f89bd4f6bap-54"), hf!("0x1.868d99b4492edp+0")],
  [hf!("0x1.6e9f156864b26p-54"), hf!("0x1.8ace5422aa0dbp+0")],
  [hf!("0x1.5cc13a2e3976cp-55"), hf!("0x1.8f1ae99157736p+0")],
  [hf!("-0x1.75fc781b57ebcp-57"), hf!("0x1.93737b0cdc5e5p+0")],
  [hf!("-0x1.d185b7c1b85dp-54"), hf!("0x1.97d829fde4e5p+0")],
  [hf!("0x1.c7c46b071f2bep-56"), hf!("0x1.9c49182a3f09p+0")],
  [hf!("-0x1.359495d1cd532p-54"), hf!("0x1.a0c667b5de565p+0")],
  [hf!("-0x1.d2f6edb8d41e2p-54"), hf!("0x1.a5503b23e255dp+0")],
  [hf!("0x1.0fac90ef7fd32p-54"), hf!("0x1.a9e6b5579fdbfp+0")],
  [hf!("0x1.7a1cd345dcc82p-54"), hf!("0x1.ae89f995ad3adp+0")],
  [hf!("-0x1.2805e3084d708p-57"), hf!("0x1.b33a2b84f15fbp+0")],
  [hf!("-0x1.5584f7e54ac3ap-56"), hf!("0x1.b7f76f2fb5e47p+0")],
  [hf!("0x1.23dd07a2d9e84p-55"), hf!("0x1.bcc1e904bc1d2p+0")],
  [hf!("0x1.11065895048dep-55"), hf!("0x1.c199bdd85529cp+0")],
  [hf!("0x1.2884dff483cacp-54"), hf!("0x1.c67f12e57d14bp+0")],
  [hf!("0x1.503cbd1e949dcp-56"), hf!("0x1.cb720dcef9069p+0")],
  [hf!("-0x1.cbc3743797a9cp-54"), hf!("0x1.d072d4a07897cp+0")],
  [hf!("0x1.2ed02d75b3706p-55"), hf!("0x1.d5818dcfba487p+0")],
  [hf!("0x1.c2300696db532p-54"), hf!("0x1.da9e603db3285p+0")],
  [hf!("-0x1.1a5cd4f184b5cp-54"), hf!("0x1.dfc97337b9b5fp+0")],
  [hf!("0x1.39e8980a9cc9p-55"), hf!("0x1.e502ee78b3ff6p+0")],
  [hf!("-0x1.e9c23179c2894p-54"), hf!("0x1.ea4afa2a490dap+0")],
  [hf!("0x1.dc7f486a4b6bp-54"), hf!("0x1.efa1bee615a27p+0")],
  [hf!("0x1.9d3e12dd8a18ap-54"), hf!("0x1.f50765b6e454p+0")],
  [hf!("0x1.74853f3a5931ep-55"), hf!("0x1.fa7c1819e90d8p+0")],
];

#[rustfmt::skip]
static T1: [[f64; 2]; 64] = [
  [hf!("0x0p+0"), hf!("0x1p+0")],
  [hf!("0x1.ae8e38c59c72ap-54"), hf!("0x1.000b175effdc7p+0")],
  [hf!("-0x1.7b5d0d58ea8f4p-58"), hf!("0x1.00162f3904052p+0")],
  [hf!("0x1.4115cb6b16a8ep-54"), hf!("0x1.0021478e11ce6p+0")],
  [hf!("-0x1.d7c96f201bb2ep-55"), hf!("0x1.002c605e2e8cfp+0")],
  [hf!("0x1.84711d4c35eap-54"), hf!("0x1.003779a95f959p+0")],
  [hf!("-0x1.0484245243778p-55"), hf!("0x1.0042936faa3d8p+0")],
  [hf!("-0x1.4b237da2025fap-54"), hf!("0x1.004dadb113dap+0")],
  [hf!("-0x1.5e00e62d6b30ep-56"), hf!("0x1.0058c86da1c0ap+0")],
  [hf!("0x1.a1d6cedbb948p-54"), hf!("0x1.0063e3a559473p+0")],
  [hf!("-0x1.4acf197a00142p-54"), hf!("0x1.006eff583fc3dp+0")],
  [hf!("-0x1.eaf2ea42391a6p-57"), hf!("0x1.007a1b865a8cap+0")],
  [hf!("0x1.da93f90835f76p-56"), hf!("0x1.0085382faef83p+0")],
  [hf!("-0x1.6a79084ab093cp-55"), hf!("0x1.00905554425d4p+0")],
  [hf!("0x1.86364f8fbe8f8p-54"), hf!("0x1.009b72f41a12bp+0")],
  [hf!("-0x1.82e8e14e3110ep-55"), hf!("0x1.00a6910f3b6fdp+0")],
  [hf!("-0x1.4f6b2a7609f72p-55"), hf!("0x1.00b1afa5abcbfp+0")],
  [hf!("-0x1.e1a258ea8f71ap-56"), hf!("0x1.00bcceb7707ecp+0")],
  [hf!("0x1.4362ca5bc26f2p-56"), hf!("0x1.00c7ee448ee02p+0")],
  [hf!("0x1.095a56c919d02p-54"), hf!("0x1.00d30e4d0c483p+0")],
  [hf!("-0x1.406ac4e81a646p-57"), hf!("0x1.00de2ed0ee0f5p+0")],
  [hf!("0x1.b5a6902767e08p-54"), hf!("0x1.00e94fd0398ep+0")],
  [hf!("-0x1.91b206085932p-54"), hf!("0x1.00f4714af41d3p+0")],
  [hf!("0x1.427068ab22306p-55"), hf!("0x1.00ff93412315cp+0")],
  [hf!("0x1.c1d0660524e08p-54"), hf!("0x1.010ab5b2cbd11p+0")],
  [hf!("-0x1.e7bdfb3204be8p-54"), hf!("0x1.0115d89ff3a8bp+0")],
  [hf!("0x1.843aa8b9cbbc6p-55"), hf!("0x1.0120fc089ff63p+0")],
  [hf!("-0x1.34104ee7edae8p-56"), hf!("0x1.012c1fecd613bp+0")],
  [hf!("-0x1.2b6aeb6176892p-56"), hf!("0x1.0137444c9b5b5p+0")],
  [hf!("0x1.a8cd33b8a1bb2p-56"), hf!("0x1.01426927f5278p+0")],
  [hf!("0x1.2edc08e5da99ap-56"), hf!("0x1.014d8e7ee8d2fp+0")],
  [hf!("0x1.57ba2dc7e0c72p-55"), hf!("0x1.0158b4517bb88p+0")],
  [hf!("0x1.b61299ab8cdb8p-54"), hf!("0x1.0163da9fb3335p+0")],
  [hf!("-0x1.90565902c5f44p-54"), hf!("0x1.016f0169949edp+0")],
  [hf!("0x1.70fc41c5c2d54p-55"), hf!("0x1.017a28af25567p+0")],
  [hf!("0x1.4b9a6e145d76cp-54"), hf!("0x1.018550706ab62p+0")],
  [hf!("-0x1.008eff5142bfap-56"), hf!("0x1.019078ad6a19fp+0")],
  [hf!("-0x1.77669f033c7dep-54"), hf!("0x1.019ba16628de2p+0")],
  [hf!("-0x1.09bb78eeead0ap-54"), hf!("0x1.01a6ca9aac5f3p+0")],
  [hf!("0x1.371231477ece6p-54"), hf!("0x1.01b1f44af9f9ep+0")],
  [hf!("0x1.5e7626621eb5ap-56"), hf!("0x1.01bd1e77170b4p+0")],
  [hf!("-0x1.bc72b100828a4p-54"), hf!("0x1.01c8491f08f08p+0")],
  [hf!("-0x1.ce39cbbab8bbep-57"), hf!("0x1.01d37442d507p+0")],
  [hf!("0x1.16996709da2e2p-55"), hf!("0x1.01de9fe280ac8p+0")],
  [hf!("-0x1.c11f5239bf536p-55"), hf!("0x1.01e9cbfe113efp+0")],
  [hf!("0x1.e1d4eb5edc6b4p-55"), hf!("0x1.01f4f8958c1c6p+0")],
  [hf!("-0x1.afb99946ee3fp-54"), hf!("0x1.020025a8f6a35p+0")],
  [hf!("-0x1.8f06d8a148a32p-54"), hf!("0x1.020b533856324p+0")],
  [hf!("-0x1.2bf310fc54eb6p-55"), hf!("0x1.02168143b0281p+0")],
  [hf!("-0x1.c95a035eb4176p-54"), hf!("0x1.0221afcb09e3ep+0")],
  [hf!("-0x1.491793e46834cp-54"), hf!("0x1.022cdece68c4fp+0")],
  [hf!("-0x1.3e8d0d9c4909p-56"), hf!("0x1.02380e4dd22adp+0")],
  [hf!("-0x1.314aa16278aa4p-54"), hf!("0x1.02433e494b755p+0")],
  [hf!("0x1.48daf888e965p-55"), hf!("0x1.024e6ec0da046p+0")],
  [hf!("0x1.56dc8046821f4p-55"), hf!("0x1.02599fb483385p+0")],
  [hf!("0x1.45b42356b9d46p-54"), hf!("0x1.0264d1244c719p+0")],
  [hf!("-0x1.082ef51b61d7ep-56"), hf!("0x1.027003103b10ep+0")],
  [hf!("0x1.2106ed0920a34p-56"), hf!("0x1.027b357854772p+0")],
  [hf!("-0x1.fd4cf26ea5d0ep-54"), hf!("0x1.0286685c9e059p+0")],
  [hf!("-0x1.09f8775e78084p-54"), hf!("0x1.02919bbd1d1d8p+0")],
  [hf!("0x1.64cbba902ca28p-58"), hf!("0x1.029ccf99d720ap+0")],
  [hf!("0x1.4383ef231d206p-54"), hf!("0x1.02a803f2d170dp+0")],
  [hf!("0x1.4a47a505b3a46p-54"), hf!("0x1.02b338c811703p+0")],
  [hf!("0x1.e47120223468p-54"), hf!("0x1.02be6e199c811p+0")],
];

#[rustfmt::skip]
static ZERO_CH: [[f64; 2]; 9] = [
  [hf!("-0x1.5555555555555p-2"), hf!("-0x1.5555555554cc4p-56")],
  [hf!("0x1.1111111111111p-3"), hf!("0x1.111110f8c0178p-59")],
  [hf!("-0x1.ba1ba1ba1ba1cp-5"), hf!("0x1.7917c1d676ff5p-59")],
  [hf!("0x1.664f4882c10fap-6"), hf!("-0x1.9d5cb27c0af28p-63")],
  [hf!("-0x1.226e355e6c23cp-7"), hf!("-0x1.c9674586913f3p-61")],
  [hf!("0x1.d6d3d0e157db3p-9"), hf!("-0x1.71376fa06ce94p-65")],
  [hf!("-0x1.7da36452b5e46p-10"), hf!("-0x1.aba8d51bd9cp-65")],
  [hf!("0x1.3558247faa32dp-11"), hf!("-0x1.e0cfb423aedfdp-65")],
  [hf!("-0x1.f57d76ea30928p-13"), hf!("-0x1.c30601213cae9p-67")],
];

#[rustfmt::skip]
static ZERO_CL: [f64; 5] = [
  hf!("0x1.967e0a63ca836p-14"), hf!("-0x1.497b99d2a77d1p-15"), hf!("0x1.0ae346258cbdep-16"),
  hf!("-0x1.aade68fb2f076p-18"), hf!("0x1.22e609bf8671fp-19"),
];

#[rustfmt::skip]
static DB: [[f64; 3]; 12] = [
  [hf!("0x1.ac343b179fec4p-3"), hf!("0x1.a612499c53078p-3"), hf!("0x1p-57")],
  [hf!("0x1.00764a988bf73p-2"), hf!("0x1.f676484c0703bp-3"), hf!("-0x1p-104")],
  [hf!("0x1.17d1e8a63711fp-2"), hf!("0x1.110e96a6c2d96p-2"), hf!("-0x1p-105")],
  [hf!("0x1.291c601a05276p-2"), hf!("0x1.210b7d0c03743p-2"), hf!("0x1p-56")],
  [hf!("0x1.36f33d51c264dp-2"), hf!("0x1.2dbb7b1c91363p-2"), hf!("-0x1p-106")],
  [hf!("0x1.43eaea23649c3p-2"), hf!("0x1.39877ed028641p-2"), hf!("-0x1p-55")],
  [hf!("0x1.d88d7550b2826p-2"), hf!("0x1.b9a3637366afdp-2"), hf!("0x1p-56")],
  [hf!("0x1.e611aa58ab608p-2"), hf!("0x1.c493dc899e4a6p-2"), hf!("-0x1p-55")],
  [hf!("0x1.01efe7ac8c15dp-1"), hf!("0x1.dc3fe1b524821p-2"), hf!("-0x1p-104")],
  [hf!("0x1.1005ec0bccabbp-1"), hf!("0x1.f20b1c8557dedp-2"), hf!("-0x1p-55")],
  [hf!("0x1.33dfeb0fa4bfep-1"), hf!("0x1.1372f9ee76e99p-1"), hf!("0x1p-55")],
  [hf!("0x1.49f24ac5cac35p-1"), hf!("0x1.22c495ff06104p-1"), hf!("-0x1p-104")],
];

#[rustfmt::skip]
static EXP_CH: [[f64; 2]; 3] = [
  [hf!("0x1p+0"), hf!("0x1.6c16bd194535dp-94")],
  [hf!("0x1p-1"), hf!("-0x1.8259d904fd34fp-93")],
  [hf!("0x1.5555555555555p-3"), hf!("0x1.53e93e9f26e62p-57")],
];

#[rustfmt::skip]
static CH: [f64; 4] = [
  hf!("0x1p+1"), hf!("0x1p+1"), hf!("0x1.55555557e54ffp+0"), hf!("0x1.55555553a12f4p-1"),
];

#[rustfmt::skip]
static SMALL_C: [f64; 8] = [
  hf!("-0x1.5555555555555p-2"), hf!("0x1.1111111110f33p-3"), hf!("-0x1.ba1ba1b9b8ea6p-5"),
  hf!("0x1.664f4838e0a43p-6"), hf!("-0x1.226e17d1bc09bp-7"), hf!("0x1.d6c64dfba2565p-9"),
  hf!("-0x1.7bdd094d327afp-10"), hf!("0x1.1535ad0c31d0ep-11"),
];

fn as_exp_accurate(x: f64, t: f64, th: f64, tl: f64, l: &mut f64) -> f64 {
    let l2h = hf!("0x1.62e42ffp-13");
    let l2l = hf!("0x1.718432a1b0e26p-47");
    let l2ll = hf!("0x1.9ff0342542fc3p-102");
    let dx = x - l2h * t;
    let mut dxl = l2l * t;
    let dxll = l2ll * t + l2l.mul_add(t, -dxl);
    let dxh = dx + dxl;
    dxl = ((dx - dxh) + dxl) + dxll;
    let mut fl = dxh
        * (hf!("0x1.5555555555555p-5")
            + dxh * (hf!("0x1.11111113e93e9p-7") + dxh * hf!("0x1.6c16c169400a7p-10")));
    let mut fh = polydd(dxh, dxl, 3, &EXP_CH, &mut fl);
    fh = muldd_acc(dxh, dxl, fh, fl, &mut fl);
    fh = muldd_acc(th, tl, fh, fl, &mut fl);
    let zh = th + fh;
    let zl = (th - zh) + fh;
    let uh = zh + tl;
    let ul = ((zh - uh) + tl) + zl;
    let vh = uh + fl;
    let vl = ((uh - vh) + fl) + ul;
    *l = vl;
    vh
}

/// |x| < 0.25
#[cold]
#[inline(never)]
fn as_tanh_zero(x: f64) -> f64 {
    let x2 = x * x;
    let x2l = x.mul_add(x, -x2);
    let mut y2 = x2
        * (ZERO_CL[0]
            + x2 * (ZERO_CL[1] + x2 * (ZERO_CL[2] + x2 * (ZERO_CL[3] + x2 * ZERO_CL[4]))));
    let mut y1 = polydd(x2, x2l, 9, &ZERO_CH, &mut y2);
    y1 = mulddd_acc(y1, y2, x, &mut y2);
    y1 = muldd_acc(y1, y2, x2, x2l, &mut y2);
    let y0 = fasttwosum(x, y1, &mut y1);
    y1 = fasttwosum(y1, y2, &mut y2);
    let mut t = y1.to_bits();
    if t & MANT_MASK == 0 {
        let w = y2.to_bits();
        if (w ^ t) >> 63 != 0 {
            t = t.wrapping_sub(1);
        } else {
            t = t.wrapping_add(1);
        }
        y1 = f64::from_bits(t);
        if x.abs() == hf!("0x1.ac343b179fec4p-3") {
            return as_tanh_database(x, y0 + y1);
        }
    }
    y0 + y1
}

#[cold]
#[inline(never)]
fn as_tanh_database(x: f64, mut f: f64) -> f64 {
    let mut a: i32 = 0;
    let mut b: i32 = DB.len() as i32 - 1;
    let mut m: i32 = (a + b) / 2;
    let ax = x.abs();
    while a <= b {
        let e = &DB[m as usize];
        if e[0] < ax {
            a = m + 1;
        } else if e[0] == ax {
            f = 1.0f64.copysign(x) * e[1] + 1.0f64.copysign(x) * e[2];
            break;
        } else {
            b = m - 1;
        }
        m = (a + b) / 2;
    }
    f
}

/// CORE-MATH `cr_tanh` (tanh.c): the correctly rounded hyperbolic tangent (round to nearest).
///
/// The function tanh(x) is approximated by a minimax polynomial for |x| < 0.25. For other values
/// the identity tanh(|x|) = 1 - 2*exp(-2*|x|)/(1 + exp(-2*|x|)) is used. For large |x| > 3.683
/// the term 2*exp(-2*|x|)/(1 + exp(-2*|x|)) becomes small and a less precise formula for the
/// exponential is used.
pub fn cr_tanh(x: f64) -> f64 {
    let ax = x.abs();
    let aix = ax.to_bits();
    /* for |x| >= 0x1.30fc1931f09cap+4, tanh(x) rounds to +1 or -1 to nearest,
    this avoid a spurious overflow in the computation of v0 below */
    if aix >= 0x40330fc1931f09ca {
        if aix > 0x7ff0000000000000 {
            return x + x; // nan
        }
        let f = 1.0f64.copysign(x);
        if aix == 0x7ff0000000000000 {
            return f;
        }
        let df = hf!("0x1p-55").copysign(x);
        return f - df;
    }
    let s = hf!("-0x1.71547652b82fep+13");
    let v0 = ax.mul_add(s, hf!("0x1.8000004p+25"));
    let jt = v0.to_bits();
    // v0 with its low 27 bits cleared (the C's `_mm_and_pd` on x86-64 / `v.u &= tt` elsewhere)
    let t = f64::from_bits(jt & !((1u64 << 27) - 1)) - hf!("0x1.8p25");
    let i1 = ((jt >> 27) & 0x3f) as usize;
    let i0 = ((jt >> 33) & 0x3f) as usize;
    let ie = ((jt << 13) as i64) >> 52;
    let sp = f64::from_bits(((1023 + ie) as u64) << 52);
    let t0h = T0[i0][1];
    let t1h = T1[i1][1];
    let mut th = t0h * t1h;
    let mut tl: f64;
    if aix < 0x400d76c8b4395810 {
        // |x| ~< 3.683
        if aix < 0x3fd0000000000000 {
            // |x| < 0x1p-2
            if aix <= 0x3e4d12ed0af1a27f {
                // |x| <= 0x1.d12ed0af1a27fp-27
                if aix == 0 {
                    return x;
                }
                /* We have underflow when 0 < |x| < 2^-1022 or when |x| = 2^-1022
                and rounding towards zero. (CORE_MATH_SUPPORT_ERRNO's errno = ERANGE is not ported.) */
                return x.mul_add(hf!("-0x1p-55"), x);
            } // endif |x| <= 0x1.d12ed0af1a27fp-27
            let c = &SMALL_C;
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;
            let p1 = (c[4] + x2 * c[5]) + x4 * (c[6] + x2 * c[7]);
            let mut p0 = (c[0] + x2 * c[1]) + x4 * (c[2] + x2 * c[3]);
            p0 += x8 * p1;
            p0 *= x3;
            let mut rl = 0.0;
            let rh = fasttwosum(x, p0, &mut rl);
            /* The branch 0x1.d12ed0af1a27fp-27 <= x < 0x1p-26 was checked
            exhaustively (with and without fma contraction) with revision 1820535,
            with the error bound e = x3*0x1.4dp-52.
            It fails with 0x1.4cp-52 and x=0x1.27a0e7f47f0fap-4 (rndz, no fma
            contraction).
            The interval [0x1p-3, 0x1.00cp-3] was
            checked exhaustively with rndz and without fma contraction,
            with error bound e = x3*0x1.80p-52: no failure.
            The interval [0x1.015891c9eaef8p-3, 0x1.019891c9eaef8p-3] was
            checked exhaustively with rndz and without fma contraction,
            with error bound e = x3*0x1.80p-52: no failure. */
            let e = x3 * hf!("0x1.c0p-52");
            let lb = rh + (rl - e);
            let ub = rh + (rl + e);
            if lb == ub {
                return lb;
            }
            return as_tanh_zero(x);
        } // endif |x| < 0x1p-2

        let t0l = T0[i0][0];
        let t1l = T1[i1][0];
        tl = t0h * t1l + t1h * t0l + t0h.mul_add(t1h, -th);
        th *= sp;
        tl *= sp;
        let l2h = hf!("-0x1.62e42ffp-14");
        let l2l = hf!("-0x1.718432a1b0e26p-48");
        let dx = (l2h * t - ax) - l2l * t;
        let dx2 = dx * dx;
        let p = dx * ((CH[0] + dx * CH[1]) + dx2 * (CH[2] + dx * CH[3]));
        let mut rh = th;
        let mut rl = tl + rh * p;
        rh = fasttwosum(rh, rl, &mut rl);

        let mut ph = rh;
        let mut pl = rl;
        let mut qh = rh;
        let mut ql = rl;
        let mut qd = 0.0;
        qh = fasttwosum(1.0, qh, &mut qd);
        ql += qd;

        let rqh = 1.0 / qh;
        let rql = (ql * rqh + rqh.mul_add(qh, -1.0)) * -rqh;
        ph = muldd_acc(ph, pl, rqh, rql, &mut pl);

        /* This branch was tested exhaustively with/without fma contraction.
        During this search, a failure was found with the original error
        bound (e = rh*0x1p-62) and x=+/-0x1.a0112a16e9318p+1 (rndu, no fma
        contraction). Another failure was found with x=+/-0x1.a0bd10af4ac2bp+1
        and e = rh*0x1.0cp-62 (rndu, no fma contraction). */
        let e = rh * hf!("0x1.0dp-62");
        rh = fasttwosub(0.5, ph, &mut rl);
        rl -= pl;
        rh *= 2.0f64.copysign(x);
        rl *= 2.0f64.copysign(x);
        let lb = rh + (rl - e);
        let ub = rh + (rl + e);
        if lb == ub {
            return lb;
        }
    } else {
        // 3.683 ~< x < 0x1.30fc1931f09cap+4
        let l2 = hf!("-0x1.62e42fefa39efp-14");
        let dx = l2.mul_add(t, -ax);
        let dx2 = dx * dx;
        let p = dx * ((CH[0] + dx * CH[1]) + dx2 * (CH[2] + dx * CH[3]));
        let mut rh = th * sp;
        rh += (p + ((2.0 * hf!("0x1.3p-55")) * ax)) * rh;
        /* This branch was tested exhaustively with/without fma contraction.
        During this search, the largest 9-bit value of e for which it fails was
        found to be e = rh*0x1.fap-50 with x=0x1.09cc2de69e78cp+2
        (rndu, with/without fma contraction). Thus the bound below can be
        reduced to rh*0x1.fbp-50. */
        let e = rh * hf!("0x1.1p-49");
        rh = (2.0 * rh) / (1.0 + rh);
        let one = 1.0f64.copysign(x);
        rh = rh.copysign(x);
        let lb = one - (rh + e);
        let ub = one - (rh - e);
        if lb == ub {
            return lb;
        }

        let t0l = T0[i0][0];
        let t1l = T1[i1][0];
        tl = t0h * t1l + t1h * t0l + t0h.mul_add(t1h, -th);
        th *= sp;
        tl *= sp;
    }
    let mut rl = 0.0;
    let mut rh = as_exp_accurate(-2.0 * ax, t, th, tl, &mut rl);
    let mut qd = 0.0;
    let mut qh = fasttwosum(1.0, rh, &mut qd);
    let mut ql = rl + qd;
    qh = fasttwosum(qh, ql, &mut ql);
    let rqh = 1.0 / qh;
    let rql = (ql * rqh + rqh.mul_add(qh, -1.0)) * -rqh;
    let mut pl = 0.0;
    let ph = muldd_acc(rh, rl, rqh, rql, &mut pl);
    rh = fasttwosub(0.5, ph, &mut rl);
    rl -= pl;
    rh = fasttwosum(rh, rl, &mut rl);
    let res = 2.0f64.copysign(x) * rh + 2.0f64.copysign(x) * rl;
    let lu = rl.to_bits();
    if (lu.wrapping_add(32) & MANT_MASK) < 65 {
        return as_tanh_database(x, res);
    }
    res
}

/// C `tanh(double)`, bit-identical to glibc 2.44 (CORE-MATH, correctly rounded) on every target.
/// Use instead of `f64::tanh`. errno and the exception flags are not set.
pub fn tanh(x: f64) -> f64 {
    cr_tanh(x)
}

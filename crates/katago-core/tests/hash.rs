//! core/hash bit-exactness against the real C++ (fixtures/upstream-v1.16.5/unit/oracle/hash.txt).
//! Mirrors `hashMode()` in oracle.cpp.

mod common;

use common::Cases;
use common::dbits;
use common::fixture;
use katago_core::core::global::Global;
use katago_core::core::hash::Hash;
use katago_core::core::hash::Hash128;

fn h(x: u64) -> String {
  Global::uint64_to_hex_string(x)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn hash_bit_exact_vs_cpp() {
  let mut cases = Cases::new("hash::hash_bit_exact_vs_cpp");
  let expected = fixture("oracle", "hash.txt");
  let mut out = String::new();
  let xs: [u64; 6] = [0, 1, 12345, 0x0123456789abcdef, 0xfefefefefefefefe, 0xFFFFFFFFFFFFFFFF];
  for x in xs {
    out += &format!(
      "x {} highBits {} lowBits {} basicLCong {} basicLCong2 {} murmurMix {} splitMix64 {} rrmxmx {} nasam {}\n",
      h(x),
      Hash::high_bits(x),
      Hash::low_bits(x),
      h(Hash::basic_l_cong(x)),
      h(Hash::basic_l_cong2(x)),
      h(Hash::murmur_mix(x)),
      h(Hash::split_mix64(x)),
      h(Hash::rrmxmx(x)),
      h(Hash::nasam(x))
    );
    cases.ran();
  }
  out += &format!("combine {}\n", h(Hash::combine(0xDEADBEEF, 0x12345678)));
  let triples: [[u32; 3]; 4] = [[0, 0, 0], [1, 2, 3], [0xFFFFFFFF, 0x80000000, 7], [123456789, 987654321, 555]];
  for t in triples {
    let (mut a, mut b, mut c) = (t[0], t[1], t[2]);
    out += &format!("jenkinsMixSingle {}", Hash::jenkins_mix_single(a, b, c));
    Hash::jenkins_mix(&mut a, &mut b, &mut c);
    out += &format!(" jenkinsMix {} {} {}\n", a, b, c);
    cases.ran();
  }
  let strs = ["", "a", "hello", "Board::initHash()", "\u{e9}t\u{e9} \u{20ac}", "The quick brown fox jumps over the lazy dog."];
  for s in strs {
    out += &format!("simpleHash \"{}\" {}\n", s, h(Hash::simple_hash(s)));
    cases.ran();
  }
  let ivs: Vec<Vec<i32>> = vec![vec![], vec![0], vec![1, 2, 3], vec![-1, -2, -3, 2147483647, -2147483648], vec![42, 0, -7, 100000, 3]];
  for v in &ivs {
    out += "simpleHashInts";
    for x in v {
      out += &format!(" {}", x);
    }
    out += &format!(" -> {}\n", h(Hash::simple_hash_input_len(v, v.len() as i32)));
    cases.ran();
  }
  let sh = [("", ""), ("abc", "seed"), ("writeTrainingData", "writeTrainingDataKeepFrac"), ("x", "\u{fc}")];
  for (a, b) in sh {
    out += &format!("seededHashFloat {}|{} {}\n", a, b, dbits(Hash::seeded_hash_float(a, b)));
    cases.ran();
  }
  let h1 = Hash128::new(0x0123456789abcdef, 0xfedcba9876543210);
  let h2 = Hash128::new(0x1111111111111111, 0x0000000000000001);
  let h3 = Hash128::new(0x0000000000000002, 0x0000000000000001);
  out += &format!("h1 {} toString {}\n", h1, h1.to_string());
  out += &format!("xor {} or {} and {}\n", h1 ^ h2, h1 | h2, h1 & h2);
  let bi = |v: bool| v as i32;
  out += &format!(
    "lt {}{}{}{} le {} gt {} ge {}\n",
    bi(h2 < h3),
    bi(h3 < h2),
    bi(h1 < h2),
    bi(h2 < h1),
    bi(h2 <= h2),
    bi(h2 > h3),
    bi(h3 >= h2)
  );
  let hs = [
    "FEDCBA98765432100123456789ABCDEF",
    "fedcba98765432100123456789abcdef",
    "00000000000000000000000000000000",
    "FEDCBA9876543210",
    "FEDCBA98765432100123456789ABCDEG",
    " EDCBA98765432100123456789ABCDEF",
  ];
  for s in hs {
    match Hash128::of_string(s) {
      Ok(hh) => out += &format!("ofString {} -> {}\n", s, hh),
      Err(e) if e.is_string_error() => out += &format!("ofString {} -> ERR {}\n", s, e.what()),
      Err(e) => panic!("{}", e),
    }
    cases.ran();
  }
  common::compare_lines("oracle/hash.txt", &out, &expected);
}

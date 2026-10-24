//! SGFMetadata (neuralnet/sgfmetadata.rs) against the upstream C++: replays
//! fixtures/upstream-v1.16.5/unit/oracle/sgfmetadata.txt (made by sgfmetadata.cpp next to it) line
//! by line: getProfile on every human-SL profile name and malformed ones (result or error text),
//! getHash for both players, fillMetadataRow (64-bit FNV-1a digests of the float bits, and full rows
//! for a subset) and operator== / operator!=. Bit-exact; runs natively and on wasm32 (`just
//! test-wasm`): the row goes through port::math's ln / powf / cos / sin.

mod common;

use std::collections::BTreeMap;

use common::Cases;
use common::fixture;
use katago_core::core::datetime::SimpleDate;
use katago_core::game::board::Player;
use katago_core::neuralnet::sgfmetadata::SGFMetadata;

fn hex64(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("hex in sgfmetadata.txt")
}

fn decode_name(s: &str) -> String {
  if s == "-" {
    return String::new();
  }
  let bytes: Vec<u8> = (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("name hex")).collect();
  String::from_utf8(bytes).expect("ascii name")
}

fn b(x: bool) -> i32 {
  x as i32
}

/// The M line of the oracle for metadata `id`.
fn format_m(id: i32, m: &SGFMetadata) -> String {
  format!(
    "M {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {:016X} {:016X} {} {} {} {} {} {}",
    id,
    b(m.initialized),
    m.inverse_b_rank,
    m.inverse_w_rank,
    b(m.b_is_unranked),
    b(m.w_is_unranked),
    b(m.b_rank_is_unknown),
    b(m.w_rank_is_unknown),
    b(m.b_is_human),
    b(m.w_is_human),
    b(m.game_is_unrated),
    b(m.game_ratedness_is_unknown),
    b(m.tc_is_unknown),
    b(m.tc_is_none),
    b(m.tc_is_absolute),
    b(m.tc_is_simple),
    b(m.tc_is_byo_yomi),
    b(m.tc_is_canadian),
    b(m.tc_is_fischer),
    m.main_time_seconds.to_bits(),
    m.period_time_seconds.to_bits(),
    m.byo_yomi_periods,
    m.canadian_moves,
    m.game_date.year,
    m.game_date.month,
    m.game_date.day,
    m.source
  )
}

fn parse_m(t: &[&str]) -> (i32, SGFMetadata) {
  assert_eq!(t.len(), 28, "{:?}", t);
  let i = |k: usize| -> i32 { t[k].parse().expect("int") };
  let bo = |k: usize| -> bool { i(k) != 0 };
  let mut m = SGFMetadata::default();
  m.initialized = bo(2);
  m.inverse_b_rank = i(3);
  m.inverse_w_rank = i(4);
  m.b_is_unranked = bo(5);
  m.w_is_unranked = bo(6);
  m.b_rank_is_unknown = bo(7);
  m.w_rank_is_unknown = bo(8);
  m.b_is_human = bo(9);
  m.w_is_human = bo(10);
  m.game_is_unrated = bo(11);
  m.game_ratedness_is_unknown = bo(12);
  m.tc_is_unknown = bo(13);
  m.tc_is_none = bo(14);
  m.tc_is_absolute = bo(15);
  m.tc_is_simple = bo(16);
  m.tc_is_byo_yomi = bo(17);
  m.tc_is_canadian = bo(18);
  m.tc_is_fischer = bo(19);
  m.main_time_seconds = f64::from_bits(hex64(t[20]));
  m.period_time_seconds = f64::from_bits(hex64(t[21]));
  m.byo_yomi_periods = i(22);
  m.canadian_moves = i(23);
  // PORT NOTE (test): the oracle's dates are valid, but some come from SimpleDate arithmetic (no ctor check).
  m.game_date = SimpleDate { year: i(24), month: i(25), day: i(26) };
  m.source = i(27);
  (i(1), m)
}

fn row_of(m: &SGFMetadata, pla: Player, board_area: i32) -> Vec<f32> {
  let mut row = vec![-7.0f32; SGFMetadata::METADATA_INPUT_NUM_CHANNELS as usize];
  SGFMetadata::fill_metadata_row(Some(m), &mut row, pla, board_area);
  row
}

fn fnv(row: &[f32]) -> u64 {
  let mut h: u64 = 0xCBF29CE484222325;
  for x in row {
    for byte in x.to_bits().to_le_bytes() {
      h ^= byte as u64;
      h = h.wrapping_mul(0x100000001B3);
    }
  }
  h
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sgfmetadata_vs_upstream() {
  let text = fixture("oracle", "sgfmetadata.txt");
  let mut cases = Cases::new("sgfmetadata_vs_upstream");
  let mut metas: BTreeMap<i32, SGFMetadata> = BTreeMap::new();
  let mut pending_profile: Option<(String, SGFMetadata)> = None;
  let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
  for line in text.lines() {
    let t: Vec<&str> = line.split(' ').collect();
    *counts.entry(t[0]).or_insert(0) += 1;
    match t[0] {
      "P" => {
        assert!(pending_profile.is_none(), "{}", line);
        let name = decode_name(t[1]);
        let got = SGFMetadata::get_profile(&name);
        match t[2] {
          "OK" => {
            let m = got.unwrap_or_else(|e| panic!("{}: getProfile({:?}) failed in Rust: {}", line, name, e));
            pending_profile = Some((name, m));
          }
          "ERR" => {
            let expected = line.splitn(4, ' ').nth(3).expect("error text");
            match got {
              Ok(m) => panic!("{}: getProfile({:?}) succeeded in Rust: {:?}", line, name, m),
              Err(e) => {
                assert!(e.is_string_error(), "{}", line);
                assert_eq!(e.to_string(), expected, "{}", line);
              }
            }
          }
          _ => panic!("{}", line),
        }
      }
      "M" => {
        let (id, parsed) = parse_m(&t);
        if let Some((name, m)) = pending_profile.take() {
          assert_eq!(format_m(id, &m), line, "getProfile({:?})", name);
          metas.insert(id, m);
        } else {
          assert_eq!(format_m(id, &parsed), line, "M line round trip");
          metas.insert(id, parsed);
        }
      }
      "H" => {
        let m = &metas[&t[1].parse::<i32>().unwrap()];
        let pla: Player = t[2].parse().unwrap();
        let h = m.get_hash(pla);
        assert_eq!(format!("H {} {} {:016X} {:016X}", t[1], t[2], h.hash0, h.hash1), line);
      }
      "R" => {
        let m = &metas[&t[1].parse::<i32>().unwrap()];
        let pla: Player = t[2].parse().unwrap();
        let area: i32 = t[3].parse().unwrap();
        let row = row_of(m, pla, area);
        assert_eq!(format!("R {} {} {} {:016X}", t[1], t[2], t[3], fnv(&row)), line, "row {:?}", row);
      }
      "F" => {
        let m = &metas[&t[1].parse::<i32>().unwrap()];
        let pla: Player = t[2].parse().unwrap();
        let area: i32 = t[3].parse().unwrap();
        let row = row_of(m, pla, area);
        let mut s = format!("F {} {} {}", t[1], t[2], t[3]);
        for x in &row {
          s += &format!(" {:08X}", x.to_bits());
        }
        assert_eq!(s, line);
      }
      "E" => {
        let a = &metas[&t[1].parse::<i32>().unwrap()];
        let bm = &metas[&t[2].parse::<i32>().unwrap()];
        assert_eq!(format!("E {} {} {} {}", t[1], t[2], b(a == bm), b(a != bm)), line);
      }
      _ => panic!("unknown line {}", line),
    }
    cases.ran();
  }
  assert!(pending_profile.is_none());
  for k in ["P", "M", "H", "R", "F", "E"] {
    assert!(counts.get(k).copied().unwrap_or(0) > 0, "no {} lines", k);
  }
  crate::report!("sgfmetadata_vs_upstream: line counts {:?}", counts);
}

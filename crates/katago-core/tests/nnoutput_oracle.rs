//! NNOutput (neuralnet/nninputs.rs) against the unmodified upstream NNOutput:
//! fixtures/upstream-v1.16.5/unit/oracle/nnoutput.cpp (which documents the format) prints NNOutputs
//! (o lines), their debugPrint text (incl. NaN, infinities and values whose (int)round(x*1000) is
//! out of int range) and the averaging constructor's results (avg + r lines: owner maps, the
//! legality-mismatch fallback, policyOptimismUsed averaging). This test parses the o lines, replays
//! every op through the port and must print the same bytes. Not a port of upstream code.

mod common;

use std::rc::Rc;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use katago_core::core::hash::Hash128;
use katago_core::game::board::Board;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;

fn hexf(s: &str) -> f32 {
  f32::from_bits(u32::from_str_radix(s, 16).expect("f32 hex"))
}

/// An `o` line -> NNOutput.
fn parse_out(line: &str) -> NNOutput {
  let t: Vec<&str> = line.split(' ').collect();
  assert_eq!(t[0], "o");
  let i = |k: usize| -> i32 { t[k].parse().unwrap() };
  let mut o = NNOutput::new();
  o.nn_x_len = i(1);
  o.nn_y_len = i(2);
  o.nn_hash = Hash128::new(u64::from_str_radix(t[3], 16).unwrap(), u64::from_str_radix(t[4], 16).unwrap());
  o.white_win_prob = hexf(t[5]);
  o.white_loss_prob = hexf(t[6]);
  o.white_no_result_prob = hexf(t[7]);
  o.white_score_mean = hexf(t[8]);
  o.white_score_mean_sq = hexf(t[9]);
  o.white_lead = hexf(t[10]);
  o.var_time_left = hexf(t[11]);
  o.shortterm_winloss_error = hexf(t[12]);
  o.shortterm_score_error = hexf(t[13]);
  o.policy_optimism_used = hexf(t[14]);
  let n = i(15) as usize;
  assert_eq!(n as i32, o.nn_x_len * o.nn_y_len + 1);
  o.policy_probs.fill(-1.0);
  for k in 0..n {
    o.policy_probs[k] = hexf(t[16 + k]);
  }
  let mut k = 16 + n;
  if t[k] == "1" {
    k += 1;
    let m = (o.nn_x_len * o.nn_y_len) as usize;
    o.white_owner_map = Some((0..m).map(|j| hexf(t[k + j])).collect());
    k += m;
  } else {
    assert_eq!(t[k], "0");
    k += 1;
  }
  assert_eq!(k, t.len(), "trailing tokens in {:?}", line);
  o
}

/// printOut of the oracle.
fn out_line(tag: &str, o: &NNOutput, with_noised: bool) -> String {
  let mut s = format!("{} {} {} {:016X} {:016X}", tag, o.nn_x_len, o.nn_y_len, o.nn_hash.hash0, o.nn_hash.hash1);
  for v in [
    o.white_win_prob,
    o.white_loss_prob,
    o.white_no_result_prob,
    o.white_score_mean,
    o.white_score_mean_sq,
    o.white_lead,
    o.var_time_left,
    o.shortterm_winloss_error,
    o.shortterm_score_error,
    o.policy_optimism_used,
  ] {
    s += &format!(" {:08X}", v.to_bits());
  }
  let n = (o.nn_x_len * o.nn_y_len + 1) as usize;
  s += &format!(" {}", n);
  for k in 0..n {
    s += &format!(" {:08X}", o.policy_probs[k].to_bits());
  }
  for k in n..NNPos::MAX_NN_POLICY_SIZE as usize {
    assert_eq!(o.policy_probs[k].to_bits(), (-1.0f32).to_bits(), "policy beyond the board");
  }
  match &o.white_owner_map {
    Some(m) => {
      s += " 1";
      for v in &m[..(o.nn_x_len * o.nn_y_len) as usize] {
        s += &format!(" {:08X}", v.to_bits());
      }
    }
    None => s += " 0",
  }
  if with_noised {
    s += if o.noised_policy_probs.is_some() { " 1" } else { " 0" };
  }
  s
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nnoutput_vs_cpp() {
  init();
  run();
}

#[inline(never)]
fn run() {
  let expected = fixture("oracle", "nnoutput.txt");
  let mut debug_cases = Cases::new("nnoutput_vs_cpp: debugPrint");
  let mut avg_cases = Cases::new("nnoutput_vs_cpp: averaging constructor");
  let mut out = String::new();
  let mut pending: Vec<Rc<NNOutput>> = Vec::new();
  for line in expected.lines() {
    let op = line.split(' ').next().unwrap();
    match op {
      "o" => {
        let o = parse_out(line);
        out += &out_line("o", &o, false);
        out.push('\n');
        pending.push(Rc::new(o));
      }
      "debugPrint" => {
        out += line;
        out.push('\n');
        let t: Vec<&str> = line.split(' ').collect();
        let board = Box::new(Board::new_x_y(t[1].parse().unwrap(), t[2].parse().unwrap()).unwrap());
        let o = pending.pop().expect("an o line before debugPrint");
        assert!(pending.is_empty());
        let mut text = String::new();
        o.debug_print(&mut text, &board);
        for l in text.lines() {
          out.push('|');
          out += l;
          out.push('\n');
        }
        debug_cases.ran();
      }
      "avg" => {
        out += line;
        out.push('\n');
        let k: usize = line[4..].parse().unwrap();
        assert_eq!(k, pending.len());
        let avg = NNOutput::new_others(&pending);
        pending.clear();
        out += &out_line("r", &avg, true);
        out.push('\n');
        avg_cases.ran();
      }
      "|" | "r" => {}
      _ if op.starts_with('|') => {}
      other => panic!("unknown op {:?}", other),
    }
  }
  compare_lines("nnoutput.txt", &out, &expected);
}

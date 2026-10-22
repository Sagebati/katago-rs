//! Not a port of upstream code: checks that every extracted upstream expected block is consumed.
//! For each fixtures/upstream-v1.16.5/unit/<stem>/INDEX.tsv, every row with role "expected" must be
//! referenced by some test source in tests/ (other than this file), or be listed in ALLOWLIST with
//! a reason. A new block from `tools/extract_raw_blocks.py` that no test reads therefore fails CI,
//! and so does an allowlisted block once a test starts reading it (drop it from the list then).
//!
//! "Referenced" means one of:
//!   - the file name as a string literal: `"area-1.expected0.txt"`;
//!   - the slug as a string literal (`"area-1"`) in a file that also builds the name with
//!     `"{}.expected<k>.txt"` (the table-driven tests in testboardarea.rs / testboardbasic.rs).
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::collections::BTreeSet;

use common::Cases;
use common::fixture_dir;

/// Expected blocks that no test compares yet: (stem, file, reason). Tracked in port-manifest.toml.
const ALLOWLIST: [(&str, &str, &str); 4] = [
  (
    "testrules",
    "test-basic-game.expected0.txt",
    "'Test basic game' needs CompactSgf (cpp/dataio/sgf, an [[ignored]] entry in port-manifest.toml); stays an exception unless sgf is ported",
  ),
  ("testboardarea", "grouptaxscoring.expected0.txt", "groupTaxScoring needs NNInputs::fillScoring (M2): must be enabled in M2"),
  ("testboardarea", "grouptaxscoring2.expected0.txt", "groupTaxScoring2 needs NNInputs::fillScoring (M2): must be enabled in M2"),
  (
    "testboardarea",
    "grouptaxscoring2-but-with-unsafe-territories-on.expected0.txt",
    "groupTaxScoring2 but with unsafe territories on needs NNInputs::fillScoring (M2): must be enabled in M2",
  ),
];

fn test_sources() -> Vec<(String, String)> {
  let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
  let mut out = Vec::new();
  let mut stack = vec![dir];
  while let Some(d) = stack.pop() {
    for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {}: {}", d.display(), e)) {
      let p = entry.unwrap().path();
      if p.is_dir() {
        stack.push(p);
      } else if p.extension().is_some_and(|x| x == "rs") && p.file_name().is_some_and(|n| n != "fixture_coverage.rs") {
        let text = std::fs::read_to_string(&p).unwrap();
        out.push((p.display().to_string(), text));
      }
    }
  }
  out.sort();
  out
}

/// `<slug>.expected<k>.txt` -> (slug, k)
fn split_expected(file: &str) -> (&str, &str) {
  let stem = file.strip_suffix(".txt").unwrap_or_else(|| panic!("bad fixture name {}", file));
  let i = stem.rfind(".expected").unwrap_or_else(|| panic!("bad expected fixture name {}", file));
  (&stem[..i], &stem[i + ".expected".len()..])
}

fn is_referenced(sources: &[(String, String)], file: &str) -> bool {
  let (slug, k) = split_expected(file);
  let literal = format!("\"{}\"", file);
  let slug_literal = format!("\"{}\"", slug);
  let pattern = format!("\"{{}}.expected{}.txt\"", k);
  sources.iter().any(|(_, text)| text.contains(&literal) || (text.contains(&slug_literal) && text.contains(&pattern)))
}

#[test]
fn every_expected_block_is_consumed() {
  let mut cases = Cases::new("fixture_coverage::every_expected_block_is_consumed");
  let sources = test_sources();
  assert!(!sources.is_empty(), "no test sources found");

  let mut stems: Vec<String> = Vec::new();
  for entry in std::fs::read_dir(fixture_dir()).unwrap() {
    let p = entry.unwrap().path();
    if p.join("INDEX.tsv").is_file() {
      stems.push(p.file_name().unwrap().to_str().unwrap().to_string());
    }
  }
  stems.sort();
  assert!(stems.len() >= 4, "expected the rand/testboardarea/testboardbasic/testrules indexes, found {:?}", stems);

  let mut seen_allowlisted: BTreeSet<(&str, &str)> = BTreeSet::new();
  let mut missing: Vec<String> = Vec::new();
  let mut stale: Vec<String> = Vec::new();
  for stem in &stems {
    let index = std::fs::read_to_string(fixture_dir().join(stem).join("INDEX.tsv")).unwrap();
    let mut lines = index.lines();
    assert_eq!(lines.next(), Some("file\tline\ttest\trole"), "{}/INDEX.tsv header", stem);
    for line in lines {
      let f: Vec<&str> = line.split('\t').collect();
      assert_eq!(f.len(), 4, "{}/INDEX.tsv: bad row {:?}", stem, line);
      let (file, test, role) = (f[0], f[2], f[3]);
      assert!(fixture_dir().join(stem).join(file).is_file(), "{}/INDEX.tsv lists missing file {}", stem, file);
      if role != "expected" {
        continue;
      }
      cases.ran();
      let referenced = is_referenced(&sources, file);
      match ALLOWLIST.iter().find(|(s, n, _)| s == stem && *n == file) {
        Some((s, n, _)) => {
          seen_allowlisted.insert((s, n));
          if referenced {
            stale.push(format!("{}/{} ('{}') is read by a test now: remove it from ALLOWLIST", stem, file, test));
          }
        }
        None => {
          if !referenced {
            missing.push(format!("{}/{} ('{}')", stem, file, test));
          }
        }
      }
    }
  }
  for (s, n, _) in ALLOWLIST.iter() {
    assert!(seen_allowlisted.contains(&(*s, *n)), "ALLOWLIST entry {}/{} is not an expected block in any INDEX.tsv", s, n);
  }
  assert!(stale.is_empty(), "{}", stale.join("\n"));
  assert!(
    missing.is_empty(),
    "{} upstream expected blocks are not read by any test (add a test, or an ALLOWLIST entry with a reason):\n{}",
    missing.len(),
    missing.join("\n")
  );
}

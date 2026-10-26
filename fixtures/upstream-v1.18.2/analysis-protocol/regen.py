#!/usr/bin/env python3
"""Regenerates fixtures/upstream-v1.18.2/analysis-protocol/expected/<corpus>.jsonl: the analysis-protocol
corpora of corpus/*.json run through the reference KataGo (Eigen fp32 build of the submodule,
`just katago-eigen` -> build/eigen/katago). Consumed by crates/katago-cli/tests/analysis_protocol_corpus.rs.

Every case runs as its own process, exactly:

  cd upstream/KataGo/cpp
  ../../../build/eigen/katago analysis -config configs/analysis_example.cfg \
      -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
      -override-config "<COMMON>[, <case extra>]" [<case args>]  < <case lines joined by '\n', plus a final '\n'>

with COMMON below (deterministic: one analysis thread, one search thread, nnRandomize=false,
forDeterministicTesting). Each case runs REPS times (MULTI_REPS for multi-line cases, whose output order
races upstream); every distinct outcome is kept as a variant (the test accepts any of them).

Recorded per variant: stdout lines, the exit status (as Python's returncode: -6 = SIGABRT), and the
stderr lines that the test compares (KEEP_STDERR below; the rest is backend-specific log text).

Corpus files (inputs, frozen; lines are strings, {"hex": ...} for bytes that are not UTF-8, or {"sleep": s} to
pause the writer; "args" go after -override-config; "sort": compare sorted outputs):
  fields.json         626 cases: every request field, malformed values, raw parse errors (from gen/gen.py)
  bytes_cfg_args.json 144 cases: raw bytes / invalid UTF-8 / non-ASCII next to parse errors, config and log keys,
                      argv (TCLAP) (gen/gen2.py, plus bytes_nul_then_ff.., multibyte_*, argv_*, logcfg_*, bytes_ff_during_long_search added by hand)
  avoid_allow.json     10 cases: deep avoidMoves / allowMoves, two -override-config (hand-written)
  fuzz.json           400 cases: random field mixes (gen/fuzz.py 400 1)
  multiline.json      200 cases: 3-5 lines per session (queries, actions, errors; generated during the M3 wave-2b
                      review, generator not kept)
  order.json           18 cases: burst-order scenarios, terminate of popped / queued requests, clear_cache with
                      nnRandomize (gen/order_corpus.py)
The gen/*.py scripts of the review wrote Python str with surrogate escapes; they were converted to the format
above (surrogates -> "hex").

Usage: python3 -I regen.py [corpus-name ...]   (from anywhere; JOBS env, default 4)
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
CPP = ROOT + "/upstream/KataGo/cpp"
EIGEN = ROOT + "/build/eigen/katago"
CFG = "configs/analysis_example.cfg"
M6 = "tests/models/g170-b6c96-s175395328-d26788732.bin.gz"
COMMON = ("logDir=, logTimeStamp=false, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, "
          "nnRandSeed=analysisTest, forDeterministicTesting=true, maxVisits=8, rootSymmetryPruning=false")
REPS = 2
MULTI_REPS = 5
# stderr lines compared by the test (same regex there)
KEEP_STDERR = re.compile(r"^(terminate called|  what\(\)|: ERROR|ERROR|: WARNING|WARNING|: Warning|Warning|PARSE ERROR|             |Brief USAGE|For complete USAGE|   analysis|KataGo v|Started, ready|Error)")


def decode_line(x):
    return bytes.fromhex(x["hex"]) if isinstance(x, dict) else x.encode("utf-8")


def run(case):
    ov = COMMON + (", " + case["extra"] if case.get("extra") else "")
    # {"sleep": s} entries pause the writer (lines sent before it are flushed first)
    chunks, cur = [], b""
    for x in case["lines"]:
        if isinstance(x, dict) and "sleep" in x:
            chunks.append(cur); chunks.append(x["sleep"]); cur = b""
        else:
            cur += decode_line(x) + b"\n"
    chunks.append(cur)
    p = subprocess.Popen([EIGEN, "analysis", "-config", CFG, "-model", M6, "-override-config", ov, *case.get("args", [])], cwd=CPP,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def writer():
        for c in chunks:
            if isinstance(c, (int, float)):
                time.sleep(c)
            elif c:
                p.stdin.write(c); p.stdin.flush()
        p.stdin.close()
    got = {}
    readers = [threading.Thread(target=lambda k=k, f=f: got.__setitem__(k, f.read())) for k, f in (("out", p.stdout), ("err", p.stderr))]
    for t in [threading.Thread(target=writer)] + readers:
        t.start()
    for t in readers:
        t.join()
    p.wait(timeout=600)
    # split on '\n' only (str.splitlines also splits on U+0085, U+2028...)
    lines = lambda b: [l.decode("utf-8", "backslashreplace") for l in (b[:-1] if b.endswith(b"\n") else b).split(b"\n")] if b else []
    out = lines(got["out"])
    err = [l for l in lines(got["err"]) if KEEP_STDERR.match(l)]
    return {"stdout": out, "rc": p.returncode, "stderr": err}


def do_case(case):
    reps = MULTI_REPS if len(case["lines"]) > 1 else REPS
    variants = []
    for _ in range(reps):
        v = run(case)
        if v not in variants:
            variants.append(v)
    rec = dict(case)
    rec["variants"] = variants
    return rec


def main():
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(HERE + "/corpus") if f.endswith(".json"))
    jobs = int(os.environ.get("JOBS", "4"))
    os.makedirs(HERE + "/expected", exist_ok=True)
    for name in names:
        cases = json.load(open(HERE + "/corpus/" + name + ".json"))
        with cf.ThreadPoolExecutor(jobs) as ex:
            recs = list(ex.map(do_case, cases))
        with open(HERE + "/expected/" + name + ".jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        nvar = sum(len(r["variants"]) > 1 for r in recs)
        print("%s: %d cases, %d with more than one upstream outcome" % (name, len(recs), nvar), flush=True)


if __name__ == "__main__":
    main()

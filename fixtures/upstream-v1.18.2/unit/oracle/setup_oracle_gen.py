#!/usr/bin/env python3
"""Generates setup_oracle_cases.txt, the config cases run by BOTH oracle/setup_oracle.cpp (against upstream's
program/setup.cpp) and crates/katago-core/tests/setup_oracle.rs (against the port).
Case format: "CASE <n> <nn:0|1|2> <name>" (nn=1: also the initializeNNEvaluators ops, 2: only those), the config lines, "END".
Cases: every upstream config that is a plain key list (configs/**.cfg, tests/data/configs/*.cfg: 29 of the 34; the
other 5 use @include, circular includes or duplicate keys and are config-parser tests, not Setup input), each as is
and with N_MUT randomly mutated variants (keys deleted / replaced by boundary, invalid and random values), then
N_RANDOM fuzzed configs (parameter keys with boundary/invalid values, rules, board size, winrate side, and for
every fourth the NN keys of initializeNNEvaluators), then N_DENSE configs of NN keys (see dense_case). nnCacheSizePowerOfTwo is capped (<= 12) in the upstream-config
cases, and absent in only a few fuzzed ones (the setup_for defaults are 2^19..2^23 entries per evaluator).
Run from the repo root (the checked-in file is the oracle input; the output is deterministic for a given Python):
  python3 -I fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_gen.py upstream/KataGo/cpp \\
    fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_cases.txt 200 5 36
"""
import glob, os, random, re, sys

UP = sys.argv[1]  # upstream/KataGo/cpp
OUT = sys.argv[2]
N_RANDOM = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_MUT = int(sys.argv[4]) if len(sys.argv) > 4 else 5
rng = random.Random(12345)

src = open(os.path.join(UP, "program/setup.cpp")).read()
pat = re.compile(r'cfg\.get(Int64|Int|Double|Float|Bool|String|Enabled)\("([A-Za-z0-9_-]+)"(?:\s*,\s*([^,)]+(?:\([^)]*\))?[^,)]*)\s*,\s*([^)]+?))?\)')
keys = {}
for m in pat.finditer(src):
  t, k, a, b = m.groups()
  keys.setdefault(k, [t, a, b])

def num(s):
  s = s.strip().replace("(int64_t)", "").replace("f", "") if s else s
  return s

RANGES = {}
for k, (t, a, b) in keys.items():
  if t in ("Int", "Int64", "Double", "Float"):
    if t == "Int64":
      lo, hi = 1, 1 << 50
    else:
      a2 = num(a); b2 = num(b)
      sub = {"NNPos::MAX_BOARD_LEN": "19", "Board::MAX_LEN": "19", "SymmetryHelpers::NUM_SYMMETRIES-1": "7",
             "SymmetryHelpers::NUM_SYMMETRIES": "8", "MAX_BOT_PARAMS_FROM_CFG": "4096",
             "Rules::MIN_USER_KOMI": "-150", "Rules::MAX_USER_KOMI": "150"}
      a2 = sub.get(a2, a2); b2 = sub.get(b2, b2)
      lo, hi = float(a2), float(b2)
    RANGES[k] = (t, lo, hi)
  else:
    RANGES[k] = (t, None, None)

# Keys handled specially (NN cases only, or with restricted values to keep the C++ side cheap and safe)
NN_ONLY = {"debugSkipNeuralNet", "maxBoardXSizeForNNBuffer", "maxBoardSizeForNNBuffer", "maxBoardYSizeForNNBuffer",
           "requireMaxBoardSize", "inputsUseNHWC", "nnRandomize", "nnRandSeed", "numNNServerThreadsPerModel",
           "numEigenThreadsPerModel", "deviceToUse", "gpuToUse", "openclTunerFile", "openclReTunePerBoardSize",
           "useFP16", "useNHWC", "nnForcedSymmetry", "nnCacheSizePowerOfTwo", "nnMutexPoolSizePowerOfTwo",
           "nnMaxBatchSize", "homeDataDir"}
SKIP = {"graphSearchCatchUpProp", "autoAvoidRepeatDir"}

def fmt_double(x):
  r = rng.random()
  if r < 0.3: return repr(x)
  if r < 0.5: return "%.6g" % x
  if r < 0.6: return "%.3e" % x
  return "%.17g" % x

def value_for(k, t, lo, hi):
  r = rng.random()
  if t == "Bool":
    return rng.choice(["true", "false", "true", "false", "1", "0", "yes", "no", "t", "f", "True", "FALSE", "bogus"])
  if t == "Enabled":
    return rng.choice(["auto", "true", "false", "AUTO", "1", "0", "bogus"])
  if t == "String":
    return "s%d" % rng.randint(0, 99)
  if t in ("Int", "Int64"):
    lo = int(lo); hi = int(hi)
    if r < 0.12: return str(lo)
    if r < 0.24: return str(hi)
    if r < 0.30: return str(lo - 1)
    if r < 0.36: return str(hi + 1)
    if r < 0.40: return rng.choice(["abc", "1.5", " 7", "0x10", "+3", "1e2", "99999999999999999999"])
    span = min(hi - lo, 1000)
    return str(rng.randint(lo, lo + span))
  # Double / Float
  if r < 0.10: return fmt_double(lo)
  if r < 0.20: return fmt_double(hi)
  if r < 0.25: return fmt_double(lo - abs(lo) * 0.01 - 1e-3)
  if r < 0.30: return fmt_double(hi + abs(hi) * 0.01 + 1e-3)
  if r < 0.35: return rng.choice(["abc", "nan", "inf", "-inf", "1e400", "0x1p-3", "1,5", "5.", ".5", "-0"])
  return fmt_double(rng.uniform(lo, hi))

PARAM_KEYS = [k for k in RANGES if k not in NN_ONLY and k not in SKIP and k not in
              ("rules", "koRule", "scoringRule", "taxRule", "multiStoneSuicideLegal", "hasButton", "whiteBonusPerHandicapStone",
               "friendlyPassOk", "whiteHandicapBonus", "komi", "reportAnalysisWinratesAs", "defaultBoardXSize", "defaultBoardSize",
               "defaultBoardYSize", "numBots", "humanSLProfile", "playoutDoublingAdvantagePla")]

# "burn" (the port's own backend) is left out: its keys are marked used by the port only.
PREFIXES = ["dummybackend", "opencl", "cuda", "eigen", "trt", "metal"]

def rules_kvs(kv):
  r = rng.random()
  if r < 0.35:
    kv["rules"] = rng.choice(["tromp-taylor", "japanese", "chinese", "aga", "korean", "new-zealand", "stone-scoring",
                              "bogus", "koSIMPLEscoreAREAtaxNONEsui1button1", "chinese-ogs", "Japanese"])
    if rng.random() < 0.15:
      kv[rng.choice(["koRule", "scoringRule", "multiStoneSuicideLegal", "hasButton", "taxRule", "whiteHandicapBonus", "friendlyPassOk", "whiteBonusPerHandicapStone"])] = "true"
  elif r < 0.8:
    if rng.random() < 0.95: kv["koRule"] = rng.choice(["SIMPLE", "POSITIONAL", "SITUATIONAL", "SPIGHT", "simple", "bogus"])
    if rng.random() < 0.95: kv["scoringRule"] = rng.choice(["AREA", "TERRITORY", "area", "bogus"])
    if rng.random() < 0.95: kv["multiStoneSuicideLegal"] = rng.choice(["true", "false", "bogus"])
    if rng.random() < 0.4: kv["hasButton"] = rng.choice(["true", "false"])
    if rng.random() < 0.4: kv["taxRule"] = rng.choice(["NONE", "SEKI", "ALL", "bogus"])
    if rng.random() < 0.3: kv["whiteHandicapBonus"] = rng.choice(["0", "N", "N-1", "bogus"])
    if rng.random() < 0.3: kv["whiteBonusPerHandicapStone"] = rng.choice(["0", "1", "2", "x"])
    if rng.random() < 0.3: kv["friendlyPassOk"] = rng.choice(["true", "false", "x"])
  if rng.random() < 0.5:
    kv["komi"] = rng.choice(["7.5", "6.5", "0", "-150", "150", "150.5", "7.25", "x", "1e3", "-0.5", "3.0"])
  if rng.random() < 0.3:
    kv["reportAnalysisWinratesAs"] = rng.choice(["BLACK", "white", "b", "W", "SIDETOMOVE", "sideToMove", "x"])
  if rng.random() < 0.3:
    for k in rng.sample(["defaultBoardXSize", "defaultBoardSize", "defaultBoardYSize"], rng.randint(1, 3)):
      kv[k] = rng.choice(["9", "19", "2", "1", "20", "13", "x"])

ABSENT_CACHE_BUDGET = [3]

def nn_kvs(kv):
  # Keys of initializeNNEvaluators, with values that keep the C++ side cheap (small caches, few threads).
  # Without the key the cache gets the setup_for default (2^19 .. 2^23 entries): only a few cases, they are big.
  if ABSENT_CACHE_BUDGET[0] > 0 and rng.random() < 0.1:
    ABSENT_CACHE_BUDGET[0] -= 1
  else:
    kv["nnCacheSizePowerOfTwo"] = rng.choice(["-1", "0", "3", "10", "-2", "49", "x"])
  if rng.random() < 0.5:
    kv["nnMutexPoolSizePowerOfTwo"] = rng.choice(["-1", "0", "2", "12", "25", "-2", "x"])
  if rng.random() < 0.3: kv["debugSkipNeuralNet"] = "true"
  for base in ["maxBoardXSizeForNNBuffer", "maxBoardYSizeForNNBuffer", "maxBoardSizeForNNBuffer", "requireMaxBoardSize", "nnRandSeed"]:
    for suf in ["", "0", "1"]:
      if rng.random() < 0.15:
        t = RANGES[base][0]
        kv[base + suf] = value_for(base, t, 2, 19) if t == "Int" else (rng.choice(["true", "false", "x"]) if t == "Bool" else "seed%d" % rng.randint(0, 9))
  if rng.random() < 0.3: kv["nnRandomize"] = rng.choice(["true", "false", "x"])
  if rng.random() < 0.4: kv["numNNServerThreadsPerModel"] = rng.choice(["1", "2", "3", "0", "1025", "x"])
  if rng.random() < 0.2: kv["numEigenThreadsPerModel"] = rng.choice(["1", "2", "x"])
  if rng.random() < 0.6: kv["nnMaxBatchSize"] = rng.choice(["1", "8", "64", "0", "65537", "x"])
  if rng.random() < 0.3: kv["nnForcedSymmetry"] = rng.choice(["0", "3", "7", "8", "-1", "x"])
  if rng.random() < 0.2: kv["homeDataDir"] = "h%d" % rng.randint(0, 9)
  if rng.random() < 0.2: kv["openclTunerFile"] = "t%d" % rng.randint(0, 9)
  if rng.random() < 0.2: kv["openclReTunePerBoardSize"] = rng.choice(["true", "false", "x"])
  for _ in range(rng.randint(0, 6)):
    p = rng.choice(PREFIXES + ["", ""])
    which = rng.choice(["InputsUseNHWC", "UseFP16", "UseNHWC", "DeviceToUse", "GpuToUse"])
    if p == "":
      which = which[0].lower() + which[1:]
    idx = rng.choice(["", "0", "1"])
    if which.endswith("UseFP16") and idx != "": idx = "-" + idx
    if which.endswith("ToUse"):
      form = rng.choice(["", "Model" + rng.choice("01"), "Model" + rng.choice("01") + "Thread" + rng.choice("012"), "Thread" + rng.choice("012")])
      kv[p + which + form] = rng.choice(["0", "1", "5", "1023", "1024", "-1", "x"])
    elif "UseNHWC" == which[-7:] and which.startswith(("inputs", "Inputs")) or which.endswith("InputsUseNHWC") or which == "inputsUseNHWC":
      kv[p + which + idx] = rng.choice(["true", "false", "x"])
    else:
      kv[p + which + idx] = rng.choice(["auto", "true", "false", "x"])

def random_case(nn):
  kv = {}
  nbots = None
  r = rng.random()
  if r < 0.25:
    nbots = rng.choice(["1", "2", "3", "0", "4097", "x"])
    kv["numBots"] = nbots
  if rng.random() < 0.92:
    kv["numSearchThreads"] = rng.choice(["1", "2", "4096", "8", "1", "2", "4097", "0", "x"])
  idxs = ["", "", "", "0", "1", "2"]
  for _ in range(rng.randint(0, 25)):
    k = rng.choice(PARAM_KEYS)
    t, lo, hi = RANGES[k]
    kv[k + rng.choice(idxs)] = value_for(k, t, lo, hi)
  if rng.random() < 0.15:
    kv["playoutDoublingAdvantagePla" + rng.choice(idxs)] = rng.choice(["BLACK", "WHITE", "b", "w", "x", "EMPTY"])
  if rng.random() < 0.15:
    kv["humanSLProfile" + rng.choice(idxs)] = rng.choice(["_"])
  if rng.random() < 0.1:
    k = rng.choice([k for k in PARAM_KEYS if k.startswith("humanSL")])
    t, lo, hi = RANGES[k]
    kv[k + rng.choice(idxs)] = value_for(k, t, lo, hi)
  if rng.random() < 0.1:
    kv["fpuParentWeightByVisitedPolicy" + rng.choice(idxs)] = rng.choice(["true", "false"])
    kv["fpuParentWeight" + rng.choice(idxs)] = rng.choice(["0.5", "2", "x"])
  rules_kvs(kv)
  if nn:
    nn_kvs(kv)
  return kv

def read_cfg(path):
  kv = {}
  ok = True
  for line in open(path):
    line = line.split("#", 1)[0].strip()
    if not line: continue
    if line.startswith("@"):
      return None
    if "=" not in line: return None
    k, v = line.split("=", 1)
    k = k.strip(); v = v.strip()
    if k in kv: return None
    kv[k] = v
  return kv

cases = []
cfgs = sorted(glob.glob(os.path.join(UP, "configs/**/*.cfg"), recursive=True) + glob.glob(os.path.join(UP, "tests/data/configs/*.cfg")))
for p in cfgs:
  kv = read_cfg(p)
  if kv is None:
    continue
  name = os.path.relpath(p, UP)
  base = dict(kv)
  # keep NN-side cheap: real configs use caches of up to 2^24 entries; cap them, and the threads.
  base["nnCacheSizePowerOfTwo"] = str(min(12, int(base.get("nnCacheSizePowerOfTwo", "10").split()[0])))
  for k in list(base):
    if k.startswith("numNNServerThreadsPerModel") or k.startswith("numEigenThreadsPerModel"):
      base[k] = "1"
  if "debugSkipNeuralNet" in base: base["debugSkipNeuralNet"] = "true"
  cases.append((name, 1, base))
  for j in range(N_MUT):
    m = dict(kv)
    for k in list(m):
      if k.startswith("numNNServerThreadsPerModel") or k.startswith("numEigenThreadsPerModel"): m[k] = "1"
    for _ in range(rng.randint(1, 6)):
      if m and rng.random() < 0.4:
        del m[rng.choice(list(m))]
      else:
        k = rng.choice(PARAM_KEYS)
        t, lo, hi = RANGES[k]
        m[k + rng.choice(["", "", "0", "1"])] = value_for(k, t, lo, hi)
    m.setdefault("nnCacheSizePowerOfTwo", "4")
    if m["nnCacheSizePowerOfTwo"].lstrip("-").isdigit() and int(m["nnCacheSizePowerOfTwo"]) > 12: m["nnCacheSizePowerOfTwo"] = "12"
    if "debugSkipNeuralNet" in m: m["debugSkipNeuralNet"] = "true"
    cases.append((name + "#mut%d" % j, 1 if j < 3 else 0, m))

for i in range(N_RANDOM):
  nn = 1 if i % 4 == 0 else 0
  cases.append(("random%d" % i, nn, random_case(nn)))

# Precedence ladders of initializeNNEvaluators (device/gpu per model and thread, FP16, NHWC, board size, seed, ...):
# dense cases set a random part (6%..40%) of ALL the keys of a family (both prefixes the oracle builds, "cuda" for a foreign
# one, with and without model/thread suffixes), each with a different value, so that swapping two rungs of a ladder
# changes the result. N_DENSE of them, after the fuzzed ones (the cases above keep their numbers).
N_DENSE = int(sys.argv[5]) if len(sys.argv) > 5 else 36
drng = random.Random(777)
def dense_case():
  kv = {"numSearchThreads": "1", "numNNServerThreadsPerModel": "2", "nnCacheSizePowerOfTwo": "4", "nnMaxBatchSize": "8"}
  uniq = [0]
  density = drng.choice([0.06, 0.12, 0.2, 0.4])  # sparse ones reach the low rungs of a ladder
  def val(kind):
    uniq[0] += 1
    if kind == "dev": return str(uniq[0] % 1000)
    if kind == "size": return str(2 + uniq[0] % 18)
    if kind == "seed": return "seed%d" % uniq[0]
    if kind == "bool": return drng.choice(["true", "false"])
    return drng.choice(["auto", "true", "false"])
  def put(k, kind):
    if drng.random() < density: kv[k] = val(kind)
  for p in ["dummybackend", "opencl", "cuda", ""]:
    for dev in ["DeviceToUse", "GpuToUse"]:
      name = (p + dev) if p else (dev[0].lower() + dev[1:])
      for form in ["", "Model0", "Model1", "Thread0", "Thread1", "Model0Thread0", "Model0Thread1", "Model1Thread0", "Model1Thread1"]:
        put(name + form, "dev")
    for fam, kind in [("UseFP16", "en"), ("UseNHWC", "en"), ("InputsUseNHWC", "bool")]:
      name = (p + fam) if p else (fam[0].lower() + fam[1:])
      put(name, kind)
      for i in ["0", "1"]:
        put(name + ("-" + i if fam == "UseFP16" else i), kind)
  for base, kind in [("maxBoardXSizeForNNBuffer", "size"), ("maxBoardYSizeForNNBuffer", "size"), ("maxBoardSizeForNNBuffer", "size"),
                     ("requireMaxBoardSize", "bool"), ("nnRandSeed", "seed")]:
    for suf in ["", "0", "1"]:
      put(base + suf, kind)
  return kv
for i in range(N_DENSE):
  cases.append(("dense%d" % i, 2, dense_case()))

with open(OUT, "w") as f:
  for n, (name, nn, kv) in enumerate(cases):
    f.write("CASE %d %d %s\n" % (n, nn, name))
    for k, v in kv.items():
      f.write("%s = %s\n" % (k, v))
    f.write("END\n")
print("cases:", len(cases), "nn:", sum(c[1] for c in cases), "from cfg files:", len(cfgs), file=sys.stderr)

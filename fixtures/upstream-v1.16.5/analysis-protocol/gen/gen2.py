#!/usr/bin/env python3
import json, sys, random
cases = []
def add(name, lines, extra="", **kw):
    if isinstance(lines, str): lines = [lines]
    c = {"name": name, "lines": lines, "extra": extra}; c.update(kw); cases.append(c)
def q(**kw):
    d = {"id": "q", "moves": [], "rules": "tromp-taylor", "boardXSize": 9, "boardYSize": 9, "maxVisits": 3}
    d.update(kw); return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
# rules-not-supported warnings
for i, r in enumerate([{"ko": "SPIGHT"}, {"scoring": "TERRITORY", "hasButton": True}, "koSPIGHTscoreAREAtaxNONEsui1", "koSIMPLEscoreTERRITORYtaxNONEsui0button1",
                       {"ko": "SPIGHT", "scoring": "TERRITORY", "hasButton": True, "komi": 5}]):
    add("rules_unsup_%d" % i, [q(rules=r)])
    add("rules_unsup_unk_%d" % i, [q(rules=r, zz=1, overrideSettings={"qq": 1, "maxVisits": 3})])
# raw bytes: NUL, invalid UTF-8 (surrogateescape -> raw bytes), lone CR
add("bytes_nul_in_string", ['{"id":"a\x00b","action":"query_version"}'])
add("bytes_nul_outside", ['{"id":"a"}\x00'])
add("bytes_nul_only", ['\x00'])
add("bytes_invalid_utf8_in_string", ['{"id":"a\udcffb","action":"query_version"}', '{"id":"after","action":"query_version"}'])
add("bytes_invalid_utf8_outside", ['\udcff{"id":"a","action":"query_version"}', '{"id":"after","action":"query_version"}'])
add("bytes_truncated_utf8", ['{"id":"a\udce2\udc82","action":"query_version"}', '{"id":"after","action":"query_version"}'])
add("bytes_overlong_utf8", ['{"id":"\udcc0\udcaf","action":"query_version"}', '{"id":"after","action":"query_version"}'])
add("bytes_cesu_surrogate", ['{"id":"\udced\udca0\udc80","action":"query_version"}', '{"id":"after","action":"query_version"}'])
add("bytes_lone_cr", ['\r', '{"id":"after","action":"query_version"}'])
add("bytes_del_char", ['{"id":"a\x7fb","action":"query_version"}'])
add("bytes_c1_char", ['{"id":"a\u0085b","action":"query_version"}'])
add("bytes_nbsp_lead", [' {"id":"a","action":"query_version"}'])
add("bytes_4byte", ['{"id":"\U0001F600\U0010FFFF","action":"query_version"}'])
# parse-error last-read with multibyte chars
add("parse_err_multibyte", ['{"id":é}', '{"id":"a",é:1}', '{"id":"a"}é', 'é', '{"é"}', '\U0001F600'])
add("parse_err_long_token", ['{"id":"' + 'x' * 5000 + '",}'])
add("parse_err_newline_pos", ['{"id":"a",\n"b"}'])  # newline splits lines in stdin anyway
add("deep_obj", ['{"a":' * 500 + '1' + '}' * 500])
add("deep_ok_query", [q(extra=json.loads('[' * 200 + ']' * 200))])
# overrideSettings dumps
for i, o in enumerate([{"maxVisits": 1e1}, {"maxVisits": 1.0e0}, {"cpuctExploration": 1e-7, "maxVisits": 3}, {"cpuctExploration": -0.0, "maxVisits": 3},
                       {"maxVisits": 18446744073709551615}, {"maxVisits": -9223372036854775809}, {"maxVisits": "  3 "}, {"maxVisits": "3\t"},
                       {"maxVisits": "0x3"}, {"maxVisits": "+3"}, {"maxVisits": "3.5"}, {"nnRandomize": "TRUE", "maxVisits": 3}, {"nnRandomize": "yes", "maxVisits": 3},
                       {"nnRandomize": "t", "maxVisits": 3}, {"nnRandomize": 1, "maxVisits": 3}, {"cpuctExploration": "inf", "maxVisits": 3},
                       {"cpuctExploration": "nan", "maxVisits": 3}, {"cpuctExploration": "1e400", "maxVisits": 3}, {"cpuctExploration": ".5", "maxVisits": 3},
                       {"maxVisits": 3, "a=b": 1}, {"maxVisits": 3, "x,y": 1}, {"maxVisits": 3, "k": "v,w=z"}, {"maxVisits": 3, "#c": 1},
                       {"maxVisits": 3, "numVirtualLossesPerThread": 2}, {"maxVisits": 3, "nnPolicyTemperature": 2}, {"maxVisits": 3, "rootNoiseEnabled": True},
                       {"maxVisits": 3, "humanSLProfile": ""}, {"maxVisits": 3, "@include": "x.cfg"}, {"maxVisits": 3, "fooBar": "x", "fooBaz": 2},
                       {"maxVisits": 3, "playoutDoublingAdvantagePla": "WHITE", "playoutDoublingAdvantage": 0.5}, {"maxVisits": 3, "dynamicPlayoutDoublingAdvantageCapPerOppLead": 0.1},
                       {"maxVisits": 3, "searchFactorWhenWinningThreshold": 0.9}, {"maxVisits": 3, "useNoisePruning": False}, {"maxVisits": 3, "subtreeValueBiasFactor": 0.5},
                       {"maxVisits": 3, "useEvalCache": True}, {"maxVisits": 3, "enableMorePassingHacks": False}, {"maxVisits": 3, "antiMirror": True}]):
    add("ovr2_%02d" % i, [q(overrideSettings=o)])
# handicap / whb effects
hs = [["B", "C3"], ["B", "G7"], ["B", "C7"], ["B", "G3"]]
for i, (r, whb, ip) in enumerate([("chinese", None, None), ("chinese", "0", None), ("chinese", "N-1", None), ("japanese", None, None), ("aga", None, "W"), ("chinese", "N", "B")]):
    kw = {"initialStones": hs, "rules": r}
    if whb: kw["whiteHandicapBonus"] = whb
    if ip: kw["initialPlayer"] = ip
    add("hcap_%d" % i, [q(**kw, includeOwnership=True)])
# long game many analyzeTurns
random.seed(7)
def legal_game(n, size=19):
    cols = "ABCDEFGHJKLMNOPQRST"[:size]; seen = set(); mv = []
    while len(mv) < n:
        c = random.choice(cols) + str(random.randint(1, size))
        if c in seen: continue
        seen.add(c); mv.append(["B" if len(mv) % 2 == 0 else "W", c])
    return mv
g = legal_game(40)
add("long_turns", [q(boardXSize=19, boardYSize=19, moves=g, analyzeTurns=list(range(0, 41, 5)), maxVisits=2)], sort=False)
add("long_turns_own", [q(boardXSize=19, boardYSize=19, moves=g, analyzeTurns=[10, 40], maxVisits=2, includeOwnership=True, includeMovesOwnership=True, includePolicy=True)])
add("long_turns_prios", [q(boardXSize=19, boardYSize=19, moves=g, analyzeTurns=[40, 0, 20], priorities=[1, 2, 3], maxVisits=2)])
# multi-thread (order nondeterministic -> sorted compare)
add("threads2_sorted", [q(id="a"), q(id="b"), q(id="c"), q(id="d", maxVisits=5)], extra="numAnalysisThreads=2", sort=True)
add("searchthreads2", [q(maxVisits=4)], extra="numSearchThreadsPerAnalysisThread=2", sort=True)
# config-level behaviour (stderr compared manually)
for i, ex in enumerate(["analysisPVLen=0", "analysisPVLen=101", "numAnalysisThreads=0", "numAnalysisThreads=x", "fooBarCfg=1", "reportAnalysisWinratesAs=x",
                        "logToStderr=false", "warnUnusedFields=x", "maxVisits=0", "numSearchThreads=2", "humanSLProfile=rank_5k",
                        "avoidSgfPatternUtility=0.1", "logFile=/nonexistent/dir/x.log", "logDir=/proc/nope", "logToStdout=true",
                        "preventCleanupPhase=x", "useEvalCache=true", "nnCacheSizePowerOfTwo=-1", "nnMaxBatchSize=0", "logConfigContents=false"]):
    add("cfg_%02d" % i, [q(), '{"id":"v","action":"query_version"}'], extra=ex, stderr=True)
add("args_threads_conflict", [q()], args=["-analysis-threads", "2"], stderr=True)
add("args_quit_without_waiting", [q(maxVisits=50), q(id="r", maxVisits=50)], args=["-quit-without-waiting"], stderr=True)
add("args_threads_ok", [q()], args=["-analysis-threads", "1"], extra="", stderr=True)
json.dump(cases, open(sys.argv[1], "w"), indent=0)
print(len(cases))

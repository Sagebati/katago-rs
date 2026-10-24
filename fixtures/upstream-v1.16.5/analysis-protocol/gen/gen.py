#!/usr/bin/env python3
import json, sys
cases = []
def add(name, lines, extra="", **kw):
    if isinstance(lines, str): lines = [lines]
    c = {"name": name, "lines": lines, "extra": extra}; c.update(kw); cases.append(c)
def q(**kw):
    d = {"id": "q", "moves": [], "rules": "tromp-taylor", "boardXSize": 9, "boardYSize": 9}
    for k, v in kw.items():
        if v is DEL: d.pop(k, None)
        else: d[k] = v
    return json.dumps(d, separators=(",", ":"))
class _D: pass
DEL = _D()

# --- A. raw parse errors (batched: each produces a single error line, deterministic order) ---
raw = [
 'not json', '{', '}', '{"id":}', '{"id":"a",}', '{"id":"a"} x', '{"id":"a"}{"id":"b"}', '[1,2', 'nul', 'tru', 'falsey',
 '"str"', '123', '-', '-0', '01', '1.', '.5', '1e', '1e+', '1E400', '-1E400', '1e-400', '[1e400]', '{"id":"a","x":1e999}',
 '{"id":"\\x"}', '{"id":"\\u12"}', '{"id":"\\ud800"}', '{"id":"\\udc00"}', '{"id":"\\ud800\\u0041"}', '{"id":"a\x01b"}',
 '{"id":"a\tb"}', '{\'id\':"a"}', '{"id":"a" "b":1}', '{"id" "a"}', '{"id":"a",,"b":1}', '[,]', '[1,]', '{"a":1 // c\n}',
 '/* c */ {"id":"a"}', 'NaN', 'Infinity', '{"id":NaN}', '0x10', '+1', '[' * 300 + ']' * 300, '{"id":"é"}', '"\\u00e9"',
 '{"id":"\\uD83D\\uDE00"}', '[[]]', '{}', '[]', 'null', 'true', '1.5', '"a\\"b"', '{"id":"a","id":"b","action":"query_version"}',
 '﻿{"id":"bom","action":"query_version"}', '{"id":"a b","action":"query_version"}',
 '{"id":"x","action":"query_version"} ', '   {"id":"ws","action":"query_version"}\t', '{"id":"cr","action":"query_version"}\r',
 '\x0b{"id":"vt","action":"query_version"}', '\x0c{"id":"ff","action":"query_version"}',
 '{"id":"a","action":"query_version","x":"\\u0000"}', '{"id":"a\\u0000b","action":"query_version"}',
]
for i, r in enumerate(raw): add("raw_%02d" % i, [r])
add("raw_empty_lines", ["", "   ", "\t", '{"id":"after-empty","action":"query_version"}'])

# --- B. id ---
for i, idv in enumerate(['{"action":"query_version"}', '{"id":1,"action":"query_version"}', '{"id":null}', '{"id":["a"]}',
                         '{"id":{"a":1}}', '{"id":true}', '{"id":""}', '{"id":"","action":"query_version"}']):
    add("id_%d" % i, [idv])

# --- C. actions & dump fidelity ---
acts = ['{"id":"v","action":"query_version"}', '{"id":"m","action":"query_models"}', '{"id":"c","action":"clear_cache"}',
 '{"id":"x","action":"Query_Version"}', '{"id":"x","action":""}', '{"id":"x","action":"terminate"}',
 '{"id":"x","action":"terminate","terminateId":1}', '{"id":"x","action":"terminate","terminateId":"zz"}',
 '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":[1,2]}', '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":[]}',
 '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":1}', '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":["1"]}',
 '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":[1.7,true,-3,4294967297]}',
 '{"id":"x","action":"terminate","terminateId":"zz","turnNumbers":null}',
 '{"id":"x","action":"terminate_all"}', '{"id":"x","action":"terminate_all","turnNumbers":[0]}', '{"id":"x","action":"terminate_all","turnNumbers":"a"}',
 '{"id":"x","action":"terminate_all","turnNumbers":[[0]]}', '{"id":"x","action":"terminate_all","turnNumbers":{"a":1}}',
 '{"id":"x","action":"terminate_all","terminateId":5,"extra":[1,{"b":2}]}',
 '{"id":"x","action":1}', '{"id":"x","action":null,"moves":[],"rules":"chinese","boardXSize":5,"boardYSize":5}',
 '{"id":"x","action":["query_version"]}',
]
for i, a in enumerate(acts): add("act_%02d" % i, [a])
dumpvals = ['0', '-0', '0.0', '-0.0', '1.0', '1.5', '0.1', '1e-7', '1E5', '1.5e300', '2.2250738585072014e-308', '5e-324', '4.9e-324',
 '1e308', '1.7976931348623157e308', '123456789012345678901234567890', '18446744073709551615', '18446744073709551616',
 '-9223372036854775808', '-9223372036854775809', '9223372036854775807', '9223372036854775808', '0.30000000000000004',
 '100000000000000000000.0', '1e21', '1e22', '123456789.123456789', '1e15', '1e16', '1e17', '3.14159265358979323846', '-1e-5',
 '"\\u0001\\u001f\\u007f"', '"\\/\\b\\f\\n\\r\\t"', '"é\\u00e9"', '"\\ud83d\\ude00"', '[]', '{}', '[null,true,false]', '{"b":1,"a":2,"A":3,"_":4,"aa":5,"é":6}',
 '"\\u2028"', '"<>&\'"', '0.1e1', '1.0e-0', '12e1']
for i, v in enumerate(dumpvals):
    add("dump_%02d" % i, ['{"id":"v","action":"query_version","x":%s}' % v])
add("dump_clear_cache_echo", ['{"id":"c","action":"clear_cache","z":[1.25,-0.0,1e100],"a":{"y":"s"}}'])
add("dump_query_models_overwrite", ['{"id":"m","action":"query_models","models":"x","version":3}'])
add("dump_query_version_overwrite", ['{"id":"v","action":"query_version","version":3,"git_hash":null}'])

# --- D. board sizes ---
for i, (x, y) in enumerate([(1, 9), (2, 2), (19, 19), (20, 9), (9, 20), ("9", 9), (9.0, 9), (9, None), (True, 9), (-9, 9), (0, 0),
                            (2**40, 9), (2**64 - 1, 9), (9, 2**31 + 9), (2, 19), (19, 2), (3, 7)]):
    add("size_%02d" % i, [q(boardXSize=x, boardYSize=y, maxVisits=3)])
add("size_missing_x", [q(boardXSize=DEL)]); add("size_missing_y", [q(boardYSize=DEL)]); add("size_missing_both", [q(boardXSize=DEL, boardYSize=DEL)])
add("size_bad_x_missing_y", [q(boardXSize="a", boardYSize=DEL)])
add("size_raw_float", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9.0,"boardYSize":9}'])
add("size_raw_1e1", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":1e1,"boardYSize":9}'])
add("size_raw_neg0", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":-0,"boardYSize":9}'])

# --- E. moves ---
mv = [None, "x", [], [[]], [["B"]], [["B", "C3", "x"]], [[1, "C3"]], [["B", 3]], [["B", "C3"]], [["b", "c3"]], [["black", "C3"]],
      [["White", "C3"]], [["W", "C3"]], [["x", "C3"]], [["", "C3"]], [["B", ""]], [["B", "pass"]], [["B", "PASS"]], [["B", "Pass"]],
      [["B", "I3"]], [["B", "J3"]], [["B", "J10"]], [["B", "K1"]], [["B", "A0"]], [["B", "A9"]], [["B", "A10"]], [["B", " C3"]],
      [["B", "C3 "]], [["B", "(2,2)"]], [["B", "c 3"]], [["B", "tt"]], [["B", "C3"], ["B", "D3"]], [["B", "C3"], ["W", "C3"]],
      [["B", "pass"], ["W", "pass"]], [["B", "pass"], ["W", "pass"], ["B", "C3"]], [["W", "C3"]], [{"B": "C3"}], [["B", None]],
      [["B", "C3"], ["W", "pass"], ["B", "pass"]], [["B", "Z99"]], [["B", "C-1"]], [["B", "C+3"]], [["B", "C03"]], [["B", "C3.0"]],
      [["B", "é3"]], [["B", "AA3"]], [["B", "null"]], [["B", "NULL"]], [["B", "resign"]]]
for i, m in enumerate(mv):
    add("moves_%02d" % i, [q(moves=m, maxVisits=3) if m is not None else q(moves=DEL)])
# captures / ko / suicide
ko = [["B", "B1"], ["W", "C1"], ["B", "A2"], ["W", "D2"], ["B", "B3"], ["W", "C3"], ["B", "C2"], ["W", "B2"], ["B", "C2"]]
add("moves_ko_recapture", [q(moves=ko, boardXSize=5, boardYSize=5, maxVisits=3)])
add("moves_ko_recapture_chinese", [q(moves=ko, boardXSize=5, boardYSize=5, rules="chinese", maxVisits=3)])
add("moves_ko_recapture_japanese", [q(moves=ko, boardXSize=5, boardYSize=5, rules="japanese", maxVisits=3)])
sui = [["W", "A2"], ["B", "C1"], ["W", "B2"], ["B", "C2"], ["W", "pass"], ["B", "pass"], ["W", "A1"], ["B", "B1"]]
add("moves_suicide_tt", [q(moves=[["B", "B1"], ["W", "A2"], ["B", "pass"], ["W", "B2"], ["B", "pass"], ["W", "C1"], ["B", "A1"]], boardXSize=5, boardYSize=5, maxVisits=3)])
add("moves_suicide_japanese", [q(moves=[["B", "B1"], ["W", "A2"], ["B", "pass"], ["W", "B2"], ["B", "pass"], ["W", "C1"], ["B", "A1"]], boardXSize=5, boardYSize=5, rules="japanese", maxVisits=3)])
add("moves_singlesuicide", [q(moves=[["W", "A2"], ["W", "B1"], ["B", "A1"]], boardXSize=5, boardYSize=5, maxVisits=3)])
add("moves_capture", [q(moves=[["B", "A2"], ["W", "A1"], ["B", "B1"]], boardXSize=5, boardYSize=5, maxVisits=3, includeOwnership=True)])
add("moves_long", [q(moves=[["B" if i % 2 == 0 else "W", "ABCDEFGHJ"[i % 9] + str(1 + i // 9)] for i in range(60)], maxVisits=3)])
add("moves_illegal_after_turns", [q(moves=[["B", "C3"], ["W", "D3"], ["B", "C3"]], analyzeTurns=[0, 1, 2, 3], maxVisits=2)])

# --- F. initialStones / initialPlayer ---
ist = [[], [["B", "C3"]], [["B", "C3"], ["B", "G7"]], [["B", "C3"], ["W", "C3"]], [["B", "pass"]], [["B", "C3"], ["B", "C3"]],
       [["B", "A1"], ["W", "A2"], ["W", "B1"]], "x", [["Q", "C3"]], [["W", "C3"], ["W", "D4"]], [["B", "Z9"]]]
for i, s in enumerate(ist): add("istones_%02d" % i, [q(initialStones=s, maxVisits=3)])
for i, p in enumerate(["B", "W", "b", "w", "black", "WHITE", "x", "", 1, None, ["B"], "empty", "E"]):
    add("iplayer_%02d" % i, [q(initialPlayer=p, maxVisits=3)])
add("iplayer_vs_moves", [q(initialPlayer="W", moves=[["B", "C3"]], maxVisits=3)])
add("handicap_default_white", [q(initialStones=[["B", "C3"], ["B", "G7"]], maxVisits=3)])
add("handicap_moves_bbb", [q(moves=[["B", "C3"], ["B", "G7"], ["B", "C7"]], rules="chinese", maxVisits=3)])
add("handicap_moves_bbb_whb0", [q(moves=[["B", "C3"], ["B", "G7"], ["B", "C7"]], rules="chinese", whiteHandicapBonus="0", maxVisits=3)])

# --- G. analyzeTurns ---
twom = [["B", "C3"], ["W", "G7"]]
for i, at in enumerate([[], [0], [0, 1, 2], [2, 1, 0], [0, 0], [-1], [3], [1.9], [True], ["1"], 1, None, [[0]], [4294967296], [4294967298],
                        [2147483648], [-4294967296], [1e10], [0.5, 1.5], {"0": 1}]):
    add("aturns_%02d" % i, [q(moves=twom, analyzeTurns=at, maxVisits=2)])

# --- H. priorities ---
for i, pr in enumerate([[1, 2, 3], [1, 2], [1.5, 2, 3], [True, 2, 3], ["1", 2, 3], 5, [2**62, 0, 0], [2**62 - 1, 0, 0], [-(2**62), 0, 0],
                        [-(2**62) + 1, 0, 0], [2**63, 0, 0], [2**64 - 1, 0, 0], [1e30, 0, 0], [-1e30, 0, 0], [None, 0, 0], []]):
    add("prios_%02d" % i, [q(moves=twom, analyzeTurns=[0, 1, 2], priorities=pr, maxVisits=2)])
add("prios_no_aturns", [q(moves=twom, priorities=[1], maxVisits=2)])
add("prios_bad_and_no_aturns", [q(moves=twom, priorities="x", maxVisits=2)])
add("prios_dup_turns", [q(moves=twom, analyzeTurns=[1, 1], priorities=[5, 6], maxVisits=2)])
add("prios_empty_both", [q(moves=twom, analyzeTurns=[], priorities=[], maxVisits=2)])
add("prios_and_priority", [q(moves=twom, analyzeTurns=[0], priorities=[1], priority=1, maxVisits=2)])
add("prios_bad_turn_first", [q(moves=twom, analyzeTurns=[9], priorities="x", maxVisits=2)])

# --- I. rules ---
rl = ["tromp-taylor", "TROMP-TAYLOR", " chinese ", "chinese", "japanese", "korean", "aga", "bga", "french", "nz", "new zealand",
      "stone-scoring", "ancient-territory", "aga-button", "goe", "ing", "chinese-ogs", "chinese_kgs", "", " ", "x",
      "koSIMPLEscoreAREAtaxNONEsui1komi7.5", "koPOSITIONALscoreTERRITORYtaxSEKIsui0", "koSITUATIONALscoreAREAtaxALLsui1button1",
      "koSIMPLEscoreAREAtaxNONEsui1whbNfpok1", "koSIMPLE", "scoreAREA komi 5", "komi7.25", "koBAD", "ko SIMPLE score AREA",
      5, None, True, [], ["chinese"],
      {}, {"ko": "SIMPLE"}, {"ko": "simple"}, {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "suicide": True, "hasButton": False, "whiteHandicapBonus": "N", "friendlyPassOk": True},
      {"score": "TERRITORY"}, {"scoring": "TERRITORY", "komi": 5.5}, {"komi": 6}, {"komi": 6.25}, {"komi": 151}, {"komi": "6"}, {"komi": True},
      {"komi": 1.5e300}, {"ko": 1}, {"suicide": 1}, {"foo": 1}, {"tax": "ALL"}, {"tax": "SEKI", "scoring": "AREA"}, {"hasButton": True},
      {"whiteHandicapBonus": "N-1"}, {"whiteHandicapBonus": "bad"}, {"friendlyPassOk": False, "komi": -0.0}, {"ko": "SITUATIONAL", "score": "AREA"},
      {"komi": 7.50, "ko": "SIMPLE", "a": [1.0, 2]}, "{\"ko\":\"SIMPLE\"}", "{bad", "{\"komi\":7}"]
for i, r in enumerate(rl): add("rules_%02d" % i, [q(rules=r, maxVisits=3)])
add("rules_missing", [q(rules=DEL)])

# --- J. komi ---
for i, k in enumerate([7, 7.5, 7.25, 150, 150.5, -150, -150.5, 0, -0.0, "7.5", None, True, 1e10, 0.5, 1e-9, 7.000001, 2**70, -7, 6.4999999999]):
    add("komi_%02d" % i, [q(komi=k, maxVisits=3)])
add("komi_raw_int_big", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"komi":18446744073709551615}'])
add("komi_raw_neg0", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"komi":-0,"maxVisits":2}'])

# --- K. whiteHandicapBonus ---
for i, w in enumerate(["0", "N", "N-1", "n", "n-1", "x", "", 0, None, " N"]):
    add("whb_%02d" % i, [q(whiteHandicapBonus=w, maxVisits=3)])

# --- L. overrideSettings ---
osv = [{}, "x", [], None, {"maxVisits": 3}, {"maxVisits": "3"}, {"maxVisits": 3.0}, {"maxVisits": 0}, {"maxVisits": True},
       {"cpuctExploration": 0.5, "maxVisits": 3}, {"cpuctExploration": "abc"}, {"fooBar": 1, "maxVisits": 3}, {"fooBar": 1, "zed": 2, "aaa": 3, "maxVisits": 3},
       {"numSearchThreads": 2}, {"numSearchThreads": 1, "maxVisits": 3}, {"nnCacheSizePowerOfTwo": 10, "maxVisits": 3}, {"reportAnalysisWinratesAs": "WHITE", "maxVisits": 3},
       {"reportAnalysisWinratesAs": "SIDETOMOVE", "maxVisits": 3}, {"reportAnalysisWinratesAs": "BLACK", "maxVisits": 3}, {"reportAnalysisWinratesAs": "x"},
       {"humanSLProfile": "rank_5k"}, {"humanSLChosenMoveProp": 0.5}, {"playoutDoublingAdvantage": 1.0, "maxVisits": 3}, {"wideRootNoise": 0.1, "maxVisits": 3},
       {"rootNumSymmetriesToSample": 8, "maxVisits": 3}, {"rootNumSymmetriesToSample": 9}, {"maxVisits": [3]}, {"maxVisits": {"a": 1}}, {"maxVisits": None},
       {"maxPlayouts": 2, "maxVisits": 100}, {"maxTime": 0.001, "maxVisits": 100000}, {"conservativePass": False, "maxVisits": 3}, {"useGraphSearch": False, "maxVisits": 3},
       {"rootSymmetryPruning": True, "maxVisits": 3}, {"nnRandomize": True, "maxVisits": 3}, {"numAnalysisThreads": 2, "maxVisits": 3}, {"logAllRequests": True, "maxVisits": 3},
       {"a b": 1}, {"": 1}, {"maxVisits": 3, "MAXVISITS": 4}, {"cpuctExploration": 1e400 if False else 1e300, "maxVisits": 3}, {"winLossUtilityFactor": -1, "maxVisits": 3},
       {"staticScoreUtilityFactor": 0.5, "dynamicScoreUtilityFactor": 0.0, "maxVisits": 3}, {"maxVisitsPondering": 3, "maxVisits": 3}, {"searchFactorAfterOnePass": 0.5, "maxVisits": 3},
       {"avoidRepeatedPatternUtility": 0.1, "maxVisits": 3}, {"avoidSgfPatternDirs": "/tmp"}, {"includeOwnership": True, "maxVisits": 3}, {"analysisPVLen": 3, "maxVisits": 3},
       {"maxVisits": 3, "nnMaxBatchSize": 3}, {"maxVisits": 3, "rules": "japanese"}, {"maxVisits": 3, "koRule": "SIMPLE"}, {"maxVisits": 3, "logSearchInfo": True},
       {"maxVisits": 3, "fpuReductionMax": "1e1"}, {"maxVisits": 3, "chosenMoveTemperature": 0.1}, {"maxVisits": 3, "useLcbForSelection": "false"},
       {"maxVisits": 3, "useLcbForSelection": "nope"}, {"maxVisits": 3, "cpuctExploration": "0.5x"}, {"maxVisits": 3, "cpuctExploration": " 0.5 "}]
for i, o in enumerate(osv): add("ovr_%02d" % i, [q(overrideSettings=o)] if o is not None or True else [])

# --- M..O numeric fields ---
for fld, vals in [("maxVisits", [1, 0, -1, "BIG", 2**50 + 1, 1.0, "1", True, None, 2**63, 2**64 - 1, 1e2]),
                  ("analysisPVLen", [0, 1, 1000, 1001, 2, 1.0, "3", None]),
                  ("rootFpuReductionMax", [0, 0.0, 2, 2.0000001, -0.0, -1e-9, "1", None, True, 1]),
                  ("rootPolicyTemperature", [0.01, 0.00999, 100, 100.1, 1, "1", None, 0.5]),
                  ("reportDuringSearchEvery", [0.001, 0.0009, 1000000, 1000001, "1", None, 0, -1, 1e30]),
                  ("firstReportDuringSearchAfter", [0.001, 0.0009, 1000000, 1000001, "1", None, 1e6]),
                  ("priority", [0, 1, -1, 2**62 - 1, 2**62, -(2**62) + 1, -(2**62), 1.0, "1", True, None, 2**64 - 1])]:
    for i, v in enumerate(vals):
        kw = {fld: v} if v != "BIG" else {fld: 2**50, "overrideSettings": {"maxPlayouts": 3}}
        if fld != "maxVisits": kw["maxVisits"] = 3
        add("num_%s_%02d" % (fld, i), [q(**kw)])
add("num_priority_raw_neg0", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"priority":-0,"maxVisits":2}'])
add("num_maxvisits_raw_neg0", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"maxVisits":-0}'])
add("num_maxvisits_raw_1e0", ['{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"maxVisits":1e0}'])

# --- P. include flags ---
flags = ["includeOwnership", "includeOwnershipStdev", "includeMovesOwnership", "includeMovesOwnershipStdev", "includePolicy", "includePVVisits", "includeNoResultValue"]
for f in flags:
    for i, v in enumerate([True, False, 1, "true", None, 0]):
        add("flag_%s_%d" % (f, i), [q(**{f: v, "maxVisits": 4, "boardXSize": 5, "boardYSize": 7})])
add("flag_all", [q(maxVisits=6, boardXSize=7, boardYSize=5, moves=[["B", "C3"]], **{f: True for f in flags})])
add("flag_all_white_persp", [q(maxVisits=6, boardXSize=6, boardYSize=4, moves=[["B", "C3"]], overrideSettings={"reportAnalysisWinratesAs": "WHITE"}, **{f: True for f in flags})])
add("flag_all_stm_persp", [q(maxVisits=6, boardXSize=6, boardYSize=4, moves=[["B", "C3"]], overrideSettings={"reportAnalysisWinratesAs": "SIDETOMOVE"}, **{f: True for f in flags})])
add("flag_own_stdev_only", [q(maxVisits=6, boardXSize=5, boardYSize=5, includeOwnershipStdev=True, includeMovesOwnershipStdev=True)])
add("flag_pvlen1_pvvisits", [q(maxVisits=10, analysisPVLen=1, includePVVisits=True)])
add("flag_19", [q(maxVisits=5, boardXSize=19, boardYSize=19, includePolicy=True, includeOwnership=True, moves=[["B", "Q16"], ["W", "D4"]])])

# --- S. allowMoves / avoidMoves ---
am = [("allowMoves", [{"player": "B", "moves": ["C3", "D4"], "untilDepth": 1}]), ("allowMoves", []),
      ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": 1}, {"player": "W", "moves": ["C3"], "untilDepth": 1}]),
      ("allowMoves", "x"), ("allowMoves", {"player": "B"}), ("allowMoves", [5]), ("allowMoves", [{"player": "B", "moves": ["C3"]}]),
      ("allowMoves", [{"player": "X", "moves": ["C3"], "untilDepth": 1}]), ("allowMoves", [{"player": "B", "moves": "C3", "untilDepth": 1}]),
      ("allowMoves", [{"player": "B", "moves": [3], "untilDepth": 1}]), ("allowMoves", [{"player": "B", "moves": ["Z9"], "untilDepth": 1}]),
      ("allowMoves", [{"player": "B", "moves": ["pass"], "untilDepth": 1}]), ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": 0}]),
      ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": 1000000000}]), ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": 1000000001}]),
      ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": 1.0}]), ("allowMoves", [{"player": "B", "moves": ["C3"], "untilDepth": "1"}]),
      ("allowMoves", [{"player": "W", "moves": ["C3"], "untilDepth": 1}]), ("allowMoves", [{"player": 1, "moves": ["C3"], "untilDepth": 1}]),
      ("avoidMoves", [{"player": "B", "moves": ["E5", "D5", "F5", "E4", "E6"], "untilDepth": 1}]), ("avoidMoves", []),
      ("avoidMoves", [{"player": "B", "moves": ["E5"], "untilDepth": 1}, {"player": "W", "moves": ["C3"], "untilDepth": 3}, {"player": "B", "moves": ["D5"], "untilDepth": 2}]),
      ("avoidMoves", [{"player": "B", "moves": [], "untilDepth": 1}]), ("avoidMoves", [{"player": "B", "moves": ["E5"], "untilDepth": -1}]),
      ("avoidMoves", [{"player": "B", "moves": ["E5"], "untilDepth": 1}, "x"]), ("avoidMoves", [{"player": "B", "moves": ["E5", "x"], "untilDepth": 1}]),
      ("avoidMoves", [{"moves": ["E5"], "untilDepth": 1, "player": "b", "extra": 1}]), ("avoidMoves", None), ("avoidMoves", [[]]),
      ("avoidMoves", [{"player": "B", "moves": ["E5"], "untilDepth": 2**40}])]
for i, (f, v) in enumerate(am): add("am_%02d" % i, [q(**{f: v, "maxVisits": 6})])
add("am_both", [q(allowMoves=[], avoidMoves=[], maxVisits=3)])
add("am_bad_player_bad_moves_order", [q(avoidMoves=[{"player": "X", "moves": "x", "untilDepth": "y"}], maxVisits=3)])
add("am_bad_moves_bad_depth", [q(avoidMoves=[{"player": "B", "moves": "x", "untilDepth": "y"}], maxVisits=3)])

# --- T. unknown fields ---
add("unk_1", [q(foo=1, maxVisits=3)]); add("unk_order", [q(zzz=1, aaa=2, Mmm=3, maxVisits=3)])
add("unk_with_error", [q(foo=1, moves=[["B", "C3"], ["W", "C3"]], maxVisits=3)])
add("unk_with_rules_warning", [q(foo=1, rules="stone-scoring", maxVisits=3)])
add("unk_with_override_warning", [q(foo=1, overrideSettings={"bar": 1, "maxVisits": 3})])
add("unk_disabled", [q(foo=1, maxVisits=3)], extra="warnUnusedFields=false")
add("unk_case", [q(MaxVisits=3, maxVisits=3)])
add("unk_with_badfield_before", [q(foo=1, komi="x")])

# --- misc ---
add("dup_keys_last_wins", ['{"id":"a","id":"b","moves":[],"rules":"x","rules":"chinese","boardXSize":5,"boardYSize":5,"maxVisits":2}'])
add("id_unicode", [q(id="éé😀".encode("utf-16", "surrogatepass").decode("utf-16"), boardXSize=DEL)])
add("id_escapes", ['{"id":"a\\"b\\\\c\\n\\u0001","boardXSize":5}'])
add("game_over_tt", [q(moves=[["B", "pass"], ["W", "pass"]], maxVisits=3)])
add("game_over_japanese", [q(moves=[["B", "pass"], ["W", "pass"]], rules="japanese", maxVisits=3)])
add("game_over_japanese_nopreventencore", [q(moves=[["B", "pass"], ["W", "pass"]], rules="japanese", maxVisits=3)], extra="preventCleanupPhase=false")
add("tiny_2x2", [q(boardXSize=2, boardYSize=2, maxVisits=5, includePolicy=True, includeOwnership=True)])
add("tiny_2x2_moves", [q(boardXSize=2, boardYSize=2, moves=[["B", "A1"], ["W", "B2"]], maxVisits=5)])
add("rect_19x2", [q(boardXSize=19, boardYSize=2, maxVisits=4, includePolicy=True)])
add("rect_2x19", [q(boardXSize=2, boardYSize=19, maxVisits=4, includeOwnership=True)])
add("multi_query_same_id", [q(maxVisits=2), q(maxVisits=3), q(id="r", maxVisits=2)])
add("pvlen_config", [q(maxVisits=10)], extra="analysisPVLen=2")
add("persp_white_config", [q(maxVisits=4, moves=[["B", "E5"]])], extra="reportAnalysisWinratesAs=WHITE")
add("persp_stm_config", [q(maxVisits=4, moves=[["B", "E5"]])], extra="reportAnalysisWinratesAs=SIDETOMOVE")
add("cfg_logerr_false", [q(foo=1, maxVisits=2), '{"x":1}'], extra="logErrorsAndWarnings=false")
add("cfg_conservativepass_false", [q(maxVisits=4, moves=[["B", "pass"]])], extra="conservativePass=false")
add("cfg_assume_handicap_false", [q(moves=[["B", "C3"], ["B", "G7"], ["B", "C7"]], rules="chinese", maxVisits=3)], extra="assumeMultipleStartingBlackMovesAreHandicap=false")
add("cfg_pvvisits_symprune", [q(maxVisits=12, includePVVisits=True)], extra="rootSymmetryPruning=true")

json.dump(cases, open(sys.argv[1], "w"), indent=0, ensure_ascii=False)
print(len(cases), "cases")

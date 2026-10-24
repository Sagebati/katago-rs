#!/usr/bin/env python3
import json, random, sys
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
POOL = [None, True, False, 0, 1, -1, 2, 3, 7, 19, 20, 1.5, 0.5, -0.0, 1e-3, 1e6, 2**62, 2**63, 2**64 - 1, -(2**63), "", "x", "B", "W", "pass",
        "C3", "chinese", "japanese", [], [0], [1, 2], [True], ["C3"], [["B", "C3"]], [["W", "pass"]], {}, {"maxVisits": 2}, {"x": 1},
        [{"player": "B", "moves": ["C3"], "untilDepth": 1}], [{"player": "W", "moves": ["D4", "pass"], "untilDepth": 2}], {"ko": "SIMPLE"}, "N-1", "0"]
FIELDS = ["id", "action", "terminateId", "turnNumbers", "boardXSize", "boardYSize", "initialStones", "moves", "initialPlayer", "analyzeTurns",
          "priorities", "rules", "komi", "whiteHandicapBonus", "overrideSettings", "maxVisits", "analysisPVLen", "rootFpuReductionMax",
          "rootPolicyTemperature", "includeMovesOwnership", "includeMovesOwnershipStdev", "includeOwnership", "includeOwnershipStdev",
          "includePolicy", "includePVVisits", "includeNoResultValue", "reportDuringSearchEvery", "firstReportDuringSearchAfter", "priority",
          "allowMoves", "avoidMoves", "zzUnknown"]
cases = []
for i in range(int(sys.argv[1])):
    d = {"id": "f%d" % i, "moves": [["B", "C3"], ["W", "D4"]], "rules": "tromp-taylor", "boardXSize": 7, "boardYSize": 7, "maxVisits": 3}
    for _ in range(random.randint(1, 3)):
        f = random.choice(FIELDS)
        if random.random() < 0.15: d.pop(f, None)
        else: d[f] = random.choice(POOL)
    if d.get("action") is not None and isinstance(d.get("action"), str) and random.random() < 0.5:
        d["action"] = random.choice(["terminate", "terminate_all", "query_version", "clear_cache"])
    # keep searches short
    if isinstance(d.get("maxVisits"), int) and not isinstance(d.get("maxVisits"), bool) and d["maxVisits"] > 20: d["maxVisits"] = 2**50 + 1 if d["maxVisits"] > 2**50 else 3
    if "reportDuringSearchEvery" in d or "firstReportDuringSearchAfter" in d: d.pop("reportDuringSearchEvery", None); d.pop("firstReportDuringSearchAfter", None)
    cases.append({"name": "fz_%03d" % i, "lines": [json.dumps(d, separators=(",", ":"))], "extra": ""})
json.dump(cases, open(sys.argv[3] if len(sys.argv) > 3 else "fuzz.json", "w"))

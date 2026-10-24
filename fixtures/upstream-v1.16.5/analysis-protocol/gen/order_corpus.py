#!/usr/bin/env python3
"""Writes ../corpus/order.json: burst scenarios whose output order races upstream (all lines written at
once): priorities vs an idle analysis thread, terminate of popped / queued requests, clear_cache and other
actions next to a short search. Usage: python3 -I order_corpus.py ../corpus/order.json"""
import json, sys
def q(id, **kw):
    d = {"id": id, "moves": [], "rules": "tromp-taylor", "boardXSize": 9, "boardYSize": 9}; d.update(kw)
    return json.dumps(d, separators=(",", ":"))
QA = q("a", moves=[["B","C3"],["W","G7"]], analyzeTurns=[0,1,2], maxVisits=5)
QB = q("b", maxVisits=5)
SC = {
 "prio_burst": [q("a", maxVisits=30), q("b", maxVisits=5, priority=10), q("c", maxVisits=5, priority=5), q("d", maxVisits=5)],
 "prio_neg_first": [q("a", maxVisits=5, priority=-5), q("b", maxVisits=5, priority=0)],
 "priorities_turns_after_busy": [q("z", maxVisits=60), q("a", moves=[["B","C3"],["W","G7"]], analyzeTurns=[0,1,2], priorities=[1,3,2], maxVisits=5)],
 "multiturn_term_t1": [QA, '{"id":"t","action":"terminate","terminateId":"a","turnNumbers":[1]}'],
 "multiturn_term_all": [QA, '{"id":"t","action":"terminate","terminateId":"a"}'],
 "multiturn_b_term_b": [QA, QB, '{"id":"t","action":"terminate","terminateId":"b"}'],
 "multiturn_termall_02": [QA, '{"id":"t","action":"terminate_all","turnNumbers":[0,2]}'],
 "multiturn_b_termall": [QA, QB, '{"id":"t","action":"terminate_all"}'],
 "multiturn_b_termall_10": [QA, QB, '{"id":"t","action":"terminate_all","turnNumbers":[1,0]}'],
 "term_queued_test": [q("a", analyzeTurns=[0], maxVisits=5), '{"id":"term","action":"terminate","terminateId":"a"}', q("c", priority=5, maxVisits=5)],
 "q3_7x7_then_cc": [q("m", moves=[["B","C3"],["W","D4"]], boardXSize=7, boardYSize=7, maxVisits=3), '{"id":"c","action":"clear_cache"}'],
 "q20_cc_q20": [q("a", maxVisits=20), '{"id":"cc","action":"clear_cache"}', q("b", maxVisits=20)],
 "q3_then_qm": [q("m", maxVisits=3), '{"id":"m","action":"query_models"}'],
 "q3_then_err": [q("m", maxVisits=3), '{"id":"e"}'],
 "q8_then_version": [q("m", maxVisits=8), '{"id":"v","action":"query_version"}'],
 # clear_cache observable (the sleeps leave room for a debug build's start-up): with nnRandomize the NN server draws a random symmetry per evaluation, so the
 # same position evaluated again after clear_cache differs from the cached first result (M37).
 "clear_cache_randomized": [q("a", maxVisits=8), {"sleep": 4.0}, '{"id":"cc","action":"clear_cache"}', {"sleep": 1.0}, q("b", maxVisits=8)],
 "no_clear_cache_randomized": [q("a", maxVisits=8), {"sleep": 4.0}, q("b", maxVisits=8)],
 "q_then_invalid_utf8": [q("m", maxVisits=8), {"hex": b'{"id":"a\xffb","action":"query_version"}'.hex()}, '{"id":"after","action":"query_version"}'],
}
RANDOMIZED = {"clear_cache_randomized", "no_clear_cache_randomized"}
json.dump([{"name": k, "lines": v, "extra": "nnRandomize=true, nnRandSeed=cacheTest" if k in RANDOMIZED else ""} for k, v in SC.items()], open(sys.argv[1], "w"), indent=0)

#!/usr/bin/env python3
# Copyright (c) 2026 katago-rs contributors. MIT License.
"""Verify a bin2onnx-converted model against the real KataGo binary.

For a set of positions x {japanese, chinese} rules, this script:
  1. drives `katago gtp` and collects `kata-raw-nn 0` (symmetry 0) output,
  2. builds the same NN input features with upstream KataGo's MIT python code
     (python/katago/game/{board,features,gamestate}.py), emulating the parts of
     cpp NNInputs::fillRowV7 the python code leaves to the caller (komi, pass
     handling / friendly-pass history suppression, passWouldEndPhase),
  3. runs the ONNX model in onnxruntime (all positions as one batch),
  4. applies NNEvaluator post-processing (cpp/neuralnet/nneval.cpp) and compares.

Human-SL nets (metaEncoderVersion > 0, a third input `meta_input`): every position is evaluated once
per --human-sl-profiles entry. KataGo runs with `humanSLProfile=<profile>` (the net is the main model,
so `kata-raw-nn` passes the profile's SGFMetadata), and the ONNX side gets the row of
SGFMetadata::fillMetadataRow, built with upstream's python/katago/game/sgfmetadata.py from a mirror of
SGFMetadata::getProfile (cpp/neuralnet/sgfmetadata.cpp). NNEvaluator zeroes policyOptimism for them.

Run with `python -I` (it adds the paths it needs itself):
  .venv/bin/python -I tools/bin2onnx/verify.py --model path/to/net.bin.gz --onnx models/onnx/net.onnx
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "upstream", "KataGo", "python"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from katago.game.board import Board  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402

import bin2onnx  # noqa: E402

POS_LEN = 19
GTP_COLS = "ABCDEFGHJKLMNOPQRST"

BENCH = "/home/sam/src/web/katago-ws/bench"
DEFAULT_KATAGO_CANDIDATES = [
    os.path.join(REPO, "build", "eigen", "katago"),
    os.path.join(BENCH, "katago-dist", "katago"),
]

RULES = {
    "japanese": GameState.RULES_JAPANESE,
    "chinese": GameState.RULES_CHINESE,
}

B, W = Board.BLACK, Board.WHITE


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def sgf_moves(path, max_moves=None):
    """Main-line moves of a non-handicap SGF as [(pla, (x,y) or None for pass)]."""
    from sgfmill import sgf
    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    if game.get_size() != 19:
        raise ValueError(f"{path}: not 19x19")
    root = game.get_root()
    for prop in ("AB", "AW", "AE"):
        if root.has_property(prop):
            raise ValueError(f"{path}: setup stones not supported")
    moves = []
    for node in game.get_main_sequence()[1:]:
        colour, mv = node.get_move()
        if colour is None:
            continue
        pla = B if colour == "b" else W
        if mv is None:
            moves.append((pla, None))
        else:
            row, col = mv
            moves.append((pla, (col, 18 - row)))
        if max_moves is not None and len(moves) >= max_moves:
            break
    return moves


def build_positions(sgf_dir):
    """Returns [(label, board_size, moves)]."""
    P = []
    P.append(("empty", 19, []))
    P.append(("one-move", 19, [(B, (15, 3))]))
    # Capture + simple ko: white stone at (4,4) captured by black (5,4); white may not retake at (4,4).
    P.append(("ko-capture", 19, [
        (B, (3, 4)), (W, (5, 3)), (B, (4, 3)), (W, (5, 5)), (B, (4, 5)), (W, (6, 4)),
        (B, (15, 15)), (W, (4, 4)), (B, (5, 4)),
    ]))
    # Opponent just passed (exercises pass history, passWouldEndPhase and
    # chinese friendly-pass history suppression).
    P.append(("after-pass", 19, [(B, (3, 3)), (W, (15, 15)), (B, (15, 3)), (W, None)]))
    sgfs = [
        ("ogs-60000000.sgf", 60),
        ("ogs-60001931.sgf", 120),
        ("ogs-60025103.sgf", None),
        ("ogs-60046344.sgf", 91),
        ("ogs-60054068.sgf", 150),
        ("ogs-60054068.sgf", None),
    ]
    for name, n in sgfs:
        path = os.path.join(sgf_dir, name)
        if not os.path.exists(path):
            print(f"warning: {path} missing, skipping", file=sys.stderr)
            continue
        mv = sgf_moves(path, n)
        P.append((f"{name[:-4]}@{len(mv)}", 19, mv))
    # Smaller boards inside the fixed 19x19 graph: exercises mask-aware pooling
    # (katago is run with gtpForceMaxNNSize=true so it also uses a 19x19 net input with a mask).
    P.append(("9x9-empty", 9, []))
    P.append(("9x9-ko", 9, [
        (B, (3, 4)), (W, (5, 3)), (B, (4, 3)), (W, (5, 5)), (B, (4, 5)), (W, (6, 4)),
        (B, (7, 7)), (W, (4, 4)), (B, (5, 4)),
    ]))
    P.append(("13x13-midgame", 13, [
        (B, (3, 3)), (W, (9, 9)), (B, (9, 3)), (W, (3, 9)), (B, (6, 6)), (W, (10, 5)),
        (B, (9, 5)), (W, (10, 4)), (B, (10, 3)), (W, (11, 3)), (B, (11, 2)), (W, (2, 8)),
    ]))
    return P


# ---------------------------------------------------------------------------
# SGF metadata (mirrors cpp SGFMetadata::getProfile; the row is upstream python's get_metadata_row)
# ---------------------------------------------------------------------------

RANK_STRS = [f"{d}d" for d in range(9, 0, -1)] + [f"{k}k" for k in range(1, 21)]


def sgf_metadata_profile(name):
    """SGFMetadata::getProfile for the profiles it knows (all fields set: python's defaults differ)."""
    import datetime

    def base(**kw):
        m = SGFMetadata(
            inverseBRank=0, inverseWRank=0, bIsUnranked=False, wIsUnranked=False, bRankIsUnknown=False,
            wRankIsUnknown=False, bIsHuman=False, wIsHuman=False, gameIsUnrated=False,
            gameRatednessIsUnknown=False, tcIsUnknown=False, tcIsNone=False, tcIsAbsolute=False,
            tcIsSimple=False, tcIsByoYomi=False, tcIsCanadian=False, tcIsFischer=False, mainTimeSeconds=0.0,
            periodTimeSeconds=0.0, byoYomiPeriods=0, canadianMoves=0, gameDate=datetime.date(1970, 1, 1), source=0)
        for k, v in kw.items():
            setattr(m, k, v)
        return m

    if name.startswith("proyear_"):
        year = int(name[len("proyear_"):])
        if 1800 <= year <= 2020:
            src = SGFMetadata.SOURCE_GOGOD
        elif 2021 <= year <= 2023:
            src = SGFMetadata.SOURCE_GO4GO
        else:
            raise ValueError(name)
        return base(inverseBRank=1, inverseWRank=1, bIsHuman=True, wIsHuman=True, tcIsUnknown=True,
                    gameDate=datetime.date(year, 6, 1), source=src)
    for prefix, pre_az in (("rank_", False), ("preaz_", True)):
        if name.startswith(prefix):
            ranks = name[len(prefix):]
            pieces = ranks.split("_") if ranks not in RANK_STRS else [ranks, ranks]
            if len(pieces) != 2 or any(p not in RANK_STRS for p in pieces):
                raise ValueError(name)
            inv_b, inv_w = (RANK_STRS.index(p) + 1 for p in pieces)
            return base(inverseBRank=inv_b, inverseWRank=inv_w, bIsHuman=True, wIsHuman=True,
                        gameRatednessIsUnknown=True, tcIsByoYomi=True, mainTimeSeconds=1200.0,
                        periodTimeSeconds=30.0, byoYomiPeriods=5,
                        gameDate=datetime.date(2016, 9, 1) if pre_az else datetime.date(2020, 3, 1),
                        source=SGFMetadata.SOURCE_KGS)
    raise ValueError(f"Unknown human SL network profile: {name}")


# ---------------------------------------------------------------------------
# Features (mirrors cpp NNInputs::fillRowV7 on top of upstream python features)
# ---------------------------------------------------------------------------

class Position:
    def __init__(self, label, board_size, moves, rules_name, komi):
        self.label = label
        self.size = board_size
        self.rules_name = rules_name
        self.komi = komi
        self.moves = moves
        rules = dict(RULES[rules_name])
        rules["whiteKomi"] = komi
        gs = GameState(board_size, rules)
        for pla, xy in moves:
            if gs.board.pla != pla:
                raise ValueError(f"{label}: non-alternating moves are not supported")
            loc = Board.PASS_LOC if xy is None else gs.board.loc(*xy)
            gs.play(pla, loc)
        self.gs = gs
        self.pla = gs.board.pla

    def legal_mask(self):
        board = self.gs.board
        rules = self.gs.rules
        legal = np.zeros(POS_LEN * POS_LEN + 1, dtype=bool)
        for y in range(self.size):
            for x in range(self.size):
                loc = board.loc(x, y)
                ok = board.would_be_legal(self.pla, loc)
                if ok and not rules["multiStoneSuicideLegal"] and board.would_be_suicide(self.pla, loc):
                    ok = False
                legal[y * POS_LEN + x] = ok
        legal[POS_LEN * POS_LEN] = True
        return legal

    def features(self, feats: Features):
        gs = self.gs
        rules = dict(gs.rules)
        pla = self.pla
        opp = Board.get_opp(pla)
        # Simple-ko only rules, main phase, no button: a pass ends the phase iff the last move was a pass
        # (BoardHistory::passWouldEndPhase via newConsecutiveEndingPassesAfterPass). Spight-like
        # repetition endings are not emulated (none of the test positions repeat a position before a pass).
        last_was_pass = len(gs.moves) > 0 and gs.moves[-1][1] == Board.PASS_LOC
        pass_would_end_phase = last_was_pass
        pass_would_end_game = pass_would_end_phase and rules["scoringRule"] == "SCORING_AREA"
        friendly_pass_ok = self.rules_name == "chinese"
        # conservativePassAndIsRoot=false and enablePassingHacks=false in kata-raw-nn, so only the
        # friendly-pass suppression applies.
        suppress = pass_would_end_game and friendly_pass_ok and rules["scoringRule"] == "SCORING_AREA"
        rules["passWouldEndPhase"] = False if suppress else pass_would_end_phase

        bin_input = np.zeros([1, POS_LEN * POS_LEN, feats.bin_input_shape[0]], dtype=np.float32)
        global_input = np.zeros([1] + feats.global_input_shape, dtype=np.float32)
        if suppress:
            # maxTurnsOfHistoryToInclude = 0: no move history, prev boards = current board
            boards, moves, move_idx = [gs.board], [], 0
        else:
            boards, moves, move_idx = gs.boards, gs.moves, len(gs.moves)
        feats.fill_row_features(gs.board, pla, opp, boards, moves, move_idx, rules,
                                bin_input, global_input, idx=0)
        bin_input = bin_input.reshape([1, POS_LEN, POS_LEN, -1]).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(bin_input), global_input

    def gtp_commands(self):
        cmds = [f"boardsize {self.size}", f"kata-set-rules {self.rules_name}", f"komi {self.komi}", "clear_board"]
        for pla, xy in self.moves:
            c = "B" if pla == B else "W"
            if xy is None:
                cmds.append(f"play {c} pass")
            else:
                x, y = xy
                cmds.append(f"play {c} {GTP_COLS[x]}{self.size - y}")
        return cmds


# ---------------------------------------------------------------------------
# GTP
# ---------------------------------------------------------------------------

class GTP:
    def __init__(self, katago, config, model, overrides):
        cmd = [katago, "gtp", "-config", config, "-model", model, "-override-config", overrides]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError(f"katago exited while running {cmd!r}")
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line == "" and not lines:
                continue
            lines.append(line)
        if not lines[0].startswith("="):
            raise RuntimeError(f"GTP error for {cmd!r}: {lines}")
        lines[0] = lines[0][1:].strip()
        return [l for l in lines if l != ""]

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def parse_raw_nn(lines, size):
    out = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        key = parts[0]
        if key == "policy":
            rows = [lines[i + 1 + y].split() for y in range(size)]
            pol = np.full((POS_LEN, POS_LEN), np.nan)
            pol[:size, :size] = [[float("nan") if v == "NAN" else float(v) for v in r] for r in rows]
            out["policy"] = pol.reshape(-1)
            i += 1 + size
            continue
        if key == "whiteOwnership":
            rows = [lines[i + 1 + y].split() for y in range(size)]
            own = np.zeros((POS_LEN, POS_LEN))
            own[:size, :size] = [[float(v) for v in r] for r in rows]
            out["ownership"] = own.reshape(-1)
            i += 1 + size
            continue
        if key == "policyPass":
            out["policyPass"] = float("nan") if parts[1] == "NAN" else float(parts[1])
        elif len(parts) == 2:
            out[key] = float(parts[1])
        i += 1
    return out


# ---------------------------------------------------------------------------
# NNEvaluator post-processing (cpp/neuralnet/nneval.cpp, model version >= 4)
# ---------------------------------------------------------------------------

def softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


def postprocess(desc, pp, pos: Position, policy, value, score, ownership, policy_optimism):
    mv = desc.model_version
    pla = pos.pla
    # policy (eigenbackend combines optimism channels; nneval masks illegal + softmax in float32)
    if policy.shape[0] >= 2 and (policy.shape[0] == 2 or mv >= 16):
        p, popt = policy[0], policy[1]
        logits = (p + (popt - p) * np.float32(policy_optimism)).astype(np.float32)
    else:
        logits = policy[0].astype(np.float32)
    logits = logits * np.float32(pp.output_scale_multiplier / 1.0)
    legal = pos.legal_mask()
    logits = np.where(legal, logits, np.float32(-1e30))
    e = np.exp(logits - logits.max()).astype(np.float32)
    probs = e / e.sum(dtype=np.float32)
    probs = np.where(legal, probs, np.nan)

    m = pp.output_scale_multiplier
    win, loss, nores = (float(v) * m for v in value)
    koRule, scoring = pos.gs.rules["koRule"], pos.gs.rules["scoringRule"]
    if koRule != "KO_SIMPLE" and scoring != "SCORING_TERRITORY":
        nores -= 100000.0
    mx = max(win, loss, nores)
    wp, lp, np_ = math.exp(win - mx), math.exp(loss - mx), math.exp(nores - mx)
    if koRule != "KO_SIMPLE" and scoring != "SCORING_TERRITORY":
        np_ = 0.0
    s = wp + lp + np_
    wp, lp, np_ = wp / s, lp / s, np_ / s

    sc = [float(v) * m for v in score]
    score_mean = sc[0] * pp.score_mean_multiplier
    stdev = softplus(sc[1]) * pp.score_stdev_multiplier
    score_mean_sq = score_mean * score_mean + stdev * stdev
    lead = sc[2] * pp.lead_multiplier if mv >= 8 else score_mean
    var_time_left = softplus(sc[3]) * pp.variance_time_multiplier if mv >= 8 else 0.0
    score_mean *= (1.0 - np_)
    score_mean_sq *= (1.0 - np_)
    lead *= (1.0 - np_)
    if mv >= 14:
        a = softplus(sc[4] * 0.5)
        st_wl = math.sqrt(a * a * pp.shortterm_value_error_multiplier)
        a = softplus(sc[5] * 0.5)
        st_sc = math.sqrt(a * a * pp.shortterm_score_error_multiplier)
    elif mv >= 10:
        st_wl = math.sqrt(softplus(sc[4]) * pp.shortterm_value_error_multiplier)
        st_sc = math.sqrt(softplus(sc[5]) * pp.shortterm_score_error_multiplier)
    elif mv >= 9:
        st_wl = softplus(sc[4])
        st_sc = softplus(sc[5]) * 10.0
    out = {}
    if pla == W:
        out.update(whiteWin=wp, whiteLoss=lp, whiteLead=lead, whiteScoreSelfplay=score_mean)
    else:
        out.update(whiteWin=lp, whiteLoss=wp, whiteLead=-lead, whiteScoreSelfplay=-score_mean)
    out["noResult"] = np_
    out["whiteScoreSelfplaySq"] = score_mean_sq
    if mv >= 9:
        out.update(varTimeLeft=var_time_left, shorttermWinlossError=st_wl, shorttermScoreError=st_sc)
    else:
        out.update(varTimeLeft=-1.0, shorttermWinlossError=-1.0, shorttermScoreError=-1.0)
    own = np.tanh(ownership.astype(np.float64) * m)
    onboard = np.zeros((POS_LEN, POS_LEN), dtype=bool)
    onboard[:pos.size, :pos.size] = True
    own = np.where(onboard, own, 0.0).reshape(-1)
    out["ownership"] = own if pla == W else -own
    out["policy"] = probs[:-1]
    out["policyPass"] = float(probs[-1])
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="KataGo model file (.bin.gz)")
    ap.add_argument("--onnx", required=True, help="Converted ONNX model")
    ap.add_argument("--katago", default=None, help="KataGo binary (default: build/eigen/katago, else bench dist)")
    ap.add_argument("--config", default=os.path.join(BENCH, "katago-dist", "default_gtp.cfg"))
    ap.add_argument("--sgf-dir", default=os.path.join(BENCH, "sgf-quick"))
    ap.add_argument("--policy-optimism", type=float, default=0.2,
                    help="rootPolicyOptimism passed to katago and applied to 2-channel policies")
    ap.add_argument("--home-data-dir", default=None,
                    help="katago homeDataDir (e.g. existing OpenCL tuning); default: a fresh temp dir")
    ap.add_argument("--extra-overrides", default="", help="extra -override-config entries (comma separated)")
    ap.add_argument("--tol-prob", type=float, default=1e-4)
    ap.add_argument("--tol-score", type=float, default=1e-3)
    ap.add_argument("--dump-features", default=None,
                    help="Save features/raw outputs to this .npz (human-SL nets: <name>.<profile>.npz per profile)")
    ap.add_argument("--human-sl-profiles", default="preaz_18k,rank_9d,rank_1k_15k,proyear_1850,proyear_2022",
                    help="comma-separated humanSLProfile values, for nets with metadata inputs")
    args = ap.parse_args()

    katago = args.katago
    if katago is None:
        katago = next((c for c in DEFAULT_KATAGO_CANDIDATES if os.path.exists(c)), None)
        if katago is None:
            sys.exit("No katago binary found; pass --katago")

    desc = bin2onnx.load_model_desc(args.model)
    pp = desc.post_process_params
    feats = Features({"version": 10}, POS_LEN)
    assert feats.bin_input_shape[0] == desc.num_input_channels
    assert feats.global_input_shape[0] == desc.num_input_global_channels

    positions = []
    for label, size, moves in build_positions(args.sgf_dir):
        for rules_name, komi in (("japanese", 6.5), ("chinese", 7.5)):
            positions.append(Position(label, size, moves, rules_name, komi))

    meta = desc.num_input_meta_channels > 0
    profiles = [p for p in args.human_sl_profiles.split(",") if p] if meta else [None]
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    tmp = tempfile.mkdtemp(prefix="bin2onnx-verify-")
    bins, globs = zip(*(p.features(feats) for p in positions))
    bin_input = np.concatenate(bins)
    global_input = np.concatenate(globs)
    # NNEvaluator::evaluate: "Avoid using policy optimism for humanSL"
    policy_optimism = 0.0 if meta else args.policy_optimism

    print(f"model  {desc.name} (version {desc.model_version}, metaEncoderVersion {desc.meta_encoder_version})")
    print(f"onnx   {args.onnx}")
    print(f"katago {katago}")
    print(f"policyOptimism {policy_optimism}  (GTP print precision: probs 1e-6, ownership 1e-7, scores 1e-3)")
    hdr = f"{'position':<22} {'rules':<8} {'tomove':<6} {'policy':>9} {'wl/nr':>9} {'owner':>9} {'lead':>8} {'scoreM':>8} {'scoreSq':>8} {'other':>8}  ok"
    all_ok = True
    worst = {}
    for profile in profiles:
        # --- ONNX ---
        feeds = {"bin_input": bin_input, "global_input": global_input}
        if meta:
            m = sgf_metadata_profile(profile)
            feeds["meta_input"] = np.stack([
                m.get_metadata_row(p.pla, p.size * p.size).astype(np.float32) for p in positions])
        policy, value, score, ownership = sess.run(["policy", "value", "scorevalue", "ownership"], feeds)
        if args.dump_features:
            dump = args.dump_features if not meta else args.dump_features.replace(".npz", "") + f".{profile}.npz"
            np.savez_compressed(dump, policy=policy, value=value, scorevalue=score, ownership=ownership, **feeds)

        # --- KataGo ---
        overrides = ",".join(x for x in [
            f"logDir={tmp}", f"homeDataDir={args.home_data_dir or tmp}", "logAllGTPCommunication=false", "logSearchInfo=false",
            "numSearchThreads=1", "gtpForceMaxNNSize=true", f"rootPolicyOptimism={args.policy_optimism}",
            f"humanSLProfile={profile}" if meta else "", args.extra_overrides,
        ] if x)
        gtp = GTP(katago, args.config, args.model, overrides)
        ref = []
        try:
            for p in positions:
                for c in p.gtp_commands():
                    gtp.send(c)
                ref.append(parse_raw_nn(gtp.send("kata-raw-nn 0"), p.size))
        finally:
            gtp.close()

        # --- compare ---
        if meta:
            print(f"humanSLProfile {profile}")
        print(hdr)
        print("-" * len(hdr))
        for i, p in enumerate(positions):
            ours = postprocess(desc, pp, p, policy[i], value[i], score[i], ownership[i, 0], policy_optimism)
            r = ref[i]
            nan_ref = np.isnan(r["policy"])
            nan_ours = np.isnan(ours["policy"])
            legal_match = bool(np.array_equal(nan_ref, nan_ours)) and (math.isnan(r["policyPass"]) == math.isnan(ours["policyPass"]))
            d_pol = max(float(np.nanmax(np.abs(np.where(nan_ref | nan_ours, 0.0, r["policy"] - ours["policy"])))),
                        abs(r["policyPass"] - ours["policyPass"]))
            d_wl = max(abs(r[k] - ours[k]) for k in ("whiteWin", "whiteLoss", "noResult"))
            d_own = float(np.max(np.abs(r["ownership"] - ours["ownership"])))
            d_lead = abs(r["whiteLead"] - ours["whiteLead"])
            d_sm = abs(r["whiteScoreSelfplay"] - ours["whiteScoreSelfplay"])
            d_sq = abs(r["whiteScoreSelfplaySq"] - ours["whiteScoreSelfplaySq"])
            d_other = max(abs(r[k] - ours[k]) for k in ("varTimeLeft", "shorttermWinlossError", "shorttermScoreError"))
            ok = (legal_match and d_pol <= args.tol_prob and d_wl <= args.tol_prob and d_own <= args.tol_prob
                  and d_lead <= args.tol_score and d_sm <= args.tol_score)
            all_ok &= ok
            for k, v in (("policy", d_pol), ("wl/nr", d_wl), ("owner", d_own), ("lead", d_lead),
                         ("scoreM", d_sm), ("scoreSq", d_sq), ("other", d_other)):
                worst[k] = max(worst.get(k, 0.0), v)
            tomove = "B" if p.pla == B else "W"
            flag = "OK" if ok else ("FAIL" + ("" if legal_match else " (legal mask mismatch)"))
            print(f"{p.label:<22} {p.rules_name:<8} {tomove:<6} {d_pol:9.2e} {d_wl:9.2e} {d_own:9.2e} "
                  f"{d_lead:8.1e} {d_sm:8.1e} {d_sq:8.1e} {d_other:8.1e}  {flag}")
    print("-" * len(hdr))
    print("max    " + "  ".join(f"{k}={v:.2e}" for k, v in worst.items()))
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

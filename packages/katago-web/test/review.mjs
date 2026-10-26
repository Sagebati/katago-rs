// Checks of the payload v3 that reviewGame returns (M6), for test/run.mjs:
// - validatePayload: the upload contract (docs/M6-report.md, "Uploads"): shapes, ranges, enumerations, turn
//   indexing, and the derived numbers recomputed from the per-move ones;
// - keyShapes: the set of key paths of a payload, to compare with the katago-ws fixtures' (same producer code);
// - comparePayloads: a browser payload against the native one of the same deterministic search, floats within
//   the NN tier's tolerances (moves, candidates, visits and PVs identical).

const CLASSES = ["good", "inaccuracy", "mistake", "blunder"];

/** Every key path of a JSON value (`moves[].top_moves[].pv[]`), arrays collapsed. */
export function keyShapes(v, path = "", out = new Set()) {
  if (Array.isArray(v)) {
    out.add(`${path}[]`);
    for (const x of v) keyShapes(x, `${path}[]`, out);
  } else if (v !== null && typeof v === "object") {
    for (const [k, x] of Object.entries(v)) {
      out.add(`${path}.${k}`);
      keyShapes(x, `${path}.${k}`, out);
    }
  }
  return out;
}

/** Problems of a payload against the contract (empty when valid). `expect`: {boardSize, moves (GTP [color, mv]
 * of the game), maxVisits, engineName, model}. */
export function validatePayload(p, expect = {}) {
  const bad = [];
  const need = (ok, what) => {
    if (!ok) bad.push(what);
  };
  const num = (x) => typeof x === "number" && Number.isFinite(x);
  const int = (x) => Number.isInteger(x);
  const prob = (x) => num(x) && x >= 0 && x <= 1;
  need(p.payload_version === 3, `payload_version ${p.payload_version}`);
  need(int(p.board_size) && p.board_size >= 2 && p.board_size <= 25, `board_size ${p.board_size}`);
  if (expect.boardSize) need(p.board_size === expect.boardSize, `board_size ${p.board_size} != ${expect.boardSize}`);
  need(num(p.komi), "komi");
  need(p.rules === "japanese" || p.rules === "chinese", `rules ${p.rules}`);
  need(Array.isArray(p.moves), "moves");
  const n = p.moves.length;
  if (expect.moves) need(n === expect.moves.length, `moves ${n} != ${expect.moves.length}`);
  const points = p.board_size * p.board_size;
  const acc = { B: { moves: 0, wl: 0, sl: 0, a: 0 }, W: { moves: 0, wl: 0, sl: 0, a: 0 } };
  p.moves.forEach((m, i) => {
    const at = `moves[${i}]`;
    need(m.move_number === i + 1, `${at}.move_number`);
    need(m.color === "B" || m.color === "W", `${at}.color`);
    if (expect.moves) need(m.color === expect.moves[i][0] && m.mv === expect.moves[i][1], `${at} is ${m.color} ${m.mv}, the game has ${expect.moves[i]}`);
    for (const k of ["winrate", "best_winrate", "winrate_loss"]) need(prob(m[k]), `${at}.${k} ${m[k]}`);
    for (const k of ["score_lead", "score_loss"]) need(num(m[k]), `${at}.${k}`);
    need(m.winrate_loss >= 0 && m.score_loss >= 0, `${at} negative loss`);
    // loss = max(best - after, 0), in f32: within an ulp or two of the f64 difference.
    need(Math.abs(m.winrate_loss - Math.max(m.best_winrate - m.winrate, 0)) <= 1e-6, `${at}.winrate_loss != best_winrate - winrate`);
    need(CLASSES.includes(m.winrate_classification) && CLASSES.includes(m.score_classification), `${at} classification`);
    const wc = m.winrate_loss < 0.03 ? "good" : m.winrate_loss < 0.07 ? "inaccuracy" : m.winrate_loss < 0.15 ? "mistake" : "blunder";
    const sc = m.score_loss < 1 ? "good" : m.score_loss < 3 ? "inaccuracy" : m.score_loss < 6 ? "mistake" : "blunder";
    need(m.winrate_classification === wc && m.score_classification === sc, `${at} classification does not follow the losses`);
    if (m.prior !== undefined) need(prob(m.prior), `${at}.prior`);
    need(Array.isArray(m.top_moves) && (!p.engine?.top_k || m.top_moves.length <= p.engine.top_k), `${at}.top_moves`);
    for (const c of m.top_moves) {
      need(typeof c.mv === "string" && prob(c.winrate) && num(c.score_lead) && prob(c.prior) && int(c.visits) && c.visits >= 0 && Array.isArray(c.pv), `${at} candidate ${c.mv}`);
      // Parallel search threads overshoot maxVisits by up to one visit each (16 per position on WebGPU).
      if (p.engine?.max_visits) need(c.visits <= p.engine.max_visits + 64, `${at} candidate ${c.mv} has ${c.visits} visits > max_visits + 64`);
    }
    const a = acc[m.color];
    a.moves += 1;
    a.wl += m.winrate_loss;
    a.sl += m.score_loss;
    a.a += Math.min(100, Math.max(0, 103.1668 * Math.exp(-0.04354 * Math.max(m.winrate_loss * 100, 0)) - 3.1669));
  });
  need(p.report?.num_moves === n, "report.num_moves");
  for (const [color, key] of [["B", "black"], ["W", "white"]]) {
    const r = p.report?.[key];
    const a = acc[color];
    need(r && r.moves === a.moves, `report.${key}.moves`);
    if (!r) continue;
    const d = Math.max(a.moves, 1);
    need(Math.abs(r.mean_winrate_loss - a.wl / d) <= 1e-5 && Math.abs(r.mean_score_loss - a.sl / d) <= 1e-3 && Math.abs(r.accuracy - a.a / d) <= 1e-3, `report.${key} means do not follow the moves`);
    for (const lens of ["by_winrate", "by_score"]) {
      const counts = r[lens];
      const field = lens === "by_winrate" ? "winrate_classification" : "score_classification";
      for (const cl of CLASSES) need(counts?.[cl] === p.moves.filter((m) => m.color === color && m[field] === cl).length, `report.${key}.${lens}.${cl}`);
    }
  }
  const e = p.engine ?? {};
  need(typeof e.name === "string" && typeof e.model === "string" && int(e.max_visits) && int(e.top_k), "engine");
  if (expect.engineName) need(e.name === expect.engineName, `engine.name ${e.name}`);
  if (expect.model) need(e.model === expect.model, `engine.model ${e.model}`);
  if (expect.maxVisits) need(e.max_visits === expect.maxVisits, `engine.max_visits ${e.max_visits}`);
  need(e.report_perspective === "SIDETOMOVE", `engine.report_perspective ${e.report_perspective}`);
  const grid = (name, lo, hi, flag) => {
    const g = p[name];
    need((g !== undefined) === !!flag, `${name} present ${g !== undefined}, engine flag ${flag}`);
    if (g === undefined) return;
    need(Array.isArray(g) && g.length === n + 1, `${name}: ${g?.length} turns for ${n} moves`);
    g.forEach((t, i) => need(Array.isArray(t) && t.length === points && t.every((x) => int(x) && x >= lo && x <= hi), `${name}[${i}]`));
  };
  grid("ownership", -100, 100, e.ownership);
  grid("ownership_stdev", 0, 100, e.ownership_stdev);
  grid("policy", 0, 255, e.policy);
  if (expect.humanProfile) {
    // A human SL network was loaded (`humanModel`): the profile's policy at every turn, its prior per move.
    need(p.human_profile === expect.humanProfile, `human_profile ${p.human_profile}`);
    if (expect.humanModel) need(e.human_model === expect.humanModel, `engine.human_model ${e.human_model}`);
    grid("human_policy", 0, 255, true);
    p.moves.forEach((m, i) => need(prob(m.human_prior) && m.top_moves.every((c) => prob(c.human_prior)), `moves[${i}] human priors`));
  } else {
    need(p.human_policy === undefined && p.human_profile === undefined && e.human_model === undefined, "human fields without a human model");
  }
  return bad;
}

const FLOATS = {
  value: ["winrate", "best_winrate", "winrate_loss", "mean_winrate_loss"],
  score: ["score_lead", "score_loss", "mean_score_loss"],
  prior: ["prior", "human_prior"],
  accuracy: ["accuracy"],
};
const KIND = Object.fromEntries(Object.entries(FLOATS).flatMap(([k, keys]) => keys.map((x) => [x, k])));

/** Differences between two payloads of the same game and settings: `structure` lists what must be equal and is
 * not; `over` the floats / grid cells beyond `tol` ({value, score, prior, accuracy, grid, policyGrid});
 * `maxDev` the largest deviation per kind. The engine objects are not compared. */
export function comparePayloads(a, b, tol) {
  const structure = [];
  const over = [];
  const maxDev = {};
  const dev = (kind, d, path) => {
    maxDev[kind] = Math.max(maxDev[kind] ?? 0, d);
    if (d > tol[kind]) over.push(`${path}: ${d.toExponential(2)} > ${tol[kind]}`);
  };
  const walk = (x, y, path) => {
    if (Array.isArray(x) || Array.isArray(y)) {
      if (!Array.isArray(x) || !Array.isArray(y) || x.length !== y.length) return structure.push(`${path}: length ${x?.length} vs ${y?.length}`);
      x.forEach((v, i) => walk(v, y[i], `${path}[${i}]`));
      return;
    }
    if (x !== null && typeof x === "object") {
      if (y === null || typeof y !== "object") return structure.push(`${path}: object vs ${typeof y}`);
      const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
      for (const k of keys) {
        if (path === "" && k === "engine") continue;
        if (!(k in x) || !(k in y)) {
          structure.push(`${path}.${k}: only in ${k in x ? "the first" : "the second"}`);
          continue;
        }
        const kind = KIND[k];
        if (kind && typeof x[k] === "number" && typeof y[k] === "number") dev(kind, Math.abs(x[k] - y[k]), `${path}.${k}`);
        else if (["ownership", "ownership_stdev", "policy", "human_policy"].includes(k) && path === "") {
          const g = k === "policy" || k === "human_policy" ? "policyGrid" : "grid";
          if (!Array.isArray(x[k]) || !Array.isArray(y[k]) || x[k].length !== y[k].length) structure.push(`.${k}: turns`);
          else x[k].forEach((t, i) => t.forEach((c, j) => dev(g, Math.abs(c - y[k][i][j]), `.${k}[${i}][${j}]`)));
        } else walk(x[k], y[k], `${path}.${k}`);
      }
      return;
    }
    if (x !== y) structure.push(`${path}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);
  };
  walk(a, b, "");
  return { structure, over, maxDev };
}

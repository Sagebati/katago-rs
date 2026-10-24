// The analysis golden comparator of crates/katago-cli/tests/common/mod.rs, ported to JS for the browser
// tests: structure byte-exact (punctuation, key order, strings, integers, booleans, null, and whether a
// number is written as an integer or a float, its exponent form and its f32/f64 storage class), floats
// within per-key tolerances. `TOLERANCES.flex` are the Rust comparator's (burn flex vs the Eigen build);
// `TOLERANCES.webgpu` widen them to the WebGPU NN tier (PORTING.md §13: wgpu <=2e-4 logits, <=1e-4 win rate
// and ownership; scores scale those by ~20).

export const TOLERANCES = {
  flex: { value: 1e-5, score: 5e-4, weightRel: 1e-5, policyRel: 1e-3, policyAbsMax: 1e-5 },
  webgpu: { value: 1e-4, score: 2e-3, weightRel: 1e-4, policyRel: 1e-2, policyAbsMax: 1e-4 },
};

function tokenize(line) {
  const toks = [];
  let i = 0;
  while (i < line.length) {
    const c = line[i];
    if (c === '"') {
      const start = i;
      i += 1;
      while (i < line.length && line[i] !== '"') {
        if (line[i] === "\\") i += 1;
        i += 1;
      }
      i += 1;
      toks.push({ t: "str", v: line.slice(start, Math.min(i, line.length)) });
    } else if ("{}[]:,".includes(c)) {
      toks.push({ t: "punct", v: c });
      i += 1;
    } else if (c === "-" || (c >= "0" && c <= "9")) {
      const start = i;
      while (i < line.length && /[-+.eE0-9]/.test(line[i])) i += 1;
      toks.push({ t: "num", v: line.slice(start, i) });
    } else if (c === "t" || c === "f" || c === "n") {
      const start = i;
      while (i < line.length && /[a-zA-Z]/.test(line[i])) i += 1;
      toks.push({ t: "lit", v: line.slice(start, i) });
    } else {
      throw new Error(`unexpected character ${JSON.stringify(c)} at ${i}`);
    }
  }
  return toks;
}

const isFloatLiteral = (s) => s.includes(".") || s.includes("e") || s.includes("E");

/** [exponent form, significant digits of the mantissa] */
function floatFormat(s) {
  const p = s.search(/[eE]/);
  const mant = p >= 0 ? s.slice(0, p) : s;
  const digits = mant.replace(/[^0-9]/g, "").replace(/^0+/, "").replace(/0+$/, "");
  return [p >= 0, Math.max(digits.length, 1)];
}

function sameFloatClass(a, b) {
  const [ea, da] = floatFormat(a);
  const [eb, db] = floatFormat(b);
  const x = Number(a);
  const y = Number(b);
  const expForm = (v) => v !== 0 && (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e15);
  if (ea !== eb && expForm(x) === expForm(y)) return false;
  const isF32 = (v, d) => d > 4 && Math.fround(v) === v;
  const fa = isF32(x, da);
  const fb = isF32(y, db);
  if (fa !== fb && da > 4 && db > 4) return false;
  return true;
}

function tolClass(key) {
  if (key.includes("score") || key.includes("Score") || key.includes("lead") || key.includes("Lead")) return "score";
  if (key === "weight" || key === "edgeWeight" || key === "playSelectionValue") return "weight";
  if (key === "prior" || key === "policy" || key === "humanPrior" || key === "humanPolicy") return "policy";
  return "value";
}

/** Compares one line with its golden; returns null or the first violation. Updates maxDev[key]. */
export function compareLine(g, o, maxDev, tol) {
  if (g === o) return null;
  let tg, to;
  try {
    tg = tokenize(g);
    to = tokenize(o);
  } catch (e) {
    return `not tokenizable (${e.message})\n  expected: ${g}\n  got:      ${o}`;
  }
  if (tg.length !== to.length) return `${to.length} tokens vs ${tg.length} expected\n  expected: ${g}\n  got:      ${o}`;
  let key = "";
  for (let i = 0; i < tg.length; i++) {
    if (tg[i].t === "str" && tg[i + 1] && tg[i + 1].t === "punct" && tg[i + 1].v === ":") key = tg[i].v.replace(/^"|"$/g, "");
    const a = tg[i];
    const b = to[i];
    let ok;
    if (a.t === "num" && b.t === "num" && a.v !== b.v) {
      if (isFloatLiteral(a.v) && isFloatLiteral(b.v) && sameFloatClass(a.v, b.v)) {
        const x = Number(a.v);
        const y = Number(b.v);
        const d = Math.abs(x - y);
        if (!(key in maxDev) || d > maxDev[key]) maxDev[key] = d;
        switch (tolClass(key)) {
          case "score":
            ok = d <= tol.score;
            break;
          case "weight":
            ok = d <= tol.weightRel * Math.max(1, Math.abs(x));
            break;
          case "policy":
            ok = d <= Math.min(Math.max(tol.policyRel * Math.abs(x), 1e-9), tol.policyAbsMax);
            break;
          default:
            ok = d <= tol.value;
        }
      } else {
        ok = false;
      }
    } else {
      ok = a.t === b.t && a.v === b.v;
    }
    if (!ok) return `key ${JSON.stringify(key)}: ${b.v} vs expected ${a.v}\n  expected: ${g}\n  got:      ${o}`;
  }
  return null;
}

/** {numLines, byteIdentical, maxDev, divergence} */
export function compare(gold, got, tol = TOLERANCES.flex) {
  const c = { numLines: gold.length, byteIdentical: gold.length === got.length && gold.every((g, i) => g === got[i]), maxDev: {}, divergence: null };
  if (gold.length !== got.length) {
    c.divergence = `${got.length} lines vs ${gold.length} expected`;
    return c;
  }
  for (let i = 0; i < gold.length; i++) {
    const e = compareLine(gold[i], got[i], c.maxDev, tol);
    if (e) {
      c.divergence = `line ${i + 1}, ${e}`;
      return c;
    }
  }
  return c;
}

export function status(c) {
  if (c.divergence) return `DIVERGED: ${c.divergence}`;
  if (c.byteIdentical) return "exact (byte-identical)";
  const worst = Object.entries(c.maxDev)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k, v]) => `${k} ${v.toExponential(2)}`);
  return `within tolerance (max dev: ${worst.join(", ")})`;
}

// Self-check, as the Rust comparator_catches_differences test (run: node test/compare.mjs <basic golden>).
export function selfCheck(gold) {
  const assert = (cond, msg) => {
    if (!cond) throw new Error(`comparator self-check: ${msg}`);
  };
  const tol = TOLERANCES.flex;
  assert(compare(gold, gold).byteIdentical, "identical");
  const g2 = [...gold];
  g2[0] = g2[0].replace('"visits":9', '"visits":8');
  assert(g2[0] !== gold[0] && compare(gold, g2, tol).divergence, "integer change");
  const g3 = [...gold];
  g3[0] = g3[0].replace('"winrate":0.61100084', '"winrate":0.61102084');
  assert(g3[0] !== gold[0] && compare(gold, g3, tol).divergence, "winrate +2e-5");
  const g4 = [...gold];
  g4[0] = g4[0].replace('"winrate":0.61100084', '"winrate":0.61100284');
  const c4 = compare(gold, g4, tol);
  assert(!c4.divergence && !c4.byteIdentical, "winrate +2e-6 within tolerance");
  const g5 = [...gold];
  g5[0] = g5[0].replace('"edgeWeight":9.0', '"edgeWeight":9');
  assert(compare(gold, g5, tol).divergence, "float written as integer");
  assert(compare(gold, gold.slice(1), tol).divergence, "missing line");
  const differs = (g, o) => compareLine(g, o, {}, tol) !== null;
  assert(differs('{"policy":[2.5e-06]}', '{"policy":[7.5e-06]}'), "tiny policy relative");
  assert(!differs('{"prior":0.25}', '{"prior":0.250004}'), "prior abs");
  assert(differs('{"winrate":0.000123}', '{"winrate":1.23e-04}'), "exponent form");
  assert(!differs('{"winrate":0.0001}', '{"winrate":9.9999e-05}'), "straddling 1e-4");
  assert(!differs('{"scoreStdev":4.59838054}', '{"scoreStdev":4.59825107}'), "score tolerance");
  assert(differs('{"scoreLead":1.5}', '{"scoreLead":1.5006}'), "score over tolerance");
  return true;
}

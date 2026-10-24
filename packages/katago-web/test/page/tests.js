// Browser side of the katago-web tests (driven by test/run.mjs through Playwright). Each function runs
// one scenario with a fresh KataGoClient and returns plain data; the comparisons happen in node.

import { KataGoClient, fetchModelSpec } from "/pkg/index.js";

const text = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.text();
};

const logs = [];
const onLog = (kind, t) => {
  if (kind !== "info" || t.startsWith("katago-wasm:")) logs.push(`${kind}: ${t}`);
};

async function client(opts) {
  const network = opts.network ?? "g170e-b10c128";
  const model = await fetchModelSpec("/pkg/models/models.json", network);
  const c = new KataGoClient({ model, onLog, ...opts });
  const ready = await c.ready;
  return { c, ready };
}

/** One upstream analysis golden: analysis_example.cfg + regen.sh's overrides, the query file's lines as one
 * burst, end of input, every output line. */
export async function runGolden({ backend, queryFile, overrides, modelFile }) {
  const config = await text("/upstream/configs/analysis_example.cfg");
  const lines = (await text(`/upstream/tests/analysis/${queryFile}`)).split("\n").filter((l) => l.length > 0);
  const t0 = performance.now();
  const { c, ready } = await client({ backend, network: "g170-b6c96", config, overrides: [overrides], modelFile, warmupVisits: 0 });
  const out = [];
  c.onLine((l) => out.push(l));
  const t1 = performance.now();
  c.sendRaw(lines);
  const code = await c.closeInput();
  return { backend: ready.backend, fallbackReason: ready.fallbackReason ?? null, code, lines: out, startMs: t1 - t0, runMs: performance.now() - t1 };
}

/** Model load: returns the ready info (backend, model source, timings). */
export async function loadModel({ backend, network, workerUrl }) {
  const { c, ready } = await client({ backend, network, workerUrl });
  c.dispose();
  return ready;
}

const EMPTY_19 = { moves: [], rules: "tromp-taylor", komi: 7.5, boardXSize: 19, boardYSize: 19 };
// A middle-game position (moves of an AlphaGo Zero self-play opening), so the NN cache sees new positions.
const GAME_MOVES = [
  ["B", "Q16"], ["W", "D4"], ["B", "Q3"], ["W", "D16"], ["B", "C3"], ["W", "D3"], ["B", "C4"], ["W", "D5"],
  ["B", "B6"], ["W", "R5"], ["B", "R4"], ["W", "Q5"], ["B", "P3"], ["W", "O4"], ["B", "C14"], ["W", "D14"],
  ["B", "C13"], ["W", "C16"], ["B", "D13"], ["W", "F15"], ["B", "R10"], ["W", "Q11"], ["B", "Q10"], ["W", "P11"],
];

/** visits/s and evals/s of maxVisits searches on fresh positions (NN cache cleared before each). */
export async function throughput({ backend, network, visits, positions }) {
  const { c, ready } = await client({ backend, network });
  const results = [];
  for (let p = 0; p < positions; p++) {
    await c.clearCache();
    const s0 = await c.stats();
    const t0 = performance.now();
    const moves = GAME_MOVES.slice(0, 2 * p + 4);
    const finals = await c.analyze({ ...EMPTY_19, moves, maxVisits: visits }).done;
    const ms = performance.now() - t0;
    const s1 = await c.stats();
    const rows = s1.rows - s0.rows;
    const batches = s1.batches - s0.batches;
    results.push({
      moves: moves.length,
      visits: finals[0].rootInfo.visits,
      ms,
      visitsPerS: (finals[0].rootInfo.visits * 1000) / ms,
      rows,
      batches,
      avgBatch: rows / batches,
      evalsPerS: (rows * 1000) / ms,
      nnBusyEvalsPerS: (rows * 1000) / (s1.busyMs - s0.busyMs),
      top: finals[0].moveInfos[0].move,
    });
  }
  c.dispose();
  return { backend: ready.backend, ready, results };
}

/** Long searches with partial reports, each terminated at a random point after a few partials: the
 * terminate latency (cancel() to the final response) against the NN batch time of those searches. */
export async function terminateLatency({ backend, network, partialsBeforeCancel, every, trials }) {
  const { c, ready } = await client({ backend, network });
  const out = [];
  for (let trial = 0; trial < trials; trial++) {
    const s0 = await c.stats();
    const run = c.analyze({ ...EMPTY_19, moves: GAME_MOVES.slice(0, 10 + 2 * trial), maxVisits: 1000000, reportDuringSearchEvery: every });
    const partials = [];
    let final = null;
    let tCancel = 0;
    let tFinal = 0;
    let cancelled = false;
    // A random point inside the NN batch cycle, not just after a report.
    const jitterMs = Math.random() * 100;
    for await (const r of run) {
      if (r.isDuringSearch) {
        partials.push({ t: performance.now(), visits: r.rootInfo.visits });
        if (partials.length === partialsBeforeCancel && !cancelled) {
          cancelled = true;
          setTimeout(() => {
            tCancel = performance.now();
            void run.cancel();
          }, jitterMs);
        }
      } else {
        tFinal = performance.now();
        final = r;
      }
    }
    const s1 = await c.stats();
    const gaps = partials.slice(1).map((p, i) => p.t - partials[i].t);
    out.push({
      partials: partials.length,
      partialVisits: partials.map((p) => p.visits),
      partialGapMs: { min: Math.min(...gaps), max: Math.max(...gaps), mean: gaps.reduce((a, b) => a + b, 0) / gaps.length },
      finalIsDuringSearch: final?.isDuringSearch ?? null,
      finalVisits: final?.rootInfo?.visits ?? null,
      jitterMs,
      terminateLatencyMs: tFinal - tCancel,
      avgBatchMs: (s1.busyMs - s0.busyMs) / (s1.batches - s0.batches),
      avgBatchRows: (s1.rows - s0.rows) / (s1.batches - s0.batches),
      // the terminate is the last input line the engine got
      batchesAfterTerminate: s1.batchesSinceLastInput,
      rowsAfterTerminate: s1.rowsSinceLastInput,
    });
  }
  c.dispose();
  return { backend: ready.backend, trials: out };
}

/** query_version through the client API, and a reportDuringSearchEvery search through analyze(). */
export async function apiSmoke({ backend }) {
  const { c, ready } = await client({ backend, network: "g170e-b10c128" });
  const version = await c.queryVersion();
  const models = await c.queryModels();
  const responses = [];
  for await (const r of c.analyze({ id: "mine", ...EMPTY_19, analyzeTurns: [0], maxVisits: backend === "cpu" ? 40 : 400, reportDuringSearchEvery: 0.05, includeOwnership: true })) {
    responses.push({ id: r.id, isDuringSearch: r.isDuringSearch, visits: r.rootInfo.visits, ownership: r.ownership?.length ?? 0 });
  }
  let error = null;
  try {
    await c.analyze({ ...EMPTY_19, boardXSize: 99 }).done;
  } catch (e) {
    error = String(e.message);
  }
  c.dispose();
  return { backend: ready.backend, version, models, responses, error };
}

export function takeLogs() {
  return logs.splice(0);
}

window.kgwTests = { runGolden, loadModel, throughput, terminateLatency, apiSmoke, takeLogs };
window.kgwReady = true;

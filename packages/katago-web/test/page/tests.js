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

// The scenarios compare against the Eigen reference, so they run the WebGPU convolutions in f32 unless a
// scenario asks for `precision` (the client's own default is `auto`: f16 where the adapter has shader-f16).
async function client(opts) {
  const { network = "g170e-b10c128", cpuNetwork, humanNetwork, precision = "f32", ...rest } = opts;
  const model = await fetchModelSpec("/pkg/models/models.json", network);
  const cpuModel = cpuNetwork ? await fetchModelSpec("/pkg/models/models.json", cpuNetwork) : undefined;
  const humanModel = humanNetwork ? await fetchModelSpec("/pkg/models/models.json", humanNetwork) : undefined;
  const c = new KataGoClient({ model, cpuModel, humanModel, onLog, precision, ...rest });
  const ready = await c.ready;
  return { c, ready };
}

/** One upstream analysis golden: analysis_example.cfg + regen.sh's overrides, the query file's lines as one
 * burst, end of input, every output line. */
export async function runGolden({ backend, queryFile, overrides, modelFile, network = "g170-b6c96", workerUrl }) {
  const config = await text("/upstream/configs/analysis_example.cfg");
  const lines = (await text(`/upstream/tests/analysis/${queryFile}`)).split("\n").filter((l) => l.length > 0);
  const t0 = performance.now();
  const { c, ready } = await client({ backend, network, config, overrides: [overrides], modelFile, warmupVisits: 0, workerUrl });
  const out = [];
  c.onLine((l) => out.push(l));
  const t1 = performance.now();
  c.sendRaw(lines);
  const code = await c.closeInput();
  return { backend: ready.backend, fallbackReason: ready.fallbackReason ?? null, code, lines: out, startMs: t1 - t0, runMs: performance.now() - t1 };
}

/** Model load: returns the ready info (backend, model source, timings). */
export async function loadModel({ backend, network, workerUrl, cpuNetwork, allowSlowCpu, precision }) {
  const { c, ready } = await client({ backend, network, workerUrl, cpuNetwork, allowSlowCpu, precision });
  c.dispose();
  return ready;
}

/** loadModel that reports a failure (`ready` rejecting) instead of throwing: {ready} | {error}. */
export async function tryLoadModel(opts) {
  try {
    return { ready: await loadModel(opts) };
  } catch (e) {
    return { error: String(e.message ?? e) };
  }
}

/** A tiny search on the CPU with a GPU-only net (allowSlowCpu): the time per visit there, 1 search thread. */
export async function slowCpuSearch({ network, visits }) {
  const { c, ready } = await client({ backend: "cpu", network, allowSlowCpu: true, warmupVisits: 0 });
  const s0 = await c.stats();
  const t0 = performance.now();
  const finals = await c.analyze({ ...EMPTY_19, moves: GAME_MOVES.slice(0, 10), maxVisits: visits }).done;
  const ms = performance.now() - t0;
  const s1 = await c.stats();
  c.dispose();
  return { ready, visits: finals[0].rootInfo.visits, ms, rows: s1.rows - s0.rows, batches: s1.batches - s0.batches, top: finals[0].moveInfos[0].move };
}

/** Storage quota and usage of the origin (the Cache API holds the weights) and the adapter's limits, seen from the page. */
export async function storageAndLimits() {
  const est = await navigator.storage.estimate();
  const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  const l = a?.limits;
  return {
    quota: est.quota,
    usage: est.usage,
    limits: l && { maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize, maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension, maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage },
    features: a ? [...a.features] : [],
  };
}

const EMPTY_19 = { moves: [], rules: "tromp-taylor", komi: 7.5, boardXSize: 19, boardYSize: 19 };
// A middle-game position (moves of an AlphaGo Zero self-play opening), so the NN cache sees new positions.
const GAME_MOVES = [
  ["B", "Q16"], ["W", "D4"], ["B", "Q3"], ["W", "D16"], ["B", "C3"], ["W", "D3"], ["B", "C4"], ["W", "D5"],
  ["B", "B6"], ["W", "R5"], ["B", "R4"], ["W", "Q5"], ["B", "P3"], ["W", "O4"], ["B", "C14"], ["W", "D14"],
  ["B", "C13"], ["W", "C16"], ["B", "D13"], ["W", "F15"], ["B", "R10"], ["W", "Q11"], ["B", "Q10"], ["W", "P11"],
];

/** Raw network outputs (1-visit searches, symmetry 0) on the first moves of GAME_MOVES: root winrate and score
 * lead, the policy and the ownership, for comparing precisions. */
export async function nnOutputs({ backend, network, precision, positions }) {
  const { c, ready } = await client({ backend, network, precision, overrides: "nnRandomize=false" });
  const out = [];
  for (let p = 0; p < positions; p++) {
    const moves = GAME_MOVES.slice(0, 2 * p);
    const [r] = await c.analyze({ ...EMPTY_19, moves, maxVisits: 1, includePolicy: true, includeOwnership: true }).done;
    out.push({ winrate: r.rootInfo.winrate, scoreLead: r.rootInfo.scoreLead, policy: r.policy, ownership: r.ownership });
  }
  c.dispose();
  return { backend: ready.backend, precision: ready.precision ?? null, out };
}

/** visits/s and evals/s of maxVisits searches on fresh positions (NN cache cleared before each). */
export async function throughput({ backend, network, visits, positions, overrides, workerUrl, precision }) {
  const { c, ready } = await client({ backend, network, overrides, workerUrl, precision });
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

/** The compiled-kernel cache (WebGPU): a deterministic search (1 search thread, no NN randomization) in an engine
 * that compiles its kernels (cache emptied first), then in one that loads them from the cache. Returns both ready
 * infos and whether the final responses are identical. */
export async function kernelCacheParity({ network, cacheKernels = true }) {
  await caches.delete("katago-web-kernels-v1");
  const overrides = "numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=kernelCache, forDeterministicTesting=true";
  const run = async () => {
    const { c, ready } = await client({ backend: "webgpu", network, overrides, cacheKernels });
    const finals = await c.analyze({ id: "k", ...EMPTY_19, moves: GAME_MOVES.slice(0, 12), maxVisits: 48, includeOwnership: true, includePolicy: true }).done;
    c.dispose();
    return { ready, text: JSON.stringify(finals) };
  };
  const a = await run();
  const b = await run();
  const pick = (r) => ({ backend: r.ready.backend, kernelCache: r.ready.kernelCache ?? null, timings: r.ready.timings });
  return { first: pick(a), second: pick(b), identical: a.text === b.text, length: a.text.length, texts: a.text === b.text ? undefined : [a.text, b.text] };
}

/** Full-game reviews (M6) in one engine: each `{game, options, cancelAfter?}` of `reviews` in turn, through
 * KataGoClient.reviewGame. Returns per review the payload JSON text (or the error), the progress events with
 * their times, and the wall clock; `config` / `overrides` / `analysisThreads` set up the engine (deterministic
 * runs for the native comparison). With `cancelAfter`, the review is cancelled after that many progress events. */
export async function review({ backend, network, humanNetwork, allowSlowCpu, config, configUrl, overrides, analysisThreads, warmupVisits, reviews, clearCache }) {
  if (configUrl) config = await text(configUrl);
  const { c, ready } = await client({ backend, network, humanNetwork, allowSlowCpu, config, overrides, analysisThreads, warmupVisits });
  const out = [];
  for (const { game, options, cancelAfter } of reviews) {
    if (clearCache) await c.clearCache();
    const s0 = await c.stats();
    const progress = [];
    const t0 = performance.now();
    let tCancel = null;
    const run = c.reviewGame(game, options, (p) => {
      progress.push({ ...p, t: performance.now() - t0 });
      if (cancelAfter && progress.length === cancelAfter) {
        tCancel = performance.now() - t0;
        run.cancel();
      }
    });
    let json = null;
    let error = null;
    let cancelled = false;
    try {
      json = await run.json;
    } catch (e) {
      error = String(e.message);
      cancelled = e.cancelled === true;
    }
    const ms = performance.now() - t0;
    const s1 = await c.stats();
    const nn = { rows: s1.rows - s0.rows, batches: s1.batches - s0.batches, busyMs: s1.busyMs - s0.busyMs, maxBatchMs: s1.maxBatchMs };
    out.push({ json, error, cancelled, progress, ms, tCancel, nn });
  }
  // The engine still answers ordinary queries after the reviews (and nothing of theirs leaked into them).
  const after = await c.analyze({ ...EMPTY_19, moves: GAME_MOVES.slice(0, 2), maxVisits: 8 }).done;
  const stats = await c.stats();
  c.dispose();
  return { backend: ready.backend, ready, reviews: out, after: after.map((r) => ({ id: r.id, turnNumber: r.turnNumber, visits: r.rootInfo.visits })), stats };
}

/** Forward passes at a fixed batch size, the browser's counterpart of `just bench`: `batch` analysis threads of one
 * search thread each, every turn of `games` at maxVisits 1 (one root evaluation per turn), so the NN batches fill to
 * `batch` rows. evals/s are rows over the NN's busy time (get_output: upload, kernels, readback), as bench.rs times
 * them; `avgBatch` is the batch size the engine actually ran. One warm-up game first, then the NN cache emptied. */
export async function benchForward({ network, precision, batch, games }) {
  const overrides = `numAnalysisThreads=${batch}, numSearchThreadsPerAnalysisThread=1, nnMaxBatchSize=${batch}`;
  const { c, ready } = await client({ backend: "webgpu", network, precision, overrides });
  const run = (moves) => c.analyze({ ...EMPTY_19, moves, analyzeTurns: [...Array(moves.length + 1).keys()], maxVisits: 1 }).done;
  await run(games[0]);
  await c.clearCache();
  const s0 = await c.stats();
  const t0 = performance.now();
  await Promise.all(games.slice(1).map(run));
  const ms = performance.now() - t0;
  const s1 = await c.stats();
  c.dispose();
  const rows = s1.rows - s0.rows;
  const batches = s1.batches - s0.batches;
  return { precision: ready.precision, batch, rows, batches, avgBatch: rows / batches, evalsPerS: (rows * 1000) / (s1.busyMs - s0.busyMs), wallEvalsPerS: (rows * 1000) / ms };
}

/** Time to analyse one position to `visits` (KataGoClient.analyze with the drill check's query: the 8-move lines
 * of liveAnalysis.check), one engine. `positions` are move lists from the start of the board; the first is a
 * warm-up and is not timed. Per timed position: its wall ms (to the final answer) and the visits it reached.
 * `avgBatch` is the NN batch size over all the positions (the warm-up included), which one query fills only
 * up to its search threads. */
export async function benchPosition({ network, precision, overrides, positions, visits }) {
  const { c, ready } = await client({ backend: "webgpu", network, precision, overrides });
  const timed = [];
  const s0 = await c.stats();
  for (const [i, moves] of positions.entries()) {
    const t0 = performance.now();
    const [final] = await c.analyze({ ...EMPTY_19, moves, maxVisits: visits, analysisPVLen: 8 }).done;
    const ms = performance.now() - t0;
    if (i > 0) timed.push({ ms, visits: final.rootInfo.visits });
  }
  const s1 = await c.stats();
  c.dispose();
  return { precision: ready.precision, overrides: overrides ?? null, visits, positions: timed, avgBatch: (s1.rows - s0.rows) / (s1.batches - s0.batches) };
}

export function takeLogs() {
  return logs.splice(0);
}

window.kgwTests = { runGolden, loadModel, tryLoadModel, slowCpuSearch, storageAndLimits, throughput, nnOutputs, terminateLatency, apiSmoke, kernelCacheParity, review, benchForward, benchPosition, takeLogs };
window.kgwReady = true;

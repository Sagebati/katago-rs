// Browser tests of katago-web (`just web-test`): a static server for the built package, the upstream query
// files / config and the goldens, and Playwright driving headless system Chromium.
//
//   node packages/katago-web/test/run.mjs [--only goldens,cache,kernels,terminate,partials,throughput,fallback]
//        [--backends webgpu,cpu] [--goldens basic,symmetry] [--json out.json] [--headed]
//
// Env: PLAYWRIGHT (path of the playwright package; default: sai-frontend's node_modules), CHROMIUM
// (default /usr/bin/chromium). WebGPU on Linux needs --enable-unsafe-webgpu --enable-features=Vulkan
// --use-angle=vulkan (docs/M0-report.md, docs/M4-report.md). Exit code 1 if any check fails.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { readFileSync, writeFileSync } from "node:fs";
import { extname, join, normalize, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { compare, status, selfCheck, TOLERANCES } from "./compare.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = resolve(here, "..");
const root = resolve(pkg, "../..");
const PLAYWRIGHT = process.env.PLAYWRIGHT ?? "/home/sam/src/web/baduk_review/sai-frontend/node_modules/playwright";
const CHROMIUM = process.env.CHROMIUM ?? "/usr/bin/chromium";
const { chromium } = await import(pathToFileURL(join(PLAYWRIGHT, "index.mjs")).href);

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const only = opt("only", "goldens,fallback,cache,kernels,partials,terminate,throughput").split(",");
const backends = opt("backends", "webgpu,cpu").split(",");
const headed = args.includes("--headed");
const jsonOut = opt("json", null);

// The goldens of crates/katago-cli/tests/analysis_goldens.rs (fixtures/upstream-v1.16.5/analysis/regen.sh),
// minus eval_cache (deferred, D7) and humansl (M8).
const COMMON =
  "logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false";
const LOOP = "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false";
const M6 = "tests/models/g170-b6c96-s175395328-d26788732.bin.gz";
const GOLDENS = [
  ["basic", "basic.txt.stdout", "basic.txt", LOOP],
  ["basic_sidetomove", "basic_sidetomove.stdout", "basic.txt", `${LOOP}, reportAnalysisWinratesAs=SIDETOMOVE`],
  ["badoverride", "badoverride.txt.stdout", "badoverride.txt", LOOP],
  ["clear_cache", "clear_cache.txt.stdout", "clear_cache.txt", LOOP],
  ["symmetry", "symmetry.txt.stdout", "symmetry.txt", LOOP],
  ["symmetry_with_pruning", "symmetry_with_pruning.stdout", "symmetry.txt", "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=true"],
  ["evalcache_loop", "evalcache.txt.stdout", "evalcache.txt", LOOP],
  ["pvvisits", "pvvisits.txt.stdout", "pvvisits.txt", LOOP],
  ["pvvisits_nograph", "pvvisits_nograph.stdout", "pvvisits.txt", "rootSymmetryPruning=false, useGraphSearch=false"],
];
const goldenFilter = opt("goldens", null)?.split(",");

//------------------------------------------------------------------------------------------------
// Server

const MOUNTS = [
  ["/pkg/", join(pkg, "dist")],
  ["/test/", join(here, "page")],
  ["/upstream/configs/", join(root, "upstream/KataGo/cpp/configs")],
  ["/upstream/tests/analysis/", join(root, "upstream/KataGo/cpp/tests/analysis")],
];
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".cfg": "text/plain", ".txt": "text/plain", ".bpk": "application/octet-stream" };
const requestLog = [];

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  requestLog.push(url.pathname);
  for (const [prefix, dir] of MOUNTS) {
    if (!url.pathname.startsWith(prefix)) continue;
    const file = normalize(join(dir, decodeURIComponent(url.pathname.slice(prefix.length))));
    if (!file.startsWith(dir)) break;
    try {
      const st = await stat(file);
      if (!st.isFile()) break;
      res.writeHead(200, {
        "content-type": TYPES[extname(file)] ?? "application/octet-stream",
        "content-length": st.size,
        // No HTTP caching anywhere: a second load without model requests is the Cache API's doing.
        "cache-control": "no-store",
      });
      res.end(await readFile(file));
      return;
    } catch {
      break;
    }
  }
  res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

//------------------------------------------------------------------------------------------------
// Browser

// Chromium 156 headless: without --use-angle=vulkan the only adapter is SwiftShader (CPU Vulkan).
const WEBGPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan"];

async function openPage(browser) {
  const context = await browser.newContext();
  const page = await context.newPage();
  const browserRequests = [];
  context.on("request", (r) => browserRequests.push(r.url()));
  page.on("console", (m) => {
    if (process.env.KGW_ALL_CONSOLE || m.type() === "error" || m.type() === "warning") console.log(`  [page ${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`  [pageerror] ${e}`));
  await page.goto(`${origin}/test/index.html`);
  await page.waitForFunction(() => window.kgwReady === true);
  return { context, page, browserRequests };
}

const call = (page, fn, arg) => page.evaluate(([f, a]) => window.kgwTests[f](a), [fn, arg]);

const results = { origin, checks: [], goldens: [], fallback: null, cache: null, partials: [], terminate: [], throughput: [] };
let failures = 0;
function check(name, ok, detail) {
  results.checks.push({ name, ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

const readGolden = (f) =>
  readFileSync(join(root, "fixtures/upstream-v1.16.5/analysis", f), "utf8")
    .split("\n")
    .filter((l) => l.length > 0);

selfCheck(readGolden("basic.txt.stdout"));

async function goldens(page, backend, label, filter = goldenFilter) {
  for (const [name, golden, query, extra] of GOLDENS) {
    if (filter && !filter.includes(name)) continue;
    const r = await call(page, "runGolden", { backend, queryFile: query, overrides: `${COMMON}, ${extra}`, modelFile: M6 });
    const gold = readGolden(golden);
    const tol = r.backend === "webgpu" ? TOLERANCES.webgpu : TOLERANCES.flex;
    const c = compare(gold, r.lines, tol);
    // WebGPU is held to the WebGPU tier; also record whether it would pass the flex (native CPU) tier.
    const flexTier = r.backend === "webgpu" ? status(compare(gold, r.lines, TOLERANCES.flex)) : null;
    results.goldens.push({ label, backend: r.backend, name, status: status(c), flexTier, ok: !c.divergence, lines: r.lines.length, startMs: r.startMs, runMs: r.runMs, maxDev: c.maxDev, fallbackReason: r.fallbackReason });
    check(`golden ${name} [${label}]`, !c.divergence && r.code === 0, `${status(c)} (${r.lines.length} lines, run ${(r.runMs / 1000).toFixed(1)} s)${flexTier ? `; flex tier: ${flexTier.split("\n")[0]}` : ""}`);
  }
}

const gpuBrowser = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: WEBGPU_ARGS });
try {
  if (only.includes("goldens")) {
    for (const backend of backends) {
      const { context, page } = await openPage(gpuBrowser);
      await goldens(page, backend, backend);
      await context.close();
    }
  }

  if (only.includes("cache")) {
    // Fresh context (empty Cache API): first load fetches the weights, a reload reads them from the cache.
    const { context, page, browserRequests } = await openPage(gpuBrowser);
    const bpkRequests = () => requestLog.filter((p) => p.endsWith(".bpk")).length;
    const before = bpkRequests();
    const cold = await call(page, "loadModel", { backend: "auto", network: "g170e-b10c128" });
    const afterCold = bpkRequests();
    browserRequests.length = 0;
    await page.reload();
    await page.waitForFunction(() => window.kgwReady === true);
    const warm = await call(page, "loadModel", { backend: "auto", network: "g170e-b10c128" });
    const afterWarm = bpkRequests();
    const browserBpk = browserRequests.filter((u) => u.endsWith(".bpk"));
    results.cache = { cold, warm, serverBpkRequests: { cold: afterCold - before, warm: afterWarm - afterCold }, browserBpkRequestsWarm: browserBpk };
    check("first load fetches the weights once", cold.modelSource === "network" && afterCold - before === 1, `${afterCold - before} .bpk request(s), source ${cold.modelSource}`);
    check("second load does no model network request", warm.modelSource === "cache" && afterWarm === afterCold && browserBpk.length === 0, `server saw ${afterWarm - afterCold}, browser issued ${browserBpk.length}, source ${warm.modelSource}`);
    console.log(`  cold: ${JSON.stringify(cold.timings)}\n  warm: ${JSON.stringify(warm.timings)}`);
    if (cold.backend === "webgpu") {
      // The reload finds the kernels the first load compiled (docs/M4-report.md, "Start-up on WebGPU").
      check("second load compiles no WebGPU kernel", cold.kernelCache?.compiled > 0 && warm.kernelCache?.compiled === 0 && warm.kernelCache?.hits > 0, `cold ${JSON.stringify(cold.kernelCache)}, warm ${JSON.stringify(warm.kernelCache)}; ready in ${cold.timings.totalMs.toFixed(0)} ms cold, ${warm.timings.totalMs.toFixed(0)} ms warm (warm-up ${cold.timings.warmupMs.toFixed(0)} / ${warm.timings.warmupMs.toFixed(0)} ms)`);
    }
    await context.close();
  }

  if (only.includes("kernels") && backends.includes("webgpu")) {
    const { context, page } = await openPage(gpuBrowser);
    const r = await call(page, "kernelCacheParity", { network: "g170e-b10c128" });
    results.kernelCache = r;
    check("kernel cache: compiled kernels saved, then loaded", r.first.kernelCache?.compiled > 0 && r.first.kernelCache?.loaded === 0 && r.second.kernelCache?.loaded > 0 && r.second.kernelCache?.compiled === 0, `first ${JSON.stringify(r.first.kernelCache)}, second ${JSON.stringify(r.second.kernelCache)}`);
    check("kernel cache: bit-identical analysis with cached kernels", r.identical, `${r.length} chars of final responses ${r.identical ? "identical" : "DIFFER"}`);
    await context.close();
  }

  for (const backend of backends) {
    const { context, page } = await openPage(gpuBrowser);
    if (only.includes("partials")) {
      const r = await call(page, "apiSmoke", { backend });
      results.partials.push(r);
      const partials = r.responses.filter((x) => x.isDuringSearch);
      const finals = r.responses.filter((x) => !x.isDuringSearch);
      const increasing = partials.every((p, i) => i === 0 || p.visits >= partials[i - 1].visits);
      check(`reportDuringSearchEvery partials arrive [${backend}]`, partials.length >= 2 && finals.length === 1 && increasing && r.responses.every((x) => x.id === "mine" && x.ownership === 361), `${partials.length} partials (visits ${partials.map((p) => p.visits).join(",")}), final ${finals[0]?.visits} visits`);
      check(`queryVersion / queryModels [${backend}]`, r.version.version === "1.16.5" && r.models.models[0].name === "g170e-b10c128", `${r.version.version} ${r.version.git_hash}, ${r.models.models.map((m) => `${m.name} batch ${m.maxBatchSize}`).join(", ")}`);
      check(`error response rejects the run [${backend}]`, r.error !== null && r.error.includes("boardXSize"), r.error);
    }
    if (only.includes("terminate")) {
      const r = await call(page, "terminateLatency", { backend, network: "g170e-b10c128", partialsBeforeCancel: 3, every: backend === "cpu" ? 0.5 : 0.25, trials: 5 });
      results.terminate.push(r);
      // When the terminate reaches the engine, the evaluations its search threads already requested (at
      // most one per search thread: one batch's worth, 16 rows on WebGPU, 1 on the CPU) still complete,
      // normally as the one batch in flight; then the searches stop. Checked on the engine's counters
      // (rows and batches completed after the terminate line arrived) and on the wall clock (cancel() to
      // the final response: plus the hops between threads, a synchronous search step the message may wait
      // behind, and the final response's JSON), bounded by 1.5 average batches + 25 ms.
      const threads = backend === "cpu" ? 1 : 16;
      for (const [i, t] of r.trials.entries()) {
        const bound = 1.5 * t.avgBatchMs + 25;
        check(
          `terminate returns within one NN batch [${backend} #${i + 1}]`,
          t.finalIsDuringSearch === false && t.rowsAfterTerminate <= threads && t.terminateLatencyMs <= bound,
          `${t.terminateLatencyMs.toFixed(1)} ms after cancel(), ${t.rowsAfterTerminate} evals in ${t.batchesAfterTerminate} batch(es) after the terminate (avg batch ${t.avgBatchMs.toFixed(1)} ms of ${t.avgBatchRows.toFixed(1)} rows; cancel ${t.jitterMs.toFixed(0)} ms after a report; ${t.finalVisits} visits)`,
        );
      }
      const t0 = r.trials[0];
      check(`partial cadence [${backend}]`, r.trials.every((t) => t.partials >= 3), `${t0.partials} partials, gaps ${t0.partialGapMs.min.toFixed(0)}-${t0.partialGapMs.max.toFixed(0)} ms`);
    }
    if (only.includes("throughput")) {
      const r = await call(page, "throughput", { backend, network: "g170e-b10c128", visits: 100, positions: backend === "cpu" ? 2 : 5 });
      results.throughput.push(r);
      for (const x of r.results) console.log(`  ${backend} ${x.moves} moves: ${x.visits} visits ${x.ms.toFixed(0)} ms = ${x.visitsPerS.toFixed(0)} visits/s, ${x.rows} evals in ${x.batches} batches (avg ${x.avgBatch.toFixed(1)}) = ${x.evalsPerS.toFixed(0)} evals/s (NN busy ${x.nnBusyEvalsPerS.toFixed(0)}/s), top ${x.top}`);
    }
    const logs = await call(page, "takeLogs", null);
    for (const l of logs) console.log(`  [engine] ${l}`);
    await context.close();
  }
} finally {
  await gpuBrowser.close();
}

if (only.includes("fallback")) {
  // No WebGPU at all: `backend: "auto"` must land on the CPU and still match the goldens.
  const noGpu = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: ["--disable-gpu", "--disable-features=WebGPU"] });
  try {
    const { context, page } = await openPage(noGpu);
    const ready = await call(page, "loadModel", { backend: "auto", network: "g170-b6c96" });
    results.fallback = ready;
    check("no-GPU browser: auto falls back to the CPU", ready.backend === "cpu" && !!ready.fallbackReason, `backend ${ready.backend}: ${ready.fallbackReason}`);
    await goldens(page, "auto", "no-GPU auto", goldenFilter ?? ["basic", "basic_sidetomove"]);
    await context.close();
  } finally {
    await noGpu.close();
  }
}

if (only.includes("fallback")) {
  // WebGPU failing before the engine is ready (seen in M4 development: a kernel that SwiftShader could not
  // compile trapped the wasm in its first batch): `auto` restarts the worker on the CPU. Simulated by a
  // worker that fails unless asked for the CPU (test/page/failing-worker.js).
  const browser = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: WEBGPU_ARGS });
  try {
    const { context, page } = await openPage(browser);
    const ready = await call(page, "loadModel", { backend: "auto", network: "g170-b6c96", workerUrl: "/test/failing-worker.js" });
    results.fallbackAfterFailure = ready;
    check("WebGPU failing before ready: auto restarts on the CPU", ready.backend === "cpu" && /simulated WebGPU failure/.test(ready.fallbackReason ?? ""), `backend ${ready.backend}: ${ready.fallbackReason}`);
    await context.close();
  } finally {
    await browser.close();
  }
}

server.close();
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 2));
console.log(`\n${results.checks.length - failures}/${results.checks.length} checks passed`);
process.exit(failures ? 1 : 0);

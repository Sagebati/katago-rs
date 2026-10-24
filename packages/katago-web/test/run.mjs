// Browser tests of katago-web (`just web-test`): a static server for the built package, the upstream query
// files / config and the goldens, and Playwright driving headless system Chromium.
//
//   node packages/katago-web/test/run.mjs [--only goldens,cache,kernels,terminate,partials,throughput,fallback,b18]
//        [--backends webgpu,cpu] [--goldens basic,symmetry] [--json out.json] [--headed]
//
// Env: PLAYWRIGHT (path of the playwright package; default: sai-frontend's node_modules), CHROMIUM
// (default /usr/bin/chromium). WebGPU on Linux needs --enable-unsafe-webgpu --enable-features=Vulkan
// --use-angle=vulkan (docs/M0-report.md, docs/M4-report.md). Exit code 1 if any check fails.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
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
// `b18` (kata1-b18c384nbt: goldens, 105 MB weights, CPU guard, throughput; `just web-test-b18`) is not in the default set.
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
// kata1-b18c384nbt goldens: fixtures/upstream-v1.16.5/analysis/regen_b18.sh (Eigen reference; upstream has none for this net).
const B18 = "kata1-b18c384nbt";
const B18_FILE = "models/kata1-b18c384nbt.bin.gz";
const B18_GOLDENS = [
  ["b18_basic", "b18_basic.stdout", "basic.txt", LOOP],
  ["b18_symmetry", "b18_symmetry.stdout", "symmetry.txt", LOOP],
  ["b18_symmetry_with_pruning", "b18_symmetry_with_pruning.stdout", "symmetry.txt", "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=true"],
];
// The same on a device limited to WebGPU's default buffer limits (test/page/lowlimits-worker.js), batches of at most 32
// (80 MB im2col buffers; 64 rows would need 160 MB, more than the default 128 MiB binding).
const B18_GOLDENS_LOW = [["b18_basic_lowlimits", "b18_basic.stdout", "basic.txt", `${LOOP}, nnMaxBatchSize=32`]];
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

/** VRAM in use per DRM card (amdgpu's mem_info_vram_used), in bytes; {} where the kernel does not say. */
function vramUsed() {
  const out = {};
  try {
    for (const card of readdirSync("/sys/class/drm")) {
      if (!/^card\d+$/.test(card)) continue;
      try {
        out[card] = Number(readFileSync(`/sys/class/drm/${card}/device/mem_info_vram_used`, "utf8"));
      } catch {
        /* not an amdgpu card */
      }
    }
  } catch {
    /* no sysfs */
  }
  return out;
}
const vramDelta = (a, b) => Object.fromEntries(Object.keys(b).map((k) => [k, Math.round(((b[k] ?? 0) - (a[k] ?? 0)) / 1048576)]));

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

async function goldens(page, backend, label, filter = goldenFilter, list = GOLDENS, net = { network: "g170-b6c96", modelFile: M6 }) {
  for (const [name, golden, query, extra] of list) {
    if (filter && !filter.includes(name)) continue;
    const r = await call(page, "runGolden", { backend, queryFile: query, overrides: `${COMMON}, ${extra}`, ...net });
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

  if (only.includes("b18") && backends.includes("webgpu")) {
    const net = { network: B18, modelFile: B18_FILE };
    results.b18 = {};
    {
      // Fresh context (empty Cache API): the 105 MB of weights come from the network, then from the cache.
      const { context, page, browserRequests } = await openPage(gpuBrowser);
      const bpkRequests = () => requestLog.filter((p) => p.endsWith(".bpk")).length;
      const vram0 = vramUsed();
      const before = bpkRequests();
      const cold = await call(page, "loadModel", { backend: "webgpu", network: B18 });
      const afterCold = bpkRequests();
      const vram1 = vramUsed();
      const storage = await call(page, "storageAndLimits", null);
      browserRequests.length = 0;
      await page.reload();
      await page.waitForFunction(() => window.kgwReady === true);
      const warm = await call(page, "loadModel", { backend: "webgpu", network: B18 });
      const afterWarm = bpkRequests();
      const browserBpk = browserRequests.filter((u) => u.endsWith(".bpk"));
      results.b18.load = { cold, warm, storage, vramDeltaMiB: vramDelta(vram0, vram1) };
      check("b18: first load fetches the weights once", cold.backend === "webgpu" && cold.modelSource === "network" && afterCold - before === 1, `${afterCold - before} .bpk request(s), source ${cold.modelSource}, adapter ${cold.adapter}`);
      check("b18: second load does no model network request", warm.modelSource === "cache" && afterWarm === afterCold && browserBpk.length === 0, `server saw ${afterWarm - afterCold}, browser issued ${browserBpk.length}, source ${warm.modelSource}`);
      check("b18: second load compiles no WebGPU kernel", cold.kernelCache?.compiled > 0 && warm.kernelCache?.compiled === 0 && warm.kernelCache?.hits > 0, `cold ${JSON.stringify(cold.kernelCache)}, warm ${JSON.stringify(warm.kernelCache)}`);
      // The biggest buffer of a b18c384nbt forward pass is the im2col matrix of its 192-channel 3x3 convs (62 of them):
      // batch * 361 * 1728 floats = 2.5 MB per row, 80 MB at 32 rows. WebGPU's default maxStorageBufferBindingSize is
      // 128 MiB (53 rows; 64 rows fail on a default-limit device, see lowlimits-worker.js), so the engine relies on the
      // adapter's own, larger limit being passed on to the device.
      const im2col = 32 * 361 * 192 * 9 * 4;
      check("b18: adapter limits cover the biggest buffer", cold.adapterLimits && cold.adapterLimits.maxStorageBufferBindingSize >= im2col && cold.adapterLimits.maxBufferSize >= im2col, `maxStorageBufferBindingSize ${cold.adapterLimits?.maxStorageBufferBindingSize}, maxBufferSize ${cold.adapterLimits?.maxBufferSize}; im2col at batch 32 ${im2col}`);
      console.log(`  cold: ${JSON.stringify(cold.timings)}\n  warm: ${JSON.stringify(warm.timings)}\n  wasm memory ${(cold.wasmMemoryBytes / 1048576).toFixed(0)} MiB after the load\n  storage ${JSON.stringify(storage)}; VRAM +${JSON.stringify(vramDelta(vram0, vram1))} MiB`);
      await context.close();
    }
    {
      const { context, page } = await openPage(gpuBrowser);
      await goldens(page, "webgpu", "b18 webgpu", goldenFilter, B18_GOLDENS, net);
      await goldens(page, "webgpu", "b18 webgpu, default buffer limits", goldenFilter, B18_GOLDENS_LOW, { ...net, workerUrl: "/test/lowlimits-worker.js" });
      const r = await call(page, "apiSmoke", { backend: "webgpu", network: B18 });
      const partials = r.responses.filter((x) => x.isDuringSearch);
      check("b18: reportDuringSearchEvery partials arrive", partials.length >= 2 && r.responses.every((x) => x.id === "mine" && x.ownership === 361), `${partials.length} partials (visits ${partials.map((p) => p.visits).join(",")})`);
      for (const visits of [100, 400]) {
        const t = await call(page, "throughput", { backend: "webgpu", network: B18, visits, positions: 4 });
        (results.b18.throughput ??= []).push({ visits, ...t });
        for (const x of t.results) console.log(`  b18 ${visits}: ${x.moves} moves: ${x.visits} visits ${x.ms.toFixed(0)} ms = ${x.visitsPerS.toFixed(0)} visits/s, ${x.rows} evals in ${x.batches} batches (avg ${x.avgBatch.toFixed(1)}) = ${x.evalsPerS.toFixed(0)} evals/s (NN busy ${x.nnBusyEvalsPerS.toFixed(0)}/s)`);
        check(`b18: ${visits}-visit searches complete`, t.results.every((x) => x.visits >= visits), t.results.map((x) => x.visits).join(","));
      }
      const t = await call(page, "terminateLatency", { backend: "webgpu", network: B18, partialsBeforeCancel: 3, every: 0.5, trials: 5 });
      results.b18.terminate = t;
      for (const [i, x] of t.trials.entries()) {
        check(`b18: terminate returns within one NN batch #${i + 1}`, x.finalIsDuringSearch === false && x.rowsAfterTerminate <= 16 && x.terminateLatencyMs <= 1.5 * x.avgBatchMs + 25, `${x.terminateLatencyMs.toFixed(1)} ms after cancel(), ${x.rowsAfterTerminate} evals in ${x.batchesAfterTerminate} batch(es) (avg batch ${x.avgBatchMs.toFixed(1)} ms of ${x.avgBatchRows.toFixed(1)} rows)`);
      }
      for (const l of await call(page, "takeLogs", null)) console.log(`  [engine] ${l}`);
      await context.close();
    }
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


if (only.includes("b18")) {
  // No WebGPU: b18c384nbt is refused on the CPU (it would run at ~2 evals/s), replaced by `cpuModel`, or run anyway.
  const noGpu = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: ["--disable-gpu", "--disable-features=WebGPU"] });
  try {
    const { context, page } = await openPage(noGpu);
    const before = requestLog.filter((p) => p.endsWith(".bpk")).map((p) => p);
    const b18Before = requestLog.filter((p) => p.includes("kata1-b18c384nbt") && p.endsWith(".bpk")).length;
    const refused = await call(page, "tryLoadModel", { backend: "auto", network: B18 });
    check("b18: refused on the CPU without cpuModel", /needs WebGPU/.test(refused.error ?? "") && /cpuModel/.test(refused.error ?? "") && !/WebGPU engine failed/.test(refused.error ?? ""), refused.error ?? JSON.stringify(refused.ready?.backend));
    check("b18: refusal fetches no weights", requestLog.filter((p) => p.endsWith(".bpk")).length === before.length, `${requestLog.filter((p) => p.endsWith(".bpk")).length - before.length} .bpk request(s)`);
    const cpuForced = await call(page, "tryLoadModel", { backend: "cpu", network: B18 });
    check("b18: refused with backend cpu too", /needs WebGPU/.test(cpuForced.error ?? ""), cpuForced.error);
    const swapped = await call(page, "tryLoadModel", { backend: "auto", network: B18, cpuNetwork: "g170e-b10c128" });
    results.b18 ??= {};
    results.b18.cpuModel = swapped;
    const b18Fetched = requestLog.filter((p) => p.includes("kata1-b18c384nbt") && p.endsWith(".bpk")).length - b18Before;
    check("b18: auto without WebGPU picks cpuModel (b10c128)", swapped.ready?.backend === "cpu" && swapped.ready.network === "g170e-b10c128" && swapped.ready.requestedNetwork === B18 && !!swapped.ready.fallbackReason && b18Fetched === 0, `${swapped.error ?? `backend ${swapped.ready.backend}, network ${swapped.ready.network} (asked ${swapped.ready.requestedNetwork}), ${b18Fetched} b18 download(s), ready in ${swapped.ready.timings.totalMs.toFixed(0)} ms`}`);
    const slow = await call(page, "slowCpuSearch", { network: B18, visits: 4 });
    results.b18.slowCpu = slow;
    check("b18: allowSlowCpu runs it on the CPU", slow.ready.backend === "cpu" && slow.ready.network === B18 && slow.visits >= 4, `ready in ${slow.ready.timings.totalMs.toFixed(0)} ms (weights ${slow.ready.modelSource}, engine start ${slow.ready.timings.engineStartMs.toFixed(0)} ms), ${slow.visits} visits in ${slow.ms.toFixed(0)} ms = ${((slow.rows * 1000) / slow.ms).toFixed(2)} evals/s`);
    for (const l of await call(page, "takeLogs", null)) console.log(`  [engine] ${l}`);
    await context.close();
  } finally {
    await noGpu.close();
  }
}

server.close();
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 2));
console.log(`\n${results.checks.length - failures}/${results.checks.length} checks passed`);
process.exit(failures ? 1 : 0);

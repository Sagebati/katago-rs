// Browser tests of katago-web (`just web-test`): a static server for the built package, the upstream query
// files / config and the goldens, and Playwright driving headless system Chromium.
//
//   node packages/katago-web/test/run.mjs [--only goldens,cache,kernels,terminate,partials,throughput,fallback,review,b18,human]
//        [--backends webgpu,cpu] [--goldens basic,symmetry] [--json out.json] [--headed] [--profile dir]
//
// `review` (M6, full-game review): reviewGame against the native run of the same deterministic search and against
// the katago-ws parity fixtures' payloads, two-pass and moves input, cancel. `review-timing` (not in the default
// set): time per turn at the default settings. Both run in a persistent browser profile (--profile, default
// target/web-test-profile), so after the first run the WebGPU kernels come from the kernel cache.
// `b18` (M7, kata1-b18c384nbt; not in the default set, `just web-test-b18`): the Eigen b18 goldens, the 105 MB
// weights cache, the CPU guard, adapter limits, throughput, terminate.
// `human` (the human SL network as `humanModel`; not in the default set, `just web-test-human`, needs a build with
// net-b18c384nbt-humanv0): reviewGame with a humanProfile against the native run of the same deterministic search, and
// the human network left out on the CPU.
//
// Env: PLAYWRIGHT (path of the playwright package; default: the sibling ../baduk_review checkout's sai-frontend), CHROMIUM
// (default /usr/bin/chromium). WebGPU on Linux needs --enable-unsafe-webgpu --enable-features=Vulkan
// --use-angle=vulkan (docs/M0-report.md, docs/M4-report.md). Exit code 1 if any check fails.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { extname, join, normalize, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { compare, status, selfCheck, TOLERANCES } from "./compare.mjs";
import { comparePayloads, keyShapes, validatePayload } from "./review.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = resolve(here, "..");
const root = resolve(pkg, "../..");
const PLAYWRIGHT = process.env.PLAYWRIGHT ?? fileURLToPath(new URL("../../../../baduk_review/sai-frontend/node_modules/playwright", import.meta.url));
const CHROMIUM = process.env.CHROMIUM ?? "/usr/bin/chromium";
const { chromium } = await import(pathToFileURL(join(PLAYWRIGHT, "index.mjs")).href);

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
// `b18` (kata1-b18c384nbt: goldens, 105 MB weights, CPU guard, throughput; `just web-test-b18`) is not in the default set.
const only = opt("only", "goldens,fallback,cache,kernels,partials,terminate,throughput,review").split(",");
const backends = opt("backends", "webgpu,cpu").split(",");
const headed = args.includes("--headed");
const jsonOut = opt("json", null);
const profileDir = resolve(opt("profile", join(root, "target/web-test-profile")));

// The goldens of crates/katago-cli/tests/analysis_goldens.rs (fixtures/upstream-v1.18.2/analysis/regen.sh),
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
// kata1-b18c384nbt goldens: fixtures/upstream-v1.18.2/analysis/regen_b18.sh (Eigen reference; upstream has none for this net).
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

const results = { origin, checks: [], goldens: [], fallback: null, cache: null, partials: [], terminate: [], throughput: [], review: [], reviewTiming: [] };
let failures = 0;
function check(name, ok, detail) {
  results.checks.push({ name, ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

const readGolden = (f) =>
  readFileSync(join(root, "fixtures/upstream-v1.18.2/analysis", f), "utf8")
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
      const cold = await call(page, "loadModel", { backend: "webgpu", network: B18, precision: "auto" });
      const afterCold = bpkRequests();
      const vram1 = vramUsed();
      const storage = await call(page, "storageAndLimits", null);
      browserRequests.length = 0;
      await page.reload();
      await page.waitForFunction(() => window.kgwReady === true);
      const warm = await call(page, "loadModel", { backend: "webgpu", network: B18, precision: "auto" });
      const afterWarm = bpkRequests();
      const browserBpk = browserRequests.filter((u) => u.endsWith(".bpk"));
      results.b18.load = { cold, warm, storage, vramDeltaMiB: vramDelta(vram0, vram1) };
      check("b18: first load fetches the weights once", cold.backend === "webgpu" && cold.modelSource === "network" && afterCold - before === 1, `${afterCold - before} .bpk request(s), source ${cold.modelSource}, adapter ${cold.adapter}`);
      check("b18: second load does no model network request", warm.modelSource === "cache" && afterWarm === afterCold && browserBpk.length === 0, `server saw ${afterWarm - afterCold}, browser issued ${browserBpk.length}, source ${warm.modelSource}`);
      check("b18: precision auto is f16 on an adapter with shader-f16, f32 without", cold.precision === (storage.features.includes("shader-f16") ? "f16" : "f32"), `precision ${cold.precision}, shader-f16 ${storage.features.includes("shader-f16")}`);
      check("b18: second load compiles no WebGPU kernel", cold.kernelCache?.compiled > 0 && warm.kernelCache?.compiled === 0 && warm.kernelCache?.hits > 0, `cold ${JSON.stringify(cold.kernelCache)}, warm ${JSON.stringify(warm.kernelCache)}`);
      // Before M9 the biggest buffer of a b18c384nbt forward pass was the im2col matrix of its 192-channel 3x3 convs:
      // batch * 361 * 1728 floats = 2.5 MB per row, 80 MB at 32 rows. The Winograd convs (M9) need less (the input
      // conv's GEMM output, ~1.4 MB per row); the check keeps the old, larger bound. WebGPU's default
      // maxStorageBufferBindingSize is 128 MiB, so the engine relies on the adapter's own, larger limit.
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
      for (const precision of ["f32", "f16"]) {
        for (const visits of [100, 400]) {
          const t = await call(page, "throughput", { backend: "webgpu", network: B18, visits, positions: 4, precision });
          (results.b18.throughput ??= []).push({ visits, precision: t.ready.precision, ...t });
          for (const x of t.results) console.log(`  b18 ${t.ready.precision} ${visits}: ${x.moves} moves: ${x.visits} visits ${x.ms.toFixed(0)} ms = ${x.visitsPerS.toFixed(0)} visits/s, ${x.rows} evals in ${x.batches} batches (avg ${x.avgBatch.toFixed(1)}) = ${x.evalsPerS.toFixed(0)} evals/s (NN busy ${x.nnBusyEvalsPerS.toFixed(0)}/s)`);
          check(`b18: ${t.ready.precision} ${visits}-visit searches complete`, t.results.every((x) => x.visits >= visits), t.results.map((x) => x.visits).join(","));
        }
      }
      // f16 vs f32 raw outputs, bounded by upstream's own FP16 error (katago testgpuerror, OpenCL FP16 vs FP32 on an
      // RX 6800 XT, b18c384nbt, max over 2247 positions: winrate 4.45%, lead 2.27, ownership 9.23c, top policy 5.04%).
      {
        const a = await call(page, "nnOutputs", { backend: "webgpu", network: B18, precision: "f32", positions: 12 });
        const b = await call(page, "nnOutputs", { backend: "webgpu", network: B18, precision: "f16", positions: 12 });
        // Policy: skip illegal moves (-1 on both sides); ownership is in [-1, 1], every point counts.
        const maxAbs = (xs, ys, legal) => xs.reduce((m, x, i) => (!legal || (x >= 0 && ys[i] >= 0) ? Math.max(m, Math.abs(x - ys[i])) : m), 0);
        let wr = 0, lead = 0, own = 0, pol = 0;
        for (const [i, x] of a.out.entries()) {
          const y = b.out[i];
          wr = Math.max(wr, Math.abs(x.winrate - y.winrate));
          lead = Math.max(lead, Math.abs(x.scoreLead - y.scoreLead));
          own = Math.max(own, maxAbs(x.ownership, y.ownership, false));
          pol = Math.max(pol, maxAbs(x.policy, y.policy, true));
        }
        results.b18.f16 = { wr, lead, own, pol };
        check("b18: f16 outputs within upstream's FP16 error of f32", a.precision === "f32" && b.precision === "f16" && wr <= 0.0445 && lead <= 2.27 && own <= 0.0923 && pol <= 0.0504, `precisions ${a.precision}/${b.precision}, 12 positions, max |f16 - f32|: winrate ${wr.toExponential(2)}, lead ${lead.toExponential(2)}, ownership ${own.toExponential(2)}, policy ${pol.toExponential(2)}`);
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

//------------------------------------------------------------------------------------------------
// Full-game review (M6)

// Same browser flags, persistent profile: the kernel cache (Cache API) and Chrome's GPU cache survive runs.
async function openPersistent() {
  const context = await chromium.launchPersistentContext(profileDir, { executablePath: CHROMIUM, headless: !headed, args: WEBGPU_ARGS });
  const page = context.pages()[0] ?? (await context.newPage());
  page.on("console", (m) => {
    if (process.env.KGW_ALL_CONSOLE || m.type() === "error" || m.type() === "warning") console.log(`  [page ${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`  [pageerror] ${e}`));
  await page.goto(`${origin}/test/index.html`);
  await page.waitForFunction(() => window.kgwReady === true);
  return { context, page };
}

const FIX = join(root, "fixtures/katago-annotate");
const fixture = (name) => {
  const c = JSON.parse(readFileSync(join(FIX, name, "case.json"), "utf8"));
  const expected = JSON.parse(gunzipSync(readFileSync(join(FIX, name, "expected.json.gz"))).toString());
  // The wasm sets these itself (its network, no human SL model).
  const { model, humanModel, ...options } = c.options;
  return { sgf: readFileSync(join(FIX, name, "game.sgf"), "utf8"), options: { ...options, humanProfile: c.humanProfile }, expected, moves: expected.moves.map((m) => [m.color, m.mv]) };
};
// The deterministic search of fixtures/katago-annotate/browser/README.md (same overrides natively).
const DETERMINISTIC = "logDir=, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnMaxBatchSize=8, nnRandomize=false, nnRandSeed=review, forDeterministicTesting=true";
const turnGaps = (progress) => progress.map((p, i) => p.t - (i === 0 ? 0 : progress[i - 1].t));
const fmtStats = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return `mean ${mean.toFixed(0)} ms, median ${s[Math.floor(s.length / 2)].toFixed(0)}, min ${s[0].toFixed(0)}, max ${s[s.length - 1].toFixed(0)}`;
};

if (only.includes("review")) {
  const nine = fixture("hand-9x9");
  const native = JSON.parse(readFileSync(join(FIX, "browser/hand-9x9.b6c96-deterministic.json"), "utf8"));
  const eigen = JSON.parse(readFileSync(join(FIX, "browser/hand-9x9.b6c96-deterministic.eigen.json"), "utf8"));
  const thirteen = fixture("hand-13x13-h2");
  const full = fixture("ogs-60000000");
  const { context, page } = await openPersistent();
  try {
    for (const backend of backends) {
      // 1. The deterministic search of the native reference (katago-rs, burn flex): same moves, candidates, visits
      //    and PVs; floats within the NN tier's tolerances. And the payload contract, and katago-ws's key paths.
      const r = await call(page, "review", { backend, network: "g170-b6c96", configUrl: "/upstream/configs/analysis_example.cfg", overrides: [DETERMINISTIC], warmupVisits: 0, reviews: [{ game: nine.sgf, options: nine.options }] });
      const rv = r.reviews[0];
      results.review.push({ backend: r.backend, case: "hand-9x9 deterministic", ms: rv.ms, error: rv.error, progress: rv.progress.length });
      if (!rv.json) {
        check(`review: deterministic 9x9 [${backend}]`, false, rv.error);
        continue;
      }
      const p = JSON.parse(rv.json);
      const bad = validatePayload(p, { boardSize: 9, moves: nine.moves, maxVisits: 50, engineName: "katago-rs-wasm", model: "g170-b6c96" });
      check(`review: payload valid [${backend}]`, bad.length === 0 && p.engine.version === "1.16.5", bad.slice(0, 5).join("; ") || `engine ${JSON.stringify(p.engine)}`);
      const tiers = { cpu: { value: 1e-5, score: 5e-4, prior: 1e-5, accuracy: 5e-3, grid: 1, policyGrid: 1 }, webgpu: { value: 1e-4, score: 2e-3, prior: 1e-4, accuracy: 5e-2, grid: 1, policyGrid: 2 } };
      const c = comparePayloads(native, p, tiers[r.backend]);
      results.review.at(-1).compare = c;
      check(`review: matches the native run (katago-rs) [${backend}]`, c.structure.length === 0 && c.over.length === 0, `${c.structure.length} structural, ${c.over.length} over tolerance (${[...c.structure, ...c.over].slice(0, 4).join("; ")}); max dev ${JSON.stringify(c.maxDev)}`);
      const ce = comparePayloads(eigen, p, tiers.webgpu);
      results.review.at(-1).compareEigen = ce;
      check(`review: matches upstream KataGo (Eigen) [${backend}]`, ce.structure.length === 0 && ce.over.length === 0, `${ce.structure.length} structural, ${ce.over.length} over the WebGPU tier (${[...ce.structure, ...ce.over].slice(0, 4).join("; ")}); max dev ${JSON.stringify(ce.maxDev)}`);
      const shapes = keyShapes(p);
      const fixtureShapes = keyShapes(nine.expected);
      const onlyHere = [...shapes].filter((k) => !fixtureShapes.has(k));
      const onlyThere = [...fixtureShapes].filter((k) => !shapes.has(k));
      check(`review: katago-ws's key paths [${backend}]`, onlyHere.join() === ".engine.version" && onlyThere.length === 0, `only in the browser's: ${onlyHere.join(", ")}; only in katago-ws's: ${onlyThere.join(", ") || "none"}`);
      const turns = rv.progress.map((x) => x.turnNumber).sort((a, b) => a - b);
      check(`review: one progress event per turn [${backend}]`, turns.length === 24 && turns.every((t, i) => t === i) && rv.progress.every((x) => x.pass === "full" && x.total === 24), `${rv.progress.length} events; ${(rv.ms / 1000).toFixed(1)} s (${fmtStats(turnGaps(rv.progress))} per turn)`);
      check(`review: the engine still answers queries afterwards [${backend}]`, r.after.length === 1 && r.after[0].visits >= 8, JSON.stringify(r.after));
    }
    if (backends.includes("webgpu")) {
      // 2. Default config. Two-pass, the game given as moves (the 13x13 handicap fixture), then a cancel.
      const thirteenInput = { boardSize: 13, komi: thirteen.expected.komi, rules: thirteen.expected.rules, initialStones: [["B", "D10"], ["B", "K4"]], moves: thirteen.moves };
      const r = await call(page, "review", {
        backend: "webgpu",
        network: "g170e-b10c128",
        reviews: [
          { game: thirteenInput, options: thirteen.options },
          { game: full.sgf, options: {}, cancelAfter: 10 },
        ],
      });
      const [two, cancelled] = r.reviews;
      const p = two.json ? JSON.parse(two.json) : null;
      const bad = p ? validatePayload(p, { boardSize: 13, moves: thirteen.moves, maxVisits: 60, engineName: "katago-rs-wasm", model: "g170e-b10c128" }) : [two.error];
      const screen = two.progress.filter((x) => x.pass === "screen").length;
      const refine = two.progress.filter((x) => x.pass === "refine");
      check("review: two-pass, game as moves [webgpu]", bad.length === 0 && p.engine.screen_visits === 10 && p.ownership === undefined && screen === 21 && refine.length > 0 && refine.every((x) => x.total === refine.length && x.passes === 2), `${bad.slice(0, 4).join("; ")} screen ${screen} turns, refine ${refine.length}; ${(two.ms / 1000).toFixed(1)} s`);
      const shapes = keyShapes(p ?? {});
      const fixtureShapes = keyShapes(thirteen.expected);
      const diff = [...[...shapes].filter((k) => !fixtureShapes.has(k)), ...[...fixtureShapes].filter((k) => !shapes.has(k))];
      check("review: two-pass key paths are katago-ws's [webgpu]", diff.join() === ".engine.version", diff.join(", "));
      check("review: cancel [webgpu]", cancelled.cancelled && cancelled.error === "review cancelled" && cancelled.progress.length < 133 && r.after.length === 1, `${cancelled.error}; ${cancelled.progress.length} turns reported; rejected ${(cancelled.ms - cancelled.tCancel).toFixed(0)} ms after cancel()`);
      results.review.push({ backend: "webgpu", case: "13x13 two-pass + cancel", two: { ms: two.ms, screen, refine: refine.length }, cancel: { ms: cancelled.ms, tCancel: cancelled.tCancel, turns: cancelled.progress.length } });
    }
    const logs = await call(page, "takeLogs", null);
    for (const l of logs) console.log(`  [engine] ${l}`);
  } finally {
    await context.close();
  }
}

if (only.includes("review-timing")) {
  // Time per turn at katago-ws's defaults (100 visits, one pass; the browser's default engine config), b10c128:
  // the whole 132-move game on WebGPU (one pass and two-pass, screen 16), its first 10 moves on both backends.
  const full = fixture("ogs-60000000");
  const first10 = { boardSize: 19, komi: full.expected.komi, rules: full.expected.rules, moves: full.moves.slice(0, 10) };
  const { context, page } = await openPersistent();
  try {
    for (const backend of backends) {
      const which = (opt("timing", "first10,full,twopass")).split(",");
      const all = { first10: [{ game: first10, options: {} }, "first 10 moves"], full: [{ game: full.sgf, options: {} }, "132 moves"], twopass: [{ game: full.sgf, options: { screenVisits: 16 } }, "132 moves, two-pass (screen 16)"] };
      const picked = which.filter((w) => backend === "webgpu" || w === "first10");
      const reviews = picked.map((w) => all[w][0]);
      // NN cache cleared before each review: the later ones re-analyse the same positions.
      // --review-overrides: engine config for the timing (e.g. the throughput shape katago-ws uses for games:
      // many positions in parallel, few threads each).
      const ro = opt("review-overrides", null);
      const r = await call(page, "review", { backend, network: "g170e-b10c128", reviews, clearCache: true, overrides: ro ? [ro] : undefined });
      for (const [i, rv] of r.reviews.entries()) {
        const label = all[picked[i]][1];
        const turns = rv.progress.length;
        const p = rv.json ? JSON.parse(rv.json) : null;
        results.reviewTiming.push({ backend: r.backend, label, ms: rv.ms, turns, gaps: turnGaps(rv.progress), error: rv.error, ready: r.ready.timings, nn: rv.nn });
        const nn = rv.nn;
        console.log(`  NN: ${nn.rows} evals in ${nn.batches} batches (avg ${(nn.rows / nn.batches).toFixed(1)} rows, ${(nn.busyMs / nn.batches).toFixed(0)} ms, max ${nn.maxBatchMs.toFixed(0)} ms), ${((nn.rows * 1000) / rv.ms).toFixed(0)} evals/s, NN busy ${((100 * nn.busyMs) / rv.ms).toFixed(0)}% of the time`);
        const bad = p ? validatePayload(p, { maxVisits: 100 }) : [rv.error];
        const ok = bad.length === 0;
        check(`review timing: ${label} [${backend}]`, !!ok, `${(rv.ms / 1000).toFixed(1)} s for ${turns} analysed turns = ${(rv.ms / turns).toFixed(0)} ms per turn (gaps ${fmtStats(turnGaps(rv.progress))}); ready in ${r.ready.timings.totalMs.toFixed(0)} ms${bad.length ? `; ${bad.slice(0, 3).join("; ")}` : ""}`);
      }
    }
  } finally {
    await context.close();
  }
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

if (only.includes("human")) {
  const HUMAN = "b18c384nbt-humanv0";
  const nine = fixture("hand-9x9");
  const native = JSON.parse(readFileSync(join(FIX, "browser/hand-9x9.b6c96-human-deterministic.json"), "utf8"));
  const expect = { boardSize: 9, moves: nine.moves, maxVisits: 50, engineName: "katago-rs-wasm", model: "g170-b6c96" };
  // The human policy comes with includePolicy (fixtures/katago-annotate/browser/README.md).
  const humanOptions = { ...nine.options, includePolicy: true };
  if (backends.includes("webgpu")) {
    const browser = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: WEBGPU_ARGS });
    try {
      const { context, page } = await openPage(browser);
      // The deterministic search of the native reference, with the human SL network beside g170-b6c96.
      const r = await call(page, "review", { backend: "webgpu", network: "g170-b6c96", humanNetwork: HUMAN, configUrl: "/upstream/configs/analysis_example.cfg", overrides: [DETERMINISTIC], warmupVisits: 0, reviews: [{ game: nine.sgf, options: humanOptions }] });
      const rv = r.reviews[0];
      check("human: humanModel is loaded on WebGPU", r.ready.humanNetwork === HUMAN, `ready.humanNetwork ${r.ready.humanNetwork}`);
      if (!rv.json) check("human: review with rank_5k", false, rv.error);
      else {
        const p = JSON.parse(rv.json);
        const bad = validatePayload(p, { ...expect, humanProfile: "rank_5k", humanModel: HUMAN });
        check("human: payload has the profile's human policy", bad.length === 0, bad.slice(0, 5).join("; ") || `human_policy ${p.human_policy.length} turns, engine ${JSON.stringify(p.engine)}`);
        const c = comparePayloads(native, p, { value: 1e-4, score: 2e-3, prior: 1e-4, accuracy: 5e-2, grid: 1, policyGrid: 2 });
        check("human: matches the native run (katago-rs, -human-model)", c.structure.length === 0 && c.over.length === 0, `${c.structure.length} structural, ${c.over.length} over tolerance (${[...c.structure, ...c.over].slice(0, 4).join("; ")}); max dev ${JSON.stringify(c.maxDev)}`);
      }
      for (const l of await call(page, "takeLogs", null)) console.log(`  [engine] ${l}`);
      await context.close();
    } finally {
      await browser.close();
    }
  }
  // No WebGPU: the human network (b18-sized) is not loaded, nor fetched, and the profile is dropped as before.
  const noGpu = await chromium.launch({ executablePath: CHROMIUM, headless: !headed, args: ["--disable-gpu", "--disable-features=WebGPU"] });
  try {
    const { context, page } = await openPage(noGpu);
    const before = requestLog.filter((p) => p.includes(HUMAN)).length;
    const r = await call(page, "review", { backend: "cpu", network: "g170-b6c96", humanNetwork: HUMAN, configUrl: "/upstream/configs/analysis_example.cfg", overrides: [DETERMINISTIC], warmupVisits: 0, reviews: [{ game: nine.sgf, options: humanOptions }] });
    const fetched = requestLog.filter((p) => p.includes(HUMAN)).length - before;
    check("human: not loaded on the CPU (nor fetched)", r.ready.humanNetwork === undefined && fetched === 0, `ready.humanNetwork ${r.ready.humanNetwork}, ${fetched} download(s)`);
    const rv = r.reviews[0];
    const bad = rv.json ? validatePayload(JSON.parse(rv.json), expect) : [rv.error];
    check("human: without it the profile is dropped", bad.length === 0, bad.slice(0, 5).join("; "));
    await context.close();
  } finally {
    await noGpu.close();
  }
}

server.close();
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 2));
console.log(`\n${results.checks.length - failures}/${results.checks.length} checks passed`);
process.exit(failures ? 1 : 0);

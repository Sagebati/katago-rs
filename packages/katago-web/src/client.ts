// KataGoClient: the main-thread side. Starts the engine worker, assigns query ids, routes response lines
// to the query that asked for them, and turns them into async iterables.

import type { AnalysisQuery, ErrorResponse, ModelsResponse, TurnResponse, VersionResponse, WarningResponse } from "./protocol.js";
import type { BackendChoice, EngineInitOptions, EngineStats, FromWorker, ReadyInfo, ToWorker } from "./messages.js";
import type { GameAnalysis, GameInput, ReviewOptions, ReviewProgress } from "./review.js";

export interface KataGoClientOptions extends Omit<EngineInitOptions, "overrides"> {
  /** `-override-config`: "key=value, ..." string(s), or an object of key/values. */
  overrides?: string | string[] | Record<string, string | number | boolean>;
  /** The engine worker, or a factory for it (bundlers that need `new Worker(new URL(...))` in app code).
   * Default: a module worker of the `worker.js` next to this file (not followed by bundlers: pass this). */
  worker?: Worker | (() => Worker);
  /** URL of worker.js when the default location does not work. */
  workerUrl?: string | URL;
  /** Log lines of the engine ("error" | "warn" | "info") and its stderr ("stderr": unused config keys, "Started,
   * ready to begin handling requests"...). Default: the console (info lines at debug level). */
  onLog?: (kind: string, text: string) => void;
  onProgress?: (progress: Extract<FromWorker, { type: "progress" }>) => void;
}

/** An `{"error": ...}` response (or an engine failure). */
export class KataGoError extends Error {
  readonly response?: ErrorResponse;
  /** A review stopped by `ReviewRun.cancel()` (or its AbortSignal). */
  cancelled = false;
  constructor(message: string, response?: ErrorResponse) {
    super(message);
    this.name = "KataGoError";
    this.response = response;
  }
}

/** The default worker script, relative to this module. A variable rather than a literal in the
 * `new URL(..., import.meta.url)` below: bundlers (Vite, webpack) emit any literal one into the build, even
 * when the caller passes `worker` and it is never used, and that copy is not bundled (its imports break). */
const DEFAULT_WORKER_SCRIPT = "./worker.js";

function defaultWorker(url: string | URL | undefined): Worker {
  return new Worker(url ?? new URL(DEFAULT_WORKER_SCRIPT, import.meta.url), { type: "module", name: "katago-web" });
}

function overridesToStrings(o: KataGoClientOptions["overrides"]): string[] | undefined {
  if (o === undefined) return undefined;
  if (typeof o === "string") return [o];
  if (Array.isArray(o)) return o;
  return [
    Object.entries(o)
      .map(([k, v]) => `${k}=${String(v)}`)
      .join(", "),
  ];
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve(v: T): void;
  reject(e: unknown): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  // A rejection nobody awaits is not an unhandled-rejection report; awaiting it still throws.
  promise.catch(() => {});
  return { promise, resolve, reject };
}

/** One analysis query in flight: an async iterable of its responses (partials with `isDuringSearch: true`,
 * then one final per analyzed turn), ending after the last final. */
export class AnalysisRun implements AsyncIterable<TurnResponse> {
  /** The query's id as the caller gave it (or the generated one). Responses carry this id. */
  readonly id: string;
  /** The id on the wire (unique per client). */
  readonly wireId: string;
  /** Warnings the engine reported about the query's fields. */
  readonly warnings: WarningResponse[] = [];
  /** The final response of every analyzed turn, in arrival order. */
  readonly done: Promise<TurnResponse[]>;

  private readonly client: KataGoClient;
  private readonly expectedFinals: number;
  private readonly finalTurns = new Set<number>();
  private readonly finals: TurnResponse[] = [];
  private readonly queue: TurnResponse[] = [];
  private waiter: Deferred<void> | null = null;
  private finished = false;
  private error: unknown = null;
  private readonly doneDeferred = deferred<TurnResponse[]>();

  /** @internal */
  constructor(client: KataGoClient, wireId: string, id: string, expectedFinals: number) {
    this.client = client;
    this.wireId = wireId;
    this.id = id;
    this.expectedFinals = expectedFinals;
    this.done = this.doneDeferred.promise;
    if (expectedFinals === 0) this.finish();
  }

  /** Terminates the query (`terminate` action): its searches stop after the NN batch in flight and report
   * what they have, so the iteration still ends with one final per turn. */
  cancel(): Promise<void> {
    if (this.finished) return Promise.resolve();
    return this.client.cancel(this);
  }

  get isFinished(): boolean {
    return this.finished;
  }

  /** @internal */
  push(r: Record<string, unknown>): void {
    if (this.finished) return;
    if (typeof r.error === "string") {
      this.fail(new KataGoError(`${r.error}${r.field ? ` (field ${String(r.field)})` : ""}`, { ...(r as unknown as ErrorResponse), id: this.id }));
      return;
    }
    if (typeof r.warning === "string") {
      this.warnings.push({ ...(r as unknown as WarningResponse), id: this.id });
      return;
    }
    const t = { ...(r as unknown as TurnResponse), id: this.id };
    this.queue.push(t);
    if (!t.isDuringSearch) {
      this.finals.push(t);
      this.finalTurns.add(t.turnNumber);
      if (this.finalTurns.size >= this.expectedFinals) this.finish();
    }
    this.wake();
  }

  /** @internal */
  fail(e: unknown): void {
    if (this.finished) return;
    this.error = e;
    this.finished = true;
    this.client._forget(this);
    this.doneDeferred.reject(e);
    this.wake();
  }

  private finish(): void {
    this.finished = true;
    this.client._forget(this);
    this.doneDeferred.resolve(this.finals);
    this.wake();
  }

  private wake(): void {
    const w = this.waiter;
    this.waiter = null;
    w?.resolve();
  }

  [Symbol.asyncIterator](): AsyncIterator<TurnResponse> {
    return {
      next: async (): Promise<IteratorResult<TurnResponse>> => {
        for (;;) {
          const r = this.queue.shift();
          if (r) return { value: r, done: false };
          if (this.error) throw this.error;
          if (this.finished) return { value: undefined, done: true };
          this.waiter = deferred<void>();
          await this.waiter.promise;
        }
      },
      // `break` out of a for-await: stop the search too.
      return: async (): Promise<IteratorResult<TurnResponse>> => {
        if (!this.finished) void this.cancel();
        return { value: undefined, done: true };
      },
    };
  }
}

/** A full-game review in flight (`KataGoClient.reviewGame`). */
export class ReviewRun {
  /** The payload v3 (katago-ws's `GameAnalysis`); rejects with a KataGoError (`cancelled` set on cancel). */
  readonly result: Promise<GameAnalysis>;
  /** The same payload as the JSON text the engine produced (what an upload should send, byte for byte). */
  readonly json: Promise<string>;
  /** @internal */
  readonly requestId: number;
  /** @internal */
  readonly onProgress?: (p: ReviewProgress) => void;
  private readonly client: KataGoClient;
  private readonly jsonDeferred = deferred<string>();
  private settled = false;

  /** @internal */
  constructor(client: KataGoClient, requestId: number, onProgress?: (p: ReviewProgress) => void) {
    this.client = client;
    this.requestId = requestId;
    this.onProgress = onProgress;
    this.json = this.jsonDeferred.promise;
    this.result = this.json.then((text) => JSON.parse(text) as GameAnalysis);
    this.result.catch(() => {});
  }

  /** Stops the review: the query in flight is terminated (within one NN batch), no further pass starts, and
   * `result` rejects with a KataGoError whose `cancelled` is true. */
  cancel(): void {
    if (!this.settled) this.client._cancelReview(this);
  }

  /** @internal */
  _settle(json: string | null, error?: Error): void {
    if (this.settled) return;
    this.settled = true;
    if (error) this.jsonDeferred.reject(error);
    else this.jsonDeferred.resolve(json!);
  }
}

/** The engine in a Web Worker, speaking KataGo's analysis protocol. */
export class KataGoClient {
  /** Resolves when the network is loaded (and warmed up): which backend runs, versions, timings. */
  readonly ready: Promise<ReadyInfo>;

  private worker!: Worker;
  private readonly options: KataGoClientOptions;
  private readyDeferred = deferred<ReadyInfo>();
  private isReady = false;
  private retriedOnCpu = false;
  private retryReason: string | undefined;
  private queued: string[] = [];
  private seq = 0;
  private readonly runs = new Map<string, AnalysisRun>();
  private readonly actions = new Map<string, Deferred<Record<string, unknown>>>();
  private readonly statsWaiters = new Map<number, Deferred<EngineStats>>();
  private readonly reviews = new Map<number, ReviewRun>();
  /** Reviews asked for before `ready`: posted to the worker that becomes ready (a CPU restart included). */
  private queuedReviews: Extract<ToWorker, { type: "review" }>[] = [];
  private readonly lineListeners = new Set<(line: string) => void>();
  private dead: Error | null = null;
  /** Set by dispose(): no worker is started (or restarted on the CPU) after it. */
  private disposed = false;
  private exitDeferred: Deferred<number> | null = null;

  constructor(options: KataGoClientOptions) {
    this.options = options;
    this.ready = this.readyDeferred.promise;
    this.startWorker(options.backend ?? "auto");
  }

  private startWorker(backend: BackendChoice): void {
    if (this.disposed) return;
    const w = this.options.worker;
    const worker = typeof w === "function" ? w() : w ?? defaultWorker(this.options.workerUrl);
    this.worker = worker;
    // Events of a replaced worker (the WebGPU one after a CPU restart) or of a disposed client are ignored.
    worker.onmessage = (e: MessageEvent<FromWorker>) => {
      if (worker === this.worker && !this.disposed) this.onMessage(e.data);
    };
    worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      if (worker === this.worker && !this.disposed) this.onFatal(`worker error: ${e.message}`);
    };
    const init: EngineInitOptions = {
      model: this.options.model,
      cpuModel: this.options.cpuModel,
      allowSlowCpu: this.options.allowSlowCpu,
      humanModel: this.options.humanModel,
      backend,
      config: this.options.config,
      overrides: overridesToStrings(this.options.overrides),
      analysisThreads: this.options.analysisThreads,
      modelFile: this.options.modelFile,
      warmupVisits: this.options.warmupVisits,
      cacheModels: this.options.cacheModels,
      cacheKernels: this.options.cacheKernels,
      wasmUrl: this.options.wasmUrl ? String(this.options.wasmUrl) : undefined,
      precision: this.options.precision,
    };
    this.post({ type: "init", options: init });
  }

  private post(m: ToWorker): void {
    this.worker.postMessage(m);
  }

  private onMessage(m: FromWorker): void {
    switch (m.type) {
      case "line":
        this.handleLine(m.line);
        break;
      case "log":
        if (this.options.onLog) this.options.onLog(m.kind, m.text);
        else if (m.kind === "error") console.error(`[katago] ${m.text}`);
        else if (m.kind === "warn") console.warn(`[katago] ${m.text}`);
        else if (m.kind === "stderr") console.info(`[katago] ${m.text.trimEnd()}`);
        else console.debug(`[katago] ${m.text}`);
        break;
      case "progress":
        this.options.onProgress?.(m);
        break;
      case "ready":
        this.isReady = true;
        this.readyDeferred.resolve(this.retryReason && !m.info.fallbackReason ? { ...m.info, fallbackReason: this.retryReason } : m.info);
        if (this.queued.length > 0) {
          const lines = this.queued;
          this.queued = [];
          this.post({ type: "lines", lines });
        }
        for (const r of this.queuedReviews) this.post(r);
        this.queuedReviews = [];
        break;
      case "stats": {
        const d = this.statsWaiters.get(m.requestId);
        this.statsWaiters.delete(m.requestId);
        d?.resolve(m.stats);
        break;
      }
      case "reviewProgress":
        this.reviews.get(m.requestId)?.onProgress?.(m.progress);
        break;
      case "reviewDone": {
        const run = this.reviews.get(m.requestId);
        this.reviews.delete(m.requestId);
        run?._settle(m.payload);
        break;
      }
      case "reviewError": {
        const run = this.reviews.get(m.requestId);
        this.reviews.delete(m.requestId);
        const e = new KataGoError(m.message);
        e.cancelled = m.cancelled;
        run?._settle(null, e);
        break;
      }
      case "exit":
        this.exitDeferred?.resolve(m.code);
        this.onFatal(`engine exited with code ${m.code}`);
        break;
      case "fatal":
        this.onFatal(m.message, m.retry !== false);
        break;
    }
  }

  private onFatal(message: string, retry = true): void {
    // A WebGPU failure before the engine was ready (e.g. a kernel that does not compile on this GPU traps
    // the wasm): start over in a new worker on the CPU when the backend was "auto" (not possible with a
    // Worker instance passed in).
    const canRestart = !(this.options.worker && typeof this.options.worker !== "function");
    if (retry && !this.isReady && !this.dead && !this.disposed && (this.options.backend ?? "auto") === "auto" && !this.retriedOnCpu && canRestart) {
      this.retriedOnCpu = true;
      this.retryReason = `WebGPU engine failed before it was ready: ${message}`;
      this.worker.terminate();
      this.options.onLog?.("warn", `engine failed before it was ready (${message}); retrying on the CPU`);
      this.startWorker("cpu");
      return;
    }
    if (this.dead) return;
    // The CPU restart's own failure (e.g. a GPU-only net refused on the CPU) keeps the WebGPU failure in view.
    if (this.retriedOnCpu && !this.isReady && this.retryReason && !message.includes(this.retryReason)) message = `${message} (${this.retryReason})`;
    this.dead = new KataGoError(message);
    this.exitDeferred?.reject(this.dead);
    this.readyDeferred.reject(this.dead);
    for (const run of [...this.runs.values()]) run.fail(this.dead);
    for (const d of this.actions.values()) d.reject(this.dead);
    this.actions.clear();
    for (const d of this.statsWaiters.values()) d.reject(this.dead);
    this.statsWaiters.clear();
    for (const r of this.reviews.values()) r._settle(null, this.dead);
    this.reviews.clear();
  }

  private handleLine(line: string): void {
    for (const l of this.lineListeners) l(line);
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(line) as Record<string, unknown>;
    } catch {
      this.options.onLog?.("warn", `unparseable engine output: ${line}`);
      return;
    }
    const id = typeof obj.id === "string" ? obj.id : undefined;
    if (id !== undefined) {
      const run = this.runs.get(id);
      if (run) {
        run.push(obj);
        return;
      }
      const action = this.actions.get(id);
      if (action) {
        this.actions.delete(id);
        if (typeof obj.error === "string") action.reject(new KataGoError(obj.error, obj as unknown as ErrorResponse));
        else action.resolve(obj);
        return;
      }
    }
    if (typeof obj.error === "string") this.options.onLog?.("error", `engine error: ${line}`);
  }

  /** @internal */
  _forget(run: AnalysisRun): void {
    if (this.runs.get(run.wireId) === run) this.runs.delete(run.wireId);
  }

  private nextId(kind: string): string {
    return `kgw-${kind}${++this.seq}`;
  }

  private sendLines(lines: string[]): void {
    if (this.dead) throw this.dead;
    if (this.isReady) this.post({ type: "lines", lines });
    else this.queued.push(...lines);
  }

  private action(fields: Record<string, unknown>): Promise<Record<string, unknown>> {
    const id = this.nextId("a");
    const d = deferred<Record<string, unknown>>();
    if (this.dead) return Promise.reject(this.dead);
    this.actions.set(id, d);
    this.sendLines([JSON.stringify({ id, ...fields })]);
    return d.promise;
  }

  /** Starts an analysis query. Iterate the result for its responses (partials when
   * `reportDuringSearchEvery` is set, then one final per analyzed turn). */
  analyze(query: AnalysisQuery): AnalysisRun {
    const wireId = this.nextId("q");
    const turns = query.analyzeTurns ? new Set(query.analyzeTurns).size : 1;
    const run = new AnalysisRun(this, wireId, query.id ?? wireId, turns);
    if (this.dead) {
      run.fail(this.dead);
      return run;
    }
    if (turns > 0) this.runs.set(wireId, run);
    this.sendLines([JSON.stringify({ ...query, id: wireId })]);
    return run;
  }

  /** Terminates one query (`terminate`), or every query when called without one (`terminate_all`).
   * Resolves on the engine's acknowledgement; the terminated runs then end with their partial finals. */
  async cancel(run?: AnalysisRun): Promise<void> {
    if (run) await this.action({ action: "terminate", terminateId: run.wireId });
    else await this.action({ action: "terminate_all" });
  }

  /** Reviews a whole game as katago-ws does (crate katago-annotate, in the wasm): every turn analysed (one
   * query, or a screening pass then a full-depth pass around the moves that swing, with `screenVisits`), then
   * annotated into the payload v3 that baduk_review stores for a server analysis, with
   * `engine = {name: "katago-rs-wasm", version, model}`. `game` is SGF text or a GameInput. `onProgress` runs
   * after each analysed turn. Runs alongside `analyze` queries (the engine interleaves them). */
  reviewGame(game: string | GameInput, options: ReviewOptions = {}, onProgress?: (p: ReviewProgress) => void): ReviewRun {
    const requestId = ++this.seq;
    const run = new ReviewRun(this, requestId, onProgress);
    if (this.dead) {
      run._settle(null, this.dead);
      return run;
    }
    const { signal, ...rest } = options;
    if (signal?.aborted) {
      const e = new KataGoError("review cancelled");
      e.cancelled = true;
      run._settle(null, e);
      return run;
    }
    signal?.addEventListener("abort", () => run.cancel(), { once: true });
    this.reviews.set(requestId, run);
    const m: ToWorker = { type: "review", requestId, game, options: rest };
    if (this.isReady) this.post(m);
    else this.queuedReviews.push(m);
    return run;
  }

  /** @internal */
  _cancelReview(run: ReviewRun): void {
    if (this.reviews.get(run.requestId) !== run || this.dead) return;
    const queued = this.queuedReviews.findIndex((m) => m.requestId === run.requestId);
    if (queued >= 0) {
      this.queuedReviews.splice(queued, 1);
      this.reviews.delete(run.requestId);
      const e = new KataGoError("review cancelled");
      e.cancelled = true;
      run._settle(null, e);
    } else this.post({ type: "reviewCancel", requestId: run.requestId });
  }

  /** `query_version`: the KataGo version the port follows. */
  async queryVersion(): Promise<VersionResponse> {
    return (await this.action({ action: "query_version" })) as unknown as VersionResponse;
  }

  /** `query_models`: the loaded network. */
  async queryModels(): Promise<ModelsResponse> {
    return (await this.action({ action: "query_models" })) as unknown as ModelsResponse;
  }

  /** `clear_cache`: empties the NN cache. */
  async clearCache(): Promise<void> {
    await this.action({ action: "clear_cache" });
  }

  /** NN counters of the engine (batches, rows, time in the backend). */
  stats(): Promise<EngineStats> {
    if (this.dead) return Promise.reject(this.dead);
    const requestId = ++this.seq;
    const d = deferred<EngineStats>();
    this.statsWaiters.set(requestId, d);
    this.post({ type: "stats", requestId });
    return d.promise;
  }

  /** Low level: raw protocol lines, all delivered to the engine at once (like one write to its stdin).
   * Responses to them reach `onLine` listeners only (ids are not rewritten). */
  sendRaw(lines: string[]): void {
    this.sendLines(lines);
  }

  /** Low level: every response line, unparsed. Returns the unsubscribe function. */
  onLine(listener: (line: string) => void): () => void {
    this.lineListeners.add(listener);
    return () => this.lineListeners.delete(listener);
  }

  /** End of input (stdin closed): the engine finishes every query already sent, then exits. Resolves with
   * its exit code; the client is unusable afterwards. */
  closeInput(): Promise<number> {
    if (this.dead) return Promise.reject(this.dead);
    if (!this.exitDeferred) {
      this.exitDeferred = deferred<number>();
      const close = () => this.post({ type: "close" });
      // Queued lines go out with "ready"; close after them.
      if (this.isReady) close();
      else void this.ready.then(close, () => {});
    }
    return this.exitDeferred.promise;
  }

  /** Stops the worker (also one starting up, or restarted on the CPU), for good: pending queries fail,
   * `ready` rejects if it has not resolved, and no worker is started again. */
  dispose(): void {
    this.disposed = true;
    this.worker.terminate();
    this.onFatal("disposed");
  }
}

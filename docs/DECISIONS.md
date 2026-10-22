# Porting decisions log

Decisions taken while preparing the port, each with the default chosen and why. Overturn by
editing the entry and docs/PORTING.md together.

| # | Decision | Default taken | Why | Status |
|---|---|---|---|---|
| D1 | Route | Rust port (burn + wgpu), not Emscripten | M0 retired the import, parity and WebGPU risks ([M0-report](M0-report.md)) | taken 2026-10-22 |
| D2 | Network import | `.bin.gz` → ONNX with our `tools/bin2onnx` → burn-onnx | kaya-go/katago-onnx is AGPL and needs PyTorch checkpoints | taken |
| D3 | Integer widths | Follow upstream: `Loc = i16` (`short`), `Player`/`Color = i8` | Fidelity; the plan's u16/u8 was a guess | taken (PORTING.md) |
| D4 | JSON output | Port nlohmann's serializer (`port::nlohmann::dump`, MIT) for number formatting; serde_json without `preserve_order` for sorted keys | Golden `.stdout` files compare strings; ryu's shortest digits can differ from nlohmann's grisu2 | approved 2026-10-22 |
| D5 | Non-upstream helpers | One directory: `crates/katago-core/src/port/` | Keeps the mirrored tree clean; one place to audit | taken (CLAUDE.md) |
| D6 | NN backend interface | Trait objects injected through constructors (`Rc<dyn NeuralNet>`, `Box<dyn LoadedModel>`); models load from bytes | Mirrors `nninterface.h` function set; wasm has no files | taken, revisit in M2 |
| D7 | Deferred features (eval cache, pattern bonus, anti-mirror) | Fields and call sites kept; their types are empty enums so the fields are always `None` | Keeps search.cpp's shape for later sync, no stubs | taken |
| D8 | NN server start-up | `spawnServerThreads` becomes `async` (WebGPU device creation is async); `killServerThreads` doesn't join | Browser constraint | taken |
| D9 | Analysis oracle | Primary: our Eigen reruns of `tests/analysis/*.txt`; secondary: upstream `tests/results/analysis/*.stdout` (CUDA-made) with float tolerance | Upstream goldens weren't produced on a CPU fp32 backend | taken |
| D10 | Overload naming | Deterministic suffix rule from PORTING.md, with an override table there for the few that get unwieldy (e.g. `runWholeSearch` variants) | Mechanical by default, readable where it matters | taken |
| D11 | Search node storage | `NodeArena` of `Rc<SearchNode>` with `NodeIdx(u32)` handles; `Cell`/`RefCell` for fields | No `unsafe`, single-threaded, keeps pointer-equality semantics via `idx` | taken (OWNERSHIP.tsv) |
| D12 | RNG stream fidelity | Keep RNG draws that only fed mutex selection (`createMutexIdxForNode`) | Removing them would shift the stream and break golden outputs | taken |
| D13 | `atan`/`tan` for `ScoreValue` (M2) | (c): `port::math::{atan, tan}` wrap the platform libm; ≤ 1 ulp from glibc on wasm (6.1% / 3.8% of inputs), exact natively | glibc's are LGPL-2.1+ IBM code; ScoreValue only feeds score utility, compared with a tolerance. Test: `oracle/math_ulp.txt` | decided 2026-10-22 |
| D14 | Fixture host | Generate fixtures on x86-64 **with FMA**: glibc picks `__exp_fma` etc. at runtime, so last-ulp results depend on the CPU; `port::math` reproduces the FMA variants | Found by the pilot fixer | taken |

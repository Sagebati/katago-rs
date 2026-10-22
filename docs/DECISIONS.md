# Porting decisions log

Decisions taken while preparing the port, each with the default chosen and why. Overturn by
editing the entry and docs/PORTING.md together.

| # | Decision | Default taken | Why | Status |
|---|---|---|---|---|
| D1 | Route | Rust port (burn + wgpu), not Emscripten | M0 retired the import, parity and WebGPU risks ([M0-report](M0-report.md)) | taken 2026-10-22 |
| D2 | Network import | `.bin.gz` → ONNX with our `tools/bin2onnx` → burn-onnx | kaya-go/katago-onnx is AGPL and needs PyTorch checkpoints | taken |
| D3 | Integer widths | Follow upstream: `Loc = i16` (`short`), `Player`/`Color = i8` | Fidelity; the plan's u16/u8 was a guess | taken (PORTING.md) |
| D4 | JSON output | Port nlohmann's serializer (`port::nlohmann::dump`, MIT) for number formatting; serde_json without `preserve_order` for sorted keys | Golden `.stdout` files compare strings; ryu's shortest digits can differ from nlohmann's grisu2 | **needs your sign-off** |
| D5 | Non-upstream helpers | One directory: `crates/katago-core/src/port/` | Keeps the mirrored tree clean; one place to audit | taken (CLAUDE.md) |
| D6 | NN backend interface | Trait objects injected through constructors (`Rc<dyn NeuralNet>`, `Box<dyn LoadedModel>`); models load from bytes | Mirrors `nninterface.h` function set; wasm has no files | taken, revisit in M2 |
| D7 | Deferred features (eval cache, pattern bonus, anti-mirror) | Fields and call sites kept; their types are empty enums so the fields are always `None` | Keeps search.cpp's shape for later sync, no stubs | taken |
| D8 | NN server start-up | `spawnServerThreads` becomes `async` (WebGPU device creation is async); `killServerThreads` doesn't join | Browser constraint | taken |
| D9 | Analysis oracle | Primary: our Eigen reruns of `tests/analysis/*.txt`; secondary: upstream `tests/results/analysis/*.stdout` (CUDA-made) with float tolerance | Upstream goldens weren't produced on a CPU fp32 backend | taken |
| D10 | Overload naming | Deterministic suffix rule from PORTING.md, with an override table there for the few that get unwieldy (e.g. `runWholeSearch` variants) | Mechanical by default, readable where it matters | taken |
| D11 | Search node storage | `NodeArena` of `Rc<SearchNode>` with `NodeIdx(u32)` handles; `Cell`/`RefCell` for fields | No `unsafe`, single-threaded, keeps pointer-equality semantics via `idx` | taken (OWNERSHIP.tsv) |
| D12 | RNG stream fidelity | Keep RNG draws that only fed mutex selection (`createMutexIdxForNode`) | Removing them would shift the stream and break golden outputs | taken |
| D13 | `atan`/`tan` for `ScoreValue` (M2) | **Open.** glibc's are IBM code under **LGPL-2.1+**; wasm's libm differs from glibc on 6.1% / 3.8% of inputs. Options: (a) port glibc's anyway and license that one file LGPL, (b) find/derive an MIT correctly-rounded version (CORE-MATH has some functions) and accept differing from glibc where glibc isn't correctly rounded, (c) accept 1-ulp drift in `ScoreValue` (only feeds score utility, tolerance-compared). | Licence choice is yours | **needs your decision before M2** |
| D14 | Fixture host | Generate fixtures on x86-64 **with FMA**: glibc picks `__exp_fma` etc. at runtime, so last-ulp results depend on the CPU; `port::math` reproduces the FMA variants | Found by the pilot fixer | taken |

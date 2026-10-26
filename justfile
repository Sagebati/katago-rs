# katago-rs task runner.

venv := justfile_directory() / ".venv/bin/python"

# Convert a KataGo .bin.gz network to ONNX (models/onnx/<name>.onnx).
convert model name:
    {{venv}} -I tools/bin2onnx/bin2onnx.py {{model}} -o models/onnx/{{name}}.onnx

# Check a converted net against the Eigen KataGo build (build/eigen/katago).
verify model name:
    {{venv}} -I tools/bin2onnx/verify.py --model {{model}} --onnx models/onnx/{{name}}.onnx

# Build KataGo's Eigen (CPU, fp32) backend from the upstream submodule: the reference for fixtures.
katago-eigen:
    test -d build/deps/eigen || git clone -q --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git build/deps/eigen
    cmake -S upstream/KataGo/cpp -B build/eigen -G Ninja -DUSE_BACKEND=EIGEN -DEIGEN3_INCLUDE_DIRS={{justfile_directory()}}/build/deps/eigen -DCMAKE_BUILD_TYPE=Release -DNO_GIT_REVISION=1
    ninja -C build/eigen katago

# The examples below build per backend: wgpu with the cubecl kernels and no burn autotune (the browser's kernel set,
# as katago-ws's images run), rocm with ROCm, flex with the defaults.

# Raw-output parity of the burn import against onnxruntime (backend: flex | wgpu | rocm; net: b10 | b18 | human).
parity backend="flex" net="b10":
    cargo run --release -p katago-nn-burn --example parity {{ if backend == "rocm" { "-F rocm" } else if backend == "wgpu" { "-F wgpu-fixed" } else { "" } }} -- {{backend}} {{net}}

# Native forward-pass throughput, evals/s with one readback per batch (backend: flex | wgpu | rocm; net: b6 | b10 |
# b18 | human; precision: f32 | f16, GPU only). README "Benchmarks".
bench backend="wgpu" net="b18" precision="f32" *batches="1 8 16 32 64":
    cargo run --release -p katago-nn-burn --example bench {{ if backend == "rocm" { "-F rocm" } else if backend == "wgpu" { "-F wgpu-fixed" } else { "" } }} -- {{backend}} {{net}} {{precision}} {{batches}}

# Per-layer timing and check of the Winograd 3x3 and NCHW 1x1 convs against im2col on b18's trunk shapes (WebGPU).
wino-bench precision="f32" *batches="8 16 32 64":
    cargo run --release -p katago-nn-burn --example wino_bench -F wgpu-fixed -- {{precision}} {{batches}}

# Upstream KataGo's own `benchmark` on the same net, for comparison (README "Benchmarks"): KATAGO is the binary
# (default katago-ws's bench copy, an OpenCL v1.16.5 release), NETS the directory of `.bin.gz` nets (default
# katago-ws's bench/models, `scripts/bench_setup.sh` there), CONFIG its GTP config (default: the one beside KATAGO,
# else upstream's gtp_example.cfg), threads the search threads to try.
bench-upstream net="kata1-b18c384nbt" threads="8,16,32,64":
    #!/usr/bin/env bash
    set -euo pipefail
    katago="${KATAGO:-{{justfile_directory()}}/../katago-ws/bench/katago-dist/katago}"
    nets="${NETS:-{{justfile_directory()}}/../katago-ws/bench/models}"
    config="${CONFIG:-$(dirname "$katago")/default_gtp.cfg}"
    [ -f "$config" ] || config={{justfile_directory()}}/upstream/KataGo/cpp/configs/gtp_example.cfg
    "$katago" benchmark -model "$nets/{{net}}.bin.gz" -config "$config" -t {{threads}} -v 800 -n 10

# Browser throughput (Chromium, headless): b10c128 100-visit searches on WebGPU and the wasm CPU, then the b18c384nbt
# set (f32 and f16, 100 and 400 visits). `just web-build` first. README "Benchmarks".
bench-web:
    node {{web_pkg}}/test/run.mjs --only throughput --backends webgpu,cpu
    node {{web_pkg}}/test/run.mjs --only b18

# Format the workspace: katago-core and the files ported from upstream are skipped (they keep upstream's line
# breaks, CLAUDE.md); each crate's rustfmt.toml has its style. `just fmt --check` only checks.
fmt *flags="":
    cargo fmt --all {{flags}}

# Every upstream cpp/ file must be ported by one module or ignored with a reason.
manifest-check:
    python3 -I tools/manifest_check.py

# serde_json features (PORTING.md §9): never `preserve_order` anywhere in the workspace (Map must
# sort like std::map); katago-core builds always have `float_roundtrip` (correctly rounded floats).
check-features:
    #!/usr/bin/env bash
    set -euo pipefail
    ws="$(cargo tree --workspace --target all -e features -i serde_json)"
    if grep -q 'serde_json feature "preserve_order"' <<<"$ws"; then
        echo 'check-features: serde_json has `preserve_order` (forbidden, PORTING.md §9):'; cargo tree --workspace --target all -e features -i serde_json | grep -B2 -A6 preserve_order; exit 1
    fi
    core="$(cargo tree -p katago-core --target all -e features -i serde_json)"
    if ! grep -q 'serde_json feature "float_roundtrip"' <<<"$core"; then
        echo 'check-features: serde_json lacks `float_roundtrip` in katago-core builds (PORTING.md §9)'; exit 1
    fi
    echo 'check-features: serde_json ok (float_roundtrip, no preserve_order)'

# Oracle tests on wasm32-unknown-unknown under node: port::math (vs glibc), Rand, hash, printf,
# rules, global, testrules, the BoardHistory fuzz oracle. On wasm the std float math is not
# glibc's, so this is where libm drift shows up. Plus katago-nn-burn's wasm_yield: the CPU backend
# yields to the JS event loop between batches (PORTING.md §11). Plus katago-annotate's katago-ws parity replay (M6).
# Needs node, wasm-bindgen-test-runner 0.2.129 (= the wasm-bindgen in Cargo.lock) and cargo-nextest (one node process
# per test, in parallel: ~100 s instead of ~230 s with cargo test).
test-wasm:
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo nextest run -p katago-core --target wasm32-unknown-unknown --test rand --test math --test hash --test port_fmt --test rules --test global --test testrules --test boardhistory_fuzz_oracle --test math_tanhf --test nninputs_oracle --test nneval_skipnn --test testsymmetries --test testnninputs --test testscore --test modelversion_desc --test nnoutput_oracle --test math_lgamma --test math_atan --test math_tanh --test testsearchnonn --test math_log10 --test fancymath --test timecontrols --test distributiontable --test config_parser --test testconfig --test search_oracle --test searchtimehelpers_oracle --test search_select_diff --test search_evalcache_diff --test setup_oracle --test setup_host --test asyncbot --test executor --test nlohmann_bytes --test math_sincos --test sgfmetadata_oracle --test datetime --test search_humansl --test localpattern_oracle --test reportedsearchvalues_oracle --test searchparams_oracle --test subtreevaluebiastable_oracle --test math_atanf
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo nextest run -p katago-nn-burn --target wasm32-unknown-unknown --no-default-features -F flex,net-b6c96 --test wasm_yield
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo nextest run -p katago-annotate --target wasm32-unknown-unknown --test fixtures

# The human-SL net (b18c384nbt-humanv0) inside wasm on burn flex vs the Eigen NNEvaluator records (M8; release, a few
# minutes of build, ~1-2 s per evaluation under node). Opt-in: not part of test-wasm.
test-wasm-human:
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo nextest run --release -p katago-nn-burn --target wasm32-unknown-unknown --no-default-features -F flex,net-b18c384nbt-humanv0 --test wasm_human --success-output final

# The native tests of every crate (debug, default features: burn flex and every compiled-in net), without the
# real-net search suites (`quick` profile, .config/nextest.toml). Filter by area with a nextest filterset:
# `just test -E 'package(katago-core)'`, `just test -E 'binary(testrules)'`, `just test -E 'test(/ko_hash/)'`.
test *flags="":
    cargo nextest run --workspace --profile quick {{flags}}

# The GPU tests (nn_parity on wgpu, f32 and f16; add `-F rocm` for ROCm), one at a time (test group `gpu`). Release.
test-gpu *flags="":
    cargo nextest run --release -p katago-nn-burn --test nn_parity -F wgpu --success-output final -E 'test(/wgpu|rocm/)' {{flags}}

# NN tier (M2): ported NNEvaluator + burn backend vs upstream NNEvaluator + Eigen (fixtures/upstream-v1.16.5/nn).
# Default features: flex with b6c96 (incl. the evals_rand.txt symmetry/lifecycle replay), b10c128 and
# b18c384nbt (~27 s release). wgpu too: `just nn-parity -F wgpu`.
nn-parity *flags="":
    cargo nextest run --release -p katago-nn-burn --test nn_parity --success-output final {{flags}}

# Search smoke (real net): ported Search + NNEvaluator over burn flex (g170-b6c96) vs upstream Search over the Eigen
# NNEvaluator (fixtures/upstream-v1.16.5/search/oracle/search_eigen_b6c96.txt, ~40 searches). Tolerance comparison:
# prints visit-vector / top-move / value-diff statistics by maxVisits. Release only (debug burn is very slow).
search-smoke *flags="":
    cargo nextest run --release -p katago-nn-burn --test search_smoke --success-output final {{flags}}

# Analysis goldens (M3, D9): cpp/tests/analysis/*.txt through the ported analysis engine (katago-cli, burn flex,
# g170-b6c96) vs our Eigen reruns fixtures/upstream-v1.16.5/analysis/*.stdout (structure exact, floats within
# the comparator's tolerances), plus the protocol tests and the katago-rs binary end to end. Release.
analysis-goldens *flags="":
    cargo nextest run --release -p katago-cli --success-output final {{flags}}

# Analysis-protocol corpora (fixtures/upstream-v1.16.5/analysis-protocol): regenerate the expected outputs with
# the reference build (`just katago-eigen` first; ~10 min).
analysis-protocol-regen *corpora="":
    cd fixtures/upstream-v1.16.5/analysis-protocol && python3 -I regen.py {{corpora}}

# Replay the analysis-protocol corpora through the katago-rs binary vs the reference outputs (release, ~1.5 min).
analysis-protocol *flags="":
    cargo nextest run --release -p katago-cli --test analysis_protocol_corpus --success-output final {{flags}}

# Real-net search tests (testsearchv9 / testsearchv8 transcriptions, b6c96 via burn flex vs the Eigen expected
# outputs). Release: the two heaviest v8 groups are ignored in debug (~15 min release in parallel).
search-tests *flags="":
    cargo nextest run --release -p katago-nn-burn --test testsearchv9 --test testsearchv8 --success-output final {{flags}}

# Tolerance study: 160 b6c96 + 32 b10c128 searches at 400-1000 visits, burn flex vs Eigen
# (fixtures/upstream-v1.16.5/search/oracle/search_tolerance_*.txt). Release, ~16 min.
search-tolerance *flags="":
    cargo nextest run --release -p katago-nn-burn --test search_tolerance --success-output final {{flags}}

# katago-annotate (M6) vs katago-ws: replay the recorded games (fixtures/katago-annotate) natively and on wasm32;
# payloads and query lines must be byte-identical to katago-ws's.
annotate-parity:
    cargo nextest run -p katago-annotate
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo nextest run -p katago-annotate --target wasm32-unknown-unknown --test fixtures

# Re-record fixtures/katago-annotate through katago-ws's own client and annotator (needs ../katago-ws and
# `just katago-eigen`; ~6 min), or `--check-only` to compare katago-ws's annotator with the checked-in fixtures.
annotate-parity-regen *args="":
    tools/annotate-parity/regen.sh {{args}}

# npm package packages/katago-web (M4): the katago-wasm engine (cargo profile `web`: size-optimized except burn's CPU
# kernels; wasm-bindgen --target web, wasm-opt when installed),
# the TypeScript client and worker (tsc), and dist/models/ (the compiled-in nets' .bpk weights + models.json with
# their sha256). Needs wasm-bindgen 0.2.129 and a TypeScript compiler (TSC, default: sai-frontend's).
tsc := env_var_or_default("TSC", justfile_directory() / "../baduk_review/sai-frontend/node_modules/.bin/tsc")
web_pkg := "packages/katago-web"

web-build *features="":
    #!/usr/bin/env bash
    set -euo pipefail
    # The .bpk files come from the out dir of the katago-nn-burn build this very build used (a `ls -t` over
    # target/ would also find the nets of an earlier build with other features).
    mkdir -p target
    cargo build --profile web -p katago-wasm --target wasm32-unknown-unknown --message-format=json-render-diagnostics {{features}} > target/web-build-messages.json
    nn_out="$(jq -r 'select(.reason=="build-script-executed" and (.package_id|test("katago-nn-burn"))) | .out_dir' target/web-build-messages.json | tail -1)"
    wasm=target/wasm32-unknown-unknown/web/katago_wasm.wasm
    rm -rf {{web_pkg}}/src/wasm {{web_pkg}}/dist
    wasm-bindgen --target web --out-dir {{web_pkg}}/src/wasm "$wasm"
    if command -v wasm-opt >/dev/null; then
        wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
            {{web_pkg}}/src/wasm/katago_wasm_bg.wasm -o {{web_pkg}}/src/wasm/katago_wasm_bg.wasm
    else
        echo "web-build: wasm-opt not found, wasm left unoptimised"
    fi
    # The worker keys its compiled-kernel cache by build: any change to the wasm invalidates it.
    build_id="$(sha256sum {{web_pkg}}/src/wasm/katago_wasm_bg.wasm | cut -c1-16)"
    echo "export const BUILD_ID = \"$build_id\";" > {{web_pkg}}/src/wasm/build-id.js
    echo "export declare const BUILD_ID: string;" > {{web_pkg}}/src/wasm/build-id.d.ts
    {{tsc}} -p {{web_pkg}}
    cp -r {{web_pkg}}/src/wasm {{web_pkg}}/dist/wasm
    mkdir -p {{web_pkg}}/dist/models
    entries=()
    for net in g170-b6c96 g170e-b10c128 kata1-b18c384nbt b18c384nbt-humanv0; do
        bpk="$nn_out/model/$net.bpk"
        [ -f "$bpk" ] || continue
        sha="$(sha256sum "$bpk" | cut -d' ' -f1)"
        file="$net.${sha:0:12}.bpk"
        cp "$bpk" "{{web_pkg}}/dist/models/$file"
        entries+=("{\"network\":\"$net\",\"file\":\"$file\",\"sha256\":\"$sha\",\"size\":$(stat -c %s "$bpk")}")
    done
    (IFS=,; echo "{\"models\":[${entries[*]}]}") > {{web_pkg}}/dist/models/models.json
    raw=$(stat -c %s {{web_pkg}}/dist/wasm/katago_wasm_bg.wasm)
    gz=$(gzip -9 -c {{web_pkg}}/dist/wasm/katago_wasm_bg.wasm | wc -c)
    br=$(command -v brotli >/dev/null && brotli -c -q 11 {{web_pkg}}/dist/wasm/katago_wasm_bg.wasm | wc -c || echo n/a)
    echo "web-build: katago_wasm_bg.wasm $raw bytes raw, $gz gzip -9, $br brotli -q 11"
    cat {{web_pkg}}/dist/models/models.json

# Browser tests of katago-web (Playwright + system Chromium, headless; WebGPU through Vulkan): the upstream
# analysis goldens on WebGPU, on the CPU and through the no-GPU fallback, the second load without network,
# terminate latency, partial reports, throughput. `just web-build` first. Options: see packages/katago-web/test/run.mjs.
web-test *args="":
    node {{web_pkg}}/test/run.mjs {{args}}

# The b18c384nbt cases of the browser tests (M7): goldens vs the Eigen reference on WebGPU, the 105 MB weights
# cache, adapter limits, throughput, terminate, and the CPU guard. Needs a build with that net:
# `just web-build` (b18 is a default feature of katago-wasm). Several minutes (cold kernels, a 105 MB download per fresh context).
web-test-b18 *args="":
    node {{web_pkg}}/test/run.mjs --only b18 {{args}}

# The human SL network as `humanModel` in the browser: reviewGame with a humanProfile against the native run with
# `-human-model`, and the network left out on the CPU. Needs a build with it: `just web-build -F net-b18c384nbt-humanv0`.
web-test-human *args="":
    node {{web_pkg}}/test/run.mjs --only human {{args}}

# Unit tests of the katago-web client against a fake worker (no wasm, no browser). `just web-build` first.
web-test-client:
    node --test {{web_pkg}}/test/client.test.mjs

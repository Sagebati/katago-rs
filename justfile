# katago-rs task runner.

venv := justfile_directory() / ".venv/bin/python"
bench_models := "/home/sam/src/web/katago-ws/bench/models"

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

# Raw-output parity of the burn import against onnxruntime (backend: flex | wgpu; net: b10 | b18).
parity backend="flex" net="b10":
    cargo run --release -p katago-nn-burn --example parity -F wgpu,net-b18c384nbt -- {{backend}} {{net}}

# Native forward-pass throughput (backend: flex | wgpu; net: b10 | b18).
bench backend="wgpu" net="b10" *batches="1 8 16 32 64":
    cargo run --release -p katago-nn-burn --example bench -F wgpu,net-b18c384nbt -- {{backend}} {{net}} {{batches}}

# Build the browser spike into crates/katago-web-bench/www (serve that directory).
web-bench:
    cargo build --release -p katago-web-bench --target wasm32-unknown-unknown
    wasm-bindgen --target web --out-dir crates/katago-web-bench/www/pkg --no-typescript target/wasm32-unknown-unknown/release/katago_web_bench.wasm
    mkdir -p crates/katago-web-bench/www/assets
    cp "$(ls -t target/wasm32-unknown-unknown/release/build/katago-nn-burn-*/out/model/g170e-b10c128.bpk | head -1)" crates/katago-web-bench/www/assets/
    cp fixtures/m0/b10c128.*.f32 crates/katago-web-bench/www/assets/

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
# yields to the JS event loop between batches (PORTING.md §11).
# Needs node and wasm-bindgen-test-runner 0.2.129 (= the wasm-bindgen in Cargo.lock).
test-wasm:
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p katago-core --target wasm32-unknown-unknown --test rand --test math --test hash --test port_fmt --test rules --test global --test testrules --test boardhistory_fuzz_oracle --test math_tanhf --test nninputs_oracle --test nneval_skipnn --test testsymmetries --test testnninputs --test testscore --test modelversion_desc --test nnoutput_oracle --test math_lgamma --test math_atan --test math_tanh --test testsearchnonn --test math_log10 --test fancymath --test timecontrols --test distributiontable --test config_parser --test testconfig --test search_oracle --test searchtimehelpers_oracle --test search_select_diff --test setup_oracle --test setup_host --test asyncbot --test executor --test nlohmann_bytes
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p katago-nn-burn --target wasm32-unknown-unknown --no-default-features -F flex,net-b6c96 --test wasm_yield

# NN tier (M2): ported NNEvaluator + burn backend vs upstream NNEvaluator + Eigen (fixtures/upstream-v1.16.5/nn).
# Default features: flex with b6c96 (incl. the evals_rand.txt symmetry/lifecycle replay), b10c128 and
# b18c384nbt (~27 s release). wgpu too: `just nn-parity -F wgpu`.
nn-parity *flags="":
    cargo test --release -p katago-nn-burn --test nn_parity {{flags}} -- --nocapture

# Search smoke (real net): ported Search + NNEvaluator over burn flex (g170-b6c96) vs upstream Search over the Eigen
# NNEvaluator (fixtures/upstream-v1.16.5/search/oracle/search_eigen_b6c96.txt, ~40 searches). Tolerance comparison:
# prints visit-vector / top-move / value-diff statistics by maxVisits. Release only (debug burn is very slow).
search-smoke *flags="":
    cargo test --release -p katago-nn-burn --test search_smoke {{flags}} -- --nocapture

# Analysis goldens (M3, D9): cpp/tests/analysis/*.txt through the ported analysis engine (katago-cli, burn flex,
# g170-b6c96) vs our Eigen reruns fixtures/upstream-v1.16.5/analysis/*.stdout (structure exact, floats within
# the comparator's tolerances), plus the protocol tests and the katago-rs binary end to end. Release.
analysis-goldens *flags="":
    cargo test --release -p katago-cli {{flags}} -- --nocapture

# Analysis-protocol corpora (fixtures/upstream-v1.16.5/analysis-protocol): regenerate the expected outputs with
# the reference build (`just katago-eigen` first; ~10 min), or replay them through katago-rs (release, ~1.5 min).
analysis-protocol-regen *corpora="":
    cd fixtures/upstream-v1.16.5/analysis-protocol && python3 -I regen.py {{corpora}}

analysis-protocol *flags="":
    cargo test --release -p katago-cli --test analysis_protocol_corpus {{flags}} -- --nocapture

# Real-net search tests (testsearchv9 / testsearchv8 transcriptions, b6c96 via burn flex vs the Eigen expected
# outputs). Release: the two heaviest v8 groups are ignored in debug (~15 min release in parallel).
search-tests *flags="":
    cargo test --release -p katago-nn-burn --test testsearchv9 --test testsearchv8 {{flags}} -- --nocapture

# Tolerance study: 160 b6c96 + 32 b10c128 searches at 400-1000 visits, burn flex vs Eigen
# (fixtures/upstream-v1.16.5/search/oracle/search_tolerance_*.txt). Release, ~16 min.
search-tolerance *flags="":
    cargo test --release -p katago-nn-burn --test search_tolerance {{flags}} -- --nocapture

# npm package packages/katago-web (M4): the katago-wasm engine (cargo profile `web`: size-optimized except burn's CPU
# kernels; wasm-bindgen --target web, wasm-opt when installed),
# the TypeScript client and worker (tsc), and dist/models/ (the compiled-in nets' .bpk weights + models.json with
# their sha256). Needs wasm-bindgen 0.2.129 and a TypeScript compiler (TSC, default: sai-frontend's).
tsc := env_var_or_default("TSC", "/home/sam/src/web/baduk_review/sai-frontend/node_modules/.bin/tsc")
web_pkg := "packages/katago-web"

web-build *features="":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build --profile web -p katago-wasm --target wasm32-unknown-unknown {{features}}
    wasm=target/wasm32-unknown-unknown/web/katago_wasm.wasm
    rm -rf {{web_pkg}}/src/wasm {{web_pkg}}/dist
    wasm-bindgen --target web --out-dir {{web_pkg}}/src/wasm "$wasm"
    if command -v wasm-opt >/dev/null; then
        wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
            {{web_pkg}}/src/wasm/katago_wasm_bg.wasm -o {{web_pkg}}/src/wasm/katago_wasm_bg.wasm
    else
        echo "web-build: wasm-opt not found, wasm left unoptimised"
    fi
    {{tsc}} -p {{web_pkg}}
    cp -r {{web_pkg}}/src/wasm {{web_pkg}}/dist/wasm
    mkdir -p {{web_pkg}}/dist/models
    entries=()
    for net in g170-b6c96 g170e-b10c128 kata1-b18c384nbt; do
        bpk="$(ls -t target/wasm32-unknown-unknown/web/build/katago-nn-burn-*/out/model/$net.bpk 2>/dev/null | head -1 || true)"
        [ -n "$bpk" ] || continue
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

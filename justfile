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
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p katago-core --target wasm32-unknown-unknown --test rand --test math --test hash --test port_fmt --test rules --test global --test testrules --test boardhistory_fuzz_oracle --test math_tanhf --test nninputs_oracle --test nneval_skipnn --test testsymmetries --test testnninputs --test testscore --test modelversion_desc --test nnoutput_oracle
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner cargo test -p katago-nn-burn --target wasm32-unknown-unknown --no-default-features -F flex,net-b6c96 --test wasm_yield

# NN tier (M2): ported NNEvaluator + burn backend vs upstream NNEvaluator + Eigen (fixtures/upstream-v1.16.5/nn).
# Default features: flex with b6c96 (incl. the evals_rand.txt symmetry/lifecycle replay), b10c128 and
# b18c384nbt (~27 s release). wgpu too: `just nn-parity -F wgpu`.
nn-parity *flags="":
    cargo test --release -p katago-nn-burn --test nn_parity {{flags}} -- --nocapture

//! Test-only link settings (no effect on the library or on dependents).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // wasm32 test binaries get an 8 MiB stack instead of rust-lld's 1 MiB: a debug-build test that is the first to
    // run the global table setup (`init()`, e.g. testrules::brute_force_testing_of_ko_hash_table run alone, as
    // cargo-nextest runs every test) overflowed the default stack ("memory access out of bounds" under node).
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        println!("cargo:rustc-link-arg-tests=-zstack-size=8388608");
    }
}

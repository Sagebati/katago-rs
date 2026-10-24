//! wasm32: a bigger shadow stack than rustc's 1 MB default. The analysis engine's futures are boxed, but
//! `Search` (~40 KB) and `BoardHistory` (~34 KB) values and the recursive playout futures are built on
//! the stack before boxing (katago-cli runs on a 256 MB thread for debug builds; release needs far less).

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
    println!("cargo:rustc-link-arg-cdylib=-zstack-size=16777216");
  }
}

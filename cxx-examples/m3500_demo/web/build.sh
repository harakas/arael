#!/bin/sh
# Build the wasm module the page loads. Afterwards this directory is the
# whole site: `pkg`, `arael` and `datasets` link to the built module, the
# g2o loader and the vendored datasets. Serve it (python3 -m http.server
# -d web), or copy it with the links dereferenced (cp -rL web/ <dest>).
set -e
cd "$(dirname "$0")/../model/wasm"
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --weak-refs --out-dir pkg \
    target/wasm32-unknown-unknown/release/m3500_demo_wasm.wasm

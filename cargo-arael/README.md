# cargo-arael

A cargo subcommand that generates C, C++, Python and JavaScript
(WebAssembly) interfaces for an [arael](https://crates.io/crates/arael)
root model. The generated classes mirror the Rust model: the same
collections, parameters, solvers, configs, reports and covariance
queries, with exact solve parity against Rust.

```bash
cargo install cargo-arael
cd mymodel/ && cargo arael export
```

`export` writes, next to the model crate:

| Path | Content |
|---|---|
| `capi/` | a Rust crate with the C ABI over the model (`cdylib` + `staticlib`) |
| `cxx/` | C++ wrapper classes with vendored math headers and CMake glue |
| `python/` | a pure-`ctypes` Python package; one cdylib serves every CPython 3.x |
| `wasm/` | a wasm-bindgen crate for the browser |

To write only some of them, list them in the model crate's
`Cargo.toml`; `export` and `check` both follow it:

```toml
[package.metadata.arael]
interfaces = ["cxx"]
```

The names are the directories: `capi`, `cxx`, `python`, `wasm`. `cxx`
and `python` call the C ABI, so either brings `capi`. Without the key,
all four are written; `--interfaces cxx,python` overrides it for one
run. A tree left out loses its generated files: `export` removes every
file it generated and no longer writes, never one of your own or a
written-once `Cargo.toml`, and a directory only once it is empty.

The interfaces are additive across releases: a name once generated
keeps its type.

Reference: [docs/CXX.md](https://github.com/harakas/arael/blob/master/docs/CXX.md),
[docs/PYTHON.md](https://github.com/harakas/arael/blob/master/docs/PYTHON.md),
[docs/WASM.md](https://github.com/harakas/arael/blob/master/docs/WASM.md);
the model description it reads is
[docs/SIDECAR.md](https://github.com/harakas/arael/blob/master/docs/SIDECAR.md).

## License

MIT, as arael.

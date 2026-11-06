//! Library surface of the `cargo-arael` tool: the sidecar IR, the
//! per-target emitters and the kernels dump. The binary
//! (`cargo arael <cmd>`) drives these; the golden tests consume them
//! directly.

pub mod emit_ffi;
pub mod emit_hpp;
pub mod emit_py;
pub mod emit_wasm;
pub mod export;
pub mod ir;
pub mod kernels;
pub mod leaves;

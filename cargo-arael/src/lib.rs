//! Library surface of the `cargo-arael` tool: the sidecar IR and the
//! per-target emitters. The binary (`cargo arael <cmd>`) drives these;
//! the golden tests consume them directly.

pub mod emit_ffi;
pub mod emit_hpp;
pub mod emit_py;
pub mod export;
pub mod ir;

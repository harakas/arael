//! The trace named by `ARAEL_GPU_TRACE`, which the process reads once: a
//! file of its own, with no other test to share the environment.

/// A trace file that cannot be created is an error, not a solve left
/// untraced.
#[test]
#[should_panic(expected = "the trace file cannot be created")]
fn a_trace_file_that_cannot_be_created_panics() {
    // SAFETY: this file's only test sets the variable before any other
    // thread of the process reads the environment.
    unsafe { std::env::set_var("ARAEL_GPU_TRACE", "/nonexistent-arael-dir/trace.jsonl") };
    arael_gpu::trace::active();
}

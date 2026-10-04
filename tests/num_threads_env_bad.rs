// A value of ARAEL_NUM_THREADS that is not a count is an error, not a
// silent 1. Own process, like tests/num_threads_env.rs: the variable is
// read once, at the first config.

use arael::simple_lm::LmConfig;

#[test]
#[should_panic(expected = "ARAEL_NUM_THREADS: expected a thread count")]
fn a_value_that_is_not_a_count_is_rejected() {
    // SAFETY: this test is alone in its binary, so no other thread reads
    // the environment while it is written.
    unsafe { std::env::set_var("ARAEL_NUM_THREADS", "four") };
    let _ = LmConfig::<f64>::default();
}

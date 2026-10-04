// ARAEL_NUM_THREADS is the default of LmConfig::num_threads, read once per
// process. This file is a process of its own, and sets the variable before
// the first config is built.

use arael::simple_lm::LmConfig;

#[test]
fn the_environment_sets_the_default_and_code_wins_over_it() {
    // SAFETY: this test is alone in its binary, so no other thread reads
    // the environment while it is written.
    unsafe { std::env::set_var("ARAEL_NUM_THREADS", "3") };
    assert_eq!(arael::threads::default_num_threads(), 3);
    assert_eq!(LmConfig::<f64>::default().num_threads, 3);
    assert_eq!(LmConfig::<f64>::conservative().num_threads, 3);
    assert_eq!(LmConfig::<f64>::well_conditioned().num_threads, 3);
    assert_eq!(LmConfig::<f32>::ill_conditioned().num_threads, 3);
    assert_eq!(LmConfig::<f64>::default().with_num_threads(1).num_threads, 1);
    assert_eq!(LmConfig::<f64>::default().assembly_threads, None);
}

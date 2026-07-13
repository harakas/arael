// Single-core pinning, and the thread-pool muzzle that has to come with it.
//
// The benchmark measures one solver on one core. Every system here can be told
// to use threads (rayon, OpenMP, Eigen's, CHOLMOD's), and a system that quietly
// spreads over eight cores is not comparable to one that does not -- so they are
// all pinned, and the pin is ASSERTED in every subprocess, not assumed.

/// Pin this process (and everything it spawns) to one core, and hold every
/// thread pool to one thread. Exports BENCH_CORE for the subprocess assert.
pub fn enforce_single_core() {
    for var in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TBB_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ] {
        std::env::set_var(var, "1");
    }
    // Pin to the LAST core: core 0 preferentially receives timer ticks and IRQs,
    // so a benchmark pinned there shares its core with kernel housekeeping.
    let last = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(last, &mut set);
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set),
            0, "cannot pin to core {}", last);
    }
    std::env::set_var("BENCH_CORE", last.to_string());
}

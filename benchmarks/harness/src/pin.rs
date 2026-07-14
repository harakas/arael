// Core pinning, and the thread-pool muzzle that has to come with it.
//
// The benchmark measures every solver on the same hardware. Each system here can
// be told to use threads (rayon, OpenMP, Eigen's, CHOLMOD's), and a system that
// quietly spreads over eight cores is not comparable to one that does not -- so
// they all get the same cores and the same thread cap, and the pin is ASSERTED in
// every subprocess, not assumed (see external.rs).
//
// BENCH_THREADS is the one knob: it sizes the pinned core count, every thread
// pool's cap, AND arael's LmConfig::num_threads. Default 1, which is the
// single-core comparison every committed number was measured under. 0 means every
// core.

/// Threads -- and cores -- every system gets. 1 by default; 0 means every core.
///
/// After [`enforce_cores`] this is the RESOLVED count: it rewrites BENCH_THREADS
/// with the number actually pinned, so 0 reads back as the real core count.
pub fn threads() -> usize {
    std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// Pin this process (and everything it spawns) to `BENCH_THREADS` cores, and hold
/// every thread pool to the same number. Exports BENCH_CORE for the subprocess
/// assert.
pub fn enforce_cores() {
    let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n = match threads() {
        0 => total,
        n => n.min(total),
    };
    // Resolved, so threads() reads back a real count and every reader agrees --
    // arael's num_threads included.
    std::env::set_var("BENCH_THREADS", n.to_string());

    // Same budget for everyone. A system given eight threads is not comparable to
    // one given one, whichever way the result falls.
    for var in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TBB_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ] {
        std::env::set_var(var, n.to_string());
    }

    // The LAST n cores: core 0 preferentially receives timer ticks and IRQs, so a
    // benchmark that lands there shares its core with kernel housekeeping.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for c in (total - n)..total {
            libc::CPU_SET(c, &mut set);
        }
        assert_eq!(
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set),
            0,
            "cannot pin to the last {n} of {total} cores"
        );
    }

    // Export what the KERNEL calls this mask, not what we think it should be
    // called: the subprocess assert compares its own Cpus_allowed_list against
    // this string, so both have to come from the same formatter.
    std::env::set_var("BENCH_CORE", cpus_allowed());
}

/// This process's `Cpus_allowed_list`, in the kernel's own formatting -- "7" for
/// one core, "4-7" for a range. The C++ runners read the same field.
pub fn cpus_allowed() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Cpus_allowed_list:"))
                .map(|l| l.split_whitespace().last().unwrap_or("?").to_string())
        })
        .unwrap_or_else(|| "?".to_string())
}

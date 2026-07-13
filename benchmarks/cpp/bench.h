// The C++ side of the benchmark harness: what every external runner does the
// same way, so each one is left with only its scene and its solve().
//
// A runner supplies a solve(max_iters, out) that returns a Result. This header
// runs the probes, applies the rules a probe's number has to satisfy, and prints
// the protocol line the Rust harness parses.
//
// The rules are not decoration. Each one is a bug this benchmark had:
//   - the first solve in a process pays cold allocator and cache costs the rest
//     do not, so an un-warmed probe charges them to whichever probe runs first;
//   - a complete iteration is t(2 iters) - t(1 iter), and differencing two noisy
//     single samples produced a NEGATIVE iteration time;
//   - a "first iteration" that rejected a step is mostly wasted factorizations
//     (g2o at the wrong damping burned six trials in its first iteration), so
//     its time, and everything derived from it, is not reported at all.
#ifndef ARAEL_BENCH_H
#define ARAEL_BENCH_H

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace bench {

// Kept in step with harness/src/probe.rs.
static const int PROBE_SUBROUNDS = 2;

struct Result {
    double ms = 0;
    // Accepted, cost-decreasing steps.
    int accepted = 0;
    // Attempts: accepted steps plus damping retries, each of which costs a
    // factorization. Equal to `accepted` for a solver with no retry loop.
    int attempts = 0;
    // Optional cross-check: the cost of the INITIAL estimate as this system's
    // own code computes it. The harness asserts it equals the reference cost,
    // which is what proves every system is minimizing the same objective.
    double initial_cost = 0;
};

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline std::string cpus_allowed() {
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("Cpus_allowed_list:", 0) == 0) {
            return line.substr(line.find_last_of(" \t") + 1);
        }
    }
    return "?";
}

inline double peak_rss_mb() {
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return atof(line.c_str() + 6) / 1024.0;
        }
    }
    return 0;
}

// Fastest of PROBE_SUBROUNDS runs of the same capped solve, after a discarded
// warmup. Solve is any callable: Result(int max_iters).
template <typename Solve>
Result probe(Solve solve, int max_iters) {
    solve(max_iters);  // warmup, discarded
    Result best = solve(max_iters);
    for (int i = 1; i < PROBE_SUBROUNDS; i++) {
        Result r = solve(max_iters);
        if (r.ms < best.ms) best = r;
    }
    return best;
}

// Run the probes and the full solve, and print the protocol line.
//
// Solve is Result(int max_iters) for the probes; Full is Result() for the real
// run, which is the one that writes the solution out.
template <typename Solve, typename Full>
void report(Solve solve, Full full_solve) {
    Result first = probe(solve, 1);
    Result two = probe(solve, 2);
    Result full = full_solve();

    std::string extra;
    if (two.accepted >= 2) {
        char buf[128];
        snprintf(buf, sizeof buf, ", \"second_run_ms\": %.3f, \"second_accepted\": %d",
                 two.ms, two.accepted);
        extra += buf;
    }
    if (full.initial_cost != 0) {
        char buf[64];
        snprintf(buf, sizeof buf, ", \"initial_cost\": %.6f", full.initial_cost);
        extra += buf;
    }
    printf("{\"solve_ms\": %.3f, \"first_iter_ms\": %.3f, "
           "\"iterations\": %d, \"accepted\": %d, "
           "\"first_attempts\": %d, \"first_accepted\": %d%s, "
           "\"peak_mb\": %.1f, \"cpus_allowed\": \"%s\"}\n",
           full.ms, first.ms, full.attempts, full.accepted,
           first.attempts, first.accepted, extra.c_str(),
           peak_rss_mb(), cpus_allowed().c_str());
}

}  // namespace bench

#endif  // ARAEL_BENCH_H

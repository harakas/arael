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
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

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

// Repeat `f` until `budget_s` seconds elapse (at least once, at most `cap`
// times); return the median wall-clock milliseconds and write the count to
// `*reps`. Mirrors harness/src/probe.rs median_ms -- covariance costs span
// microseconds (one marginal) to seconds (a dense cross-library solve), so a
// time budget adapts where a fixed rep count cannot, and the count shows how
// many samples back the median.
template <typename F>
double median_ms(double budget_s, int cap, int* reps, F f) {
    std::vector<double> s;
    double start = now_ms();
    while ((int)s.size() < cap) {
        double t = now_ms();
        f();
        s.push_back(now_ms() - t);
        if ((now_ms() - start) / 1e3 >= budget_s) break;
    }
    std::sort(s.begin(), s.end());
    *reps = (int)s.size();
    size_t mid = s.size() / 2;
    return s.empty() ? 0.0 : (s.size() % 2 == 0 ? (s[mid - 1] + s[mid]) / 2.0 : s[mid]);
}

// Per-cell wall-time cap in ms (COV_CELL_CAP_S, default 120 s). A covariance
// cell projected or measured past it is left un-run and reported "toolong" (the
// Rust harness renders `*`), so nothing waits longer than this.
inline double cov_cell_cap_ms() {
    return getenv("COV_CELL_CAP_S") ? atof(getenv("COV_CELL_CAP_S")) * 1e3 : 120e3;
}

// Projected wall time (ms) for a cell at count n, from the previous completed
// (prev_n, prev_ms), assuming ~linear per-query cost. 0 with no prior.
inline double cov_project_ms(int prev_n, double prev_ms, int n) {
    return prev_n > 0 ? prev_ms * (double)n / prev_n : 0.0;
}

// A spread of `n` indices over [base, base+count): evenly sampled, or all of them
// when n >= count. Used to pick which cameras/points to query.
inline std::vector<int> spread(int base, int count, int n) {
    std::vector<int> idx;
    if (n >= count) {
        for (int i = 0; i < count; i++) idx.push_back(base + i);
    } else {
        for (int i = 0; i < n; i++) idx.push_back(base + (int)((long)i * count / n));
    }
    return idx;
}

// BENCH_QUICK: the damping sweep's mode. No warmup, one sub-round, and the full
// solve held to two iterations -- enough to see whether the first two iterations
// are CLEAN, which is all full-iter needs, without paying for a converged solve
// on a problem that takes minutes. Never set it for a measurement.
inline bool quick() { return getenv("BENCH_QUICK") != nullptr; }

// How many iterations the reported solve runs. The runner passes what the
// benchmark wants; the sweep cuts it to two.
inline int full_iters(int wanted) { return quick() ? 2 : wanted; }

// Fastest of PROBE_SUBROUNDS runs of the same capped solve, after a discarded
// warmup. Solve is any callable: Result(int max_iters).
template <typename Solve>
Result probe(Solve solve, int max_iters) {
    if (quick()) return solve(max_iters);
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

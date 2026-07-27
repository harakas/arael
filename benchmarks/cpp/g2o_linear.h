// Linear-solver choice shared by the g2o runners: CHOLMOD (direct) or
// g2o's block-Jacobi preconditioned conjugate gradient. PCG solves
// whatever the runner hands it -- the full Hessian on a pose graph, the
// reduced camera system in BAL, where the points are already
// marginalized.
//
// PCG stops when the preconditioned residual falls to `tol` times its
// value at the start of the solve. absoluteTolerance additionally
// floors that at the previous solve's final residual, so successive
// linearizations are solved more loosely -- which is why a PCG row's
// step accuracy is not comparable to a direct row's. Defaults are
// g2o's shipped ones (1e-6, loosening); measured faster here than the
// relative 1e-8 the g2o paper states. G2O_PCG_ABSTOL=0 selects that.
#ifndef ARAEL_G2O_LINEAR_H
#define ARAEL_G2O_LINEAR_H

#include <g2o/core/batch_stats.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace g2olin {

inline double pcg_tolerance() {
    if (const char* t = getenv("G2O_PCG_TOL")) return atof(t);
    return 1e-6;
}

inline bool pcg_absolute_tolerance() {
    const char* v = getenv("G2O_PCG_ABSTOL");
    return !v || std::string(v) != "0";
}

// Hard cap on CG iterations per linear solve. 0 (the default) leaves
// g2o's own, which is the system dimension -- effectively uncapped.
inline int pcg_max_iterations() {
    if (const char* m = getenv("G2O_PCG_MAXITER")) return atoi(m);
    return 0;
}

// Compile-time block dimensions versus the dynamic BlockSolverX, for
// the pose-graph runners (BAL is fixed at 9/3 either way).
//
// CHOLMOD factorizes a flattened scalar matrix, so the block dimension
// barely reaches it -- within 5% either way. PCG multiplies through
// each block every CG iteration, and at dynamic size the compiler
// cannot see the extents: fixed is 8x faster on M3500 and 4.2x on
// sphere2500 at identical CG iteration counts. So each kind gets the
// dimensions it is fastest under. G2O_BLOCK=fixed|dyn forces one.
inline bool fixed_blocks(bool pcg) {
    if (const char* b = getenv("G2O_BLOCK")) return std::string(b) == "fixed";
    return pcg;
}

template <typename BS>
inline std::unique_ptr<BS> make_block_solver(bool pcg) {
    if (pcg) {
        auto linear = std::make_unique<g2o::LinearSolverPCG<typename BS::PoseMatrixType>>();
        linear->setAbsoluteTolerance(pcg_absolute_tolerance());
        linear->setTolerance(pcg_tolerance());
        if (int m = pcg_max_iterations()) linear->setMaxIterations(m);
        return std::make_unique<BS>(std::move(linear));
    }
    auto linear = std::make_unique<g2o::LinearSolverCholmod<typename BS::PoseMatrixType>>();
    return std::make_unique<BS>(std::move(linear));
}

// Total CG iterations over the solve, or -1 when not collected.
//
// g2o reports the count only through its batch statistics, and turning
// those on makes the optimizer time and record every iteration. That
// bookkeeping sits inside the region the benchmark times, so it is
// gated on G2O_PCG_STATS: a measurement run takes the same code path
// it did before PCG existed, and a diagnostic run gives the counts.
inline bool pcg_stats_wanted() { return getenv("G2O_PCG_STATS") != nullptr; }

inline int pcg_total_iterations(const g2o::SparseOptimizer& opt) {
    // Empty means the run never enabled them -- a direct-solver row, which
    // has no CG iterations to report rather than zero of them.
    if (!pcg_stats_wanted() || opt.batchStatistics().empty()) return -1;
    int total = 0;
    for (const auto& s : opt.batchStatistics()) total += s.iterationsLinearSolver;
    return total;
}

}  // namespace g2olin

#endif  // ARAEL_G2O_LINEAR_H

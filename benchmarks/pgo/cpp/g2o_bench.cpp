// g2o runner for the 2D pose-graph benchmark. Same protocol as the
// other runners:
//   g2o_bench <file.g2o> <lm|gn|lm-pcg|gn-pcg> <poses_out> <info|unit>
// prints JSON {solve_ms, first_iter_ms, iterations, cpus_allowed} and
// writes "x y theta" lines to poses_out. Timing wraps initialize +
// optimize. The gauge prior is a proper soft unit-weight EdgeSE2Prior
// (same convention as every other runner), not a fixed vertex.
//
// The -pcg kinds swap the CHOLMOD factorization for g2o's iterative
// block-Jacobi preconditioned conjugate gradient, leaving the rest of
// the run identical, so the pair isolates the linear solver.

#include "../../cpp/bench.h"
#include "../../cpp/g2o_linear.h"
#include <g2o/core/batch_stats.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/types/slam2d/types_slam2d.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


struct PoseIn {
    double x, y, th;
};
struct EdgeIn {
    int a, b;
    double dx, dy, dth;
    double it, ir; // information (not sqrt) for translation / rotation
};

static void parse_g2o(const char* path, bool unit, std::vector<PoseIn>& poses, std::vector<EdgeIn>& edges) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "VERTEX_SE2") {
            int id;
            PoseIn p;
            ss >> id >> p.x >> p.y >> p.th;
            poses.push_back(p);
        } else if (tag == "EDGE_SE2") {
            EdgeIn e;
            double i11, i12, i13, i22, i23, i33;
            ss >> e.a >> e.b >> e.dx >> e.dy >> e.dth >> i11 >> i12 >> i13 >> i22 >> i23 >> i33;
            e.it = unit ? 1.0 : i11;
            e.ir = unit ? 1.0 : i33;
            edges.push_back(e);
        }
    }
}

// g2o keeps its damping retries inside OptimizationAlgorithmLevenberg::solve()
// and reports the count only for the round just finished (levenbergIteration()
// is reset every round), so summing them from a post-iteration action is the
// one way to see the attempts from outside. Gauss-Newton has no retry loop, so
// there attempts always equal iterations.
struct TrialCounter : public g2o::HyperGraphAction {
    g2o::OptimizationAlgorithmLevenberg* lev = nullptr;
    int trials = 0;
    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*,
                                      Parameters* = 0) override {
        if (lev) trials += lev->levenbergIteration();
        return this;
    }
};

template <typename BS>
static bench::Result solve(const std::vector<PoseIn>& poses, const std::vector<EdgeIn>& edges,
                       bool lm, bool pcg, int max_iters, std::vector<PoseIn>* out) {
    auto block = g2olin::make_block_solver<BS>(pcg);
    g2o::OptimizationAlgorithm* algo;
    auto* counter = new TrialCounter();
    if (lm) {
        auto* lev = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
        // Problem-appropriate initial lambda (g2o's auto heuristic,
        // 1e-5 * max Hessian diagonal, over-damps heavily weighted
        // graphs and its descent then trips the gain-threshold stop;
        // see the README's initial-damping policy). Env-overridable.
        double lambda0 = 1e-12;
        if (const char* li = getenv("G2O_LAMBDA_INIT")) lambda0 = atof(li);
        lev->setUserLambdaInit(lambda0);
        algo = lev;
        counter->lev = lev;
    } else {
        algo = new g2o::OptimizationAlgorithmGaussNewton(std::move(block));
    }
    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    opt.setAlgorithm(algo);
    if (pcg && g2olin::pcg_stats_wanted()) opt.setComputeBatchStatistics(true);

    for (size_t i = 0; i < poses.size(); i++) {
        auto* v = new g2o::VertexSE2();
        v->setId((int)i);
        v->setEstimate(g2o::SE2(poses[i].x, poses[i].y, poses[i].th));
        opt.addVertex(v);
    }
    for (const EdgeIn& e : edges) {
        auto* edge = new g2o::EdgeSE2();
        edge->setVertex(0, opt.vertex(e.a));
        edge->setVertex(1, opt.vertex(e.b));
        edge->setMeasurement(g2o::SE2(e.dx, e.dy, e.dth));
        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        info(0, 0) = e.it;
        info(1, 1) = e.it;
        info(2, 2) = e.ir;
        edge->setInformation(info);
        opt.addEdge(edge);
    }
    // Soft unit gauge prior on pose 0. EdgeSE2Prior measures the vertex
    // pose in the frame of an SE2 offset parameter (identity here).
    auto* offset = new g2o::ParameterSE2Offset();
    offset->setId(0);
    opt.addParameter(offset);
    auto* prior = new g2o::EdgeSE2Prior();
    prior->setVertex(0, opt.vertex(0));
    prior->setMeasurement(g2o::SE2(poses[0].x, poses[0].y, poses[0].th));
    prior->setInformation(Eigen::Matrix3d::Identity());
    prior->setParameterId(0, 0);
    opt.addEdge(prior);

    // Same termination class as the other systems: stop when the
    // relative chi2 gain of an iteration falls below 1e-5.
    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);
    opt.addPostIterationAction(counter);

    auto t0 = std::chrono::steady_clock::now();
    opt.initializeOptimization();
    int iters = opt.optimize(max_iters);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (out) {
        int cg = g2olin::pcg_total_iterations(opt);
        if (cg >= 0) fprintf(stderr, "  [pcg] %d CG iterations over %d solver iterations\n", cg, iters);
        out->resize(poses.size());
        for (size_t i = 0; i < poses.size(); i++) {
            auto* v = static_cast<g2o::VertexSE2*>(opt.vertex((int)i));
            (*out)[i] = PoseIn{v->estimate().translation().x(), v->estimate().translation().y(),
                               v->estimate().rotation().angle()};
        }
    }
    return bench::Result{ms, iters, lm ? counter->trials : iters};
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <g2o> <lm|gn|lm-pcg|gn-pcg> <poses_out> <info|unit>\n", argv[0]);
        return 1;
    }
    const std::string kind = argv[2];
    bool lm = kind.rfind("lm", 0) == 0;
    bool pcg = kind.find("-pcg") != std::string::npos;
    bool unit = std::string(argv[4]) == "unit";
    std::vector<PoseIn> poses;
    std::vector<EdgeIn> edges;
    parse_g2o(argv[1], unit, poses, edges);

    // SE2 poses are 3-dimensional; BlockSolver_3_2's landmark slot is
    // unused on a pose graph.
    const bool fixed = g2olin::fixed_blocks(pcg);
    auto run = [&](int n, std::vector<PoseIn>* out) {
        return fixed ? solve<g2o::BlockSolver_3_2>(poses, edges, lm, pcg, n, out)
                     : solve<g2o::BlockSolverX>(poses, edges, lm, pcg, n, out);
    };

    std::vector<PoseIn> result;
    bench::report(
        [&](int n) { return run(n, nullptr); },
        [&]() { return run(bench::full_iters(100), &result); });

    std::ofstream out(argv[3]);
    for (const PoseIn& p : result) out << p.x << " " << p.y << " " << p.th << "\n";

    return 0;
}

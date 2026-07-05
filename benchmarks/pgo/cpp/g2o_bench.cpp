// g2o runner for the 2D pose-graph benchmark. Same protocol as the
// other runners:
//   g2o_bench <file.g2o> <lm|gn> <poses_out> <info|unit>
// prints JSON {solve_ms, first_iter_ms, iterations, cpus_allowed} and
// writes "x y theta" lines to poses_out. Timing wraps initialize +
// optimize. The gauge prior is a proper soft unit-weight EdgeSE2Prior
// (same convention as every other runner), not a fixed vertex.

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
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

struct RunResult {
    double ms;
    int iterations;
};

static RunResult solve(const std::vector<PoseIn>& poses, const std::vector<EdgeIn>& edges,
                       bool lm, int max_iters, std::vector<PoseIn>* out) {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    auto linear = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    g2o::OptimizationAlgorithm* algo;
    if (lm) {
        algo = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
    } else {
        algo = new g2o::OptimizationAlgorithmGaussNewton(std::move(block));
    }
    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    opt.setAlgorithm(algo);

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
    terminate->setGainThreshold(1e-5);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);

    auto t0 = std::chrono::steady_clock::now();
    opt.initializeOptimization();
    int iters = opt.optimize(max_iters);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (out) {
        out->resize(poses.size());
        for (size_t i = 0; i < poses.size(); i++) {
            auto* v = static_cast<g2o::VertexSE2*>(opt.vertex((int)i));
            (*out)[i] = PoseIn{v->estimate().translation().x(), v->estimate().translation().y(),
                               v->estimate().rotation().angle()};
        }
    }
    return RunResult{ms, iters};
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <g2o> <lm|gn> <poses_out> <info|unit>\n", argv[0]); return 1; }
    bool lm = std::string(argv[2]) == "lm";
    bool unit = std::string(argv[4]) == "unit";
    std::vector<PoseIn> poses;
    std::vector<EdgeIn> edges;
    parse_g2o(argv[1], unit, poses, edges);

    RunResult first = solve(poses, edges, lm, 1, nullptr);
    std::vector<PoseIn> result;
    RunResult full = solve(poses, edges, lm, 100, &result);

    std::ofstream out(argv[3]);
    for (const PoseIn& p : result) {
        out << p.x << " " << p.y << " " << p.th << "\n";
    }

    std::string cpus = "?";
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l)) {
        if (l.rfind("Cpus_allowed_list:", 0) == 0) {
            cpus = l.substr(l.find_last_of(" \t") + 1);
        }
    }
    printf("{\"solve_ms\": %.3f, \"first_iter_ms\": %.3f, \"iterations\": %d, \"cpus_allowed\": \"%s\"}\n",
           full.ms, first.ms, full.iterations, cpus.c_str());
    return 0;
}

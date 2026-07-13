// g2o runner for the 3D pose-graph benchmark. g2o's native EdgeSE3
// error is [ t(delta) ; vec(q(delta)) ] with delta = T_meas^-1 T_a^-1
// T_b (toVectorMQT, quaternion sign-normalized) -- the same family as
// the benchmark's canonical residual, differing only by a constant
// per-edge linear transform B = blockdiag(R_meas^T, 0.5 I):
//   r_g2o = B r_canonical.
// Feeding each edge the conjugated information
//   info' = B^-T info B^-1
//         = [ R_m^T Itt R_m   2 R_m^T Itr ]
//           [ 2 Itr^T R_m     4 Irr       ]
// makes g2o's native edge minimize the canonical cost EXACTLY (its
// chi2 equals the reference cost function value at every point, which
// the harness cross-checks via the initial_chi2 field below). Same for
// the unit gauge prior: its error frame yields info' =
// diag(1,1,1,4,4,4). Protocol:
//   g2o_bench3 <file.g2o> <lm|gn> <poses_out>
// prints JSON {solve_ms, first_iter_ms, iterations, initial_chi2,
// cpus_allowed}, writes "x y z qx qy qz qw" lines.

#include "../../cpp/bench.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


struct PoseIn {
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};
struct EdgeIn {
    int a, b;
    Eigen::Isometry3d meas;
    Eigen::Matrix<double, 6, 6> info; // transformed for g2o's error frame
};

static void parse_g2o(const char* path, std::vector<PoseIn>& poses, std::vector<EdgeIn>& edges) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "VERTEX_SE3:QUAT") {
            int id;
            double x, y, z, qx, qy, qz, qw;
            ss >> id >> x >> y >> z >> qx >> qy >> qz >> qw;
            if (id != (int)poses.size()) { fprintf(stderr, "non-dense vertices\n"); exit(1); }
            poses.push_back(PoseIn{{x, y, z}, Eigen::Quaterniond(qw, qx, qy, qz).normalized()});
        } else if (tag == "EDGE_SE3:QUAT") {
            EdgeIn e;
            double x, y, z, qx, qy, qz, qw;
            ss >> e.a >> e.b >> x >> y >> z >> qx >> qy >> qz >> qw;
            Eigen::Quaterniond dq = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            e.meas = Eigen::Isometry3d::Identity();
            e.meas.linear() = dq.toRotationMatrix();
            e.meas.translation() = Eigen::Vector3d(x, y, z);
            Eigen::Matrix<double, 6, 6> info;
            for (int i = 0; i < 6; i++) {
                for (int j = i; j < 6; j++) {
                    double v;
                    ss >> v;
                    info(i, j) = v;
                    info(j, i) = v;
                }
            }
            const Eigen::Matrix3d rm = e.meas.linear();
            e.info.block<3, 3>(0, 0) = rm.transpose() * info.block<3, 3>(0, 0) * rm;
            e.info.block<3, 3>(0, 3) = 2.0 * rm.transpose() * info.block<3, 3>(0, 3);
            e.info.block<3, 3>(3, 0) = e.info.block<3, 3>(0, 3).transpose();
            e.info.block<3, 3>(3, 3) = 4.0 * info.block<3, 3>(3, 3);
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

static bench::Result solve(const std::vector<PoseIn>& poses, const std::vector<EdgeIn>& edges,
                       bool lm, int max_iters, std::vector<PoseIn>* out) {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    auto linear = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    g2o::OptimizationAlgorithm* algo;
    auto* counter = new TrialCounter();
    if (lm) {
        auto* lev = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
        // Problem-appropriate initial lambda (see the README's
        // initial-damping policy). Env-overridable.
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

    for (size_t i = 0; i < poses.size(); i++) {
        auto* v = new g2o::VertexSE3();
        v->setId((int)i);
        Eigen::Isometry3d est = Eigen::Isometry3d::Identity();
        est.linear() = poses[i].q.toRotationMatrix();
        est.translation() = poses[i].t;
        v->setEstimate(est);
        opt.addVertex(v);
    }
    for (const EdgeIn& e : edges) {
        auto* edge = new g2o::EdgeSE3();
        edge->setVertex(0, opt.vertex(e.a));
        edge->setVertex(1, opt.vertex(e.b));
        edge->setMeasurement(e.meas);
        edge->setInformation(e.info);
        opt.addEdge(edge);
    }
    // Soft unit gauge prior on pose 0, information transformed for
    // EdgeSE3Prior's error frame (see header comment).
    auto* offset = new g2o::ParameterSE3Offset();
    offset->setId(0);
    opt.addParameter(offset);
    auto* prior = new g2o::EdgeSE3Prior();
    prior->setVertex(0, opt.vertex(0));
    Eigen::Isometry3d m0 = Eigen::Isometry3d::Identity();
    m0.linear() = poses[0].q.toRotationMatrix();
    m0.translation() = poses[0].t;
    prior->setMeasurement(m0);
    Eigen::Matrix<double, 6, 6> pinfo = Eigen::Matrix<double, 6, 6>::Identity();
    pinfo.block<3, 3>(3, 3) *= 4.0;
    prior->setInformation(pinfo);
    prior->setParameterId(0, 0);
    opt.addEdge(prior);

    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);
    opt.addPostIterationAction(counter);

    auto t0 = std::chrono::steady_clock::now();
    opt.initializeOptimization();
    // Cross-check hook: with the transformed information, g2o's chi2 is
    // the canonical reference cost -- the harness compares this value
    // at the initial estimate.
    opt.computeActiveErrors();
    double initial_chi2 = opt.chi2();
    int iters = opt.optimize(max_iters);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (out) {
        out->resize(poses.size());
        for (size_t i = 0; i < poses.size(); i++) {
            auto* v = static_cast<g2o::VertexSE3*>(opt.vertex((int)i));
            (*out)[i].t = v->estimate().translation();
            (*out)[i].q = Eigen::Quaterniond(v->estimate().rotation());
        }
    }
    return bench::Result{ms, iters, lm ? counter->trials : iters, initial_chi2};
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <g2o> <lm|gn> <poses_out>\n", argv[0]); return 1; }
    bool lm = std::string(argv[2]) == "lm";
    std::vector<PoseIn> poses;
    std::vector<EdgeIn> edges;
    parse_g2o(argv[1], poses, edges);

    std::vector<PoseIn> result;
    bench::report(
        [&](int n) { return solve(poses, edges, lm, n, nullptr); },
        [&]() { return solve(poses, edges, lm, 100, &result); });

    std::ofstream out(argv[3]);
    for (const PoseIn& p : result) {
        out << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
            << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << "\n";
    }
    return 0;
}

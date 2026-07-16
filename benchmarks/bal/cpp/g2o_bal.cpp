// g2o runner for the BAL benchmark, modeled on g2o's own
// examples/bal/bal_example.cpp: a flat 9-parameter camera vertex
// (additive updates, Rodrigues rotation inside the residual -- the
// same treatment as Ceres's example), a 3-parameter point vertex
// MARGINALIZED so g2o performs its Schur elimination (its sparse-BA
// heritage), CHOLMOD on the reduced camera system, and the Snavely
// reprojection error autodiffed with g2o's bundled AD. Unit
// information, so g2o's chi2 IS the reference cost -- asserted by the
// harness at the initial estimate. Protocol:
//   g2o_bal <problem.txt> <params_out>
// prints the shared benchmark protocol line (see ../../cpp/bench.h) --
// iterations counts every damped lambda trial, accepted the outer Levenberg
// iterations; params_out carries one camera per line (9 values) followed by
// one point per line (3 values).

#include "../../cpp/bench.h"

#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Bal {
    int n_cams = 0, n_points = 0, n_obs = 0;
    std::vector<int> cam_idx, point_idx;
    std::vector<double> xy;
    std::vector<double> cameras; // 9 per camera
    std::vector<double> points;  // 3 per point
};

static Bal load(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    Bal b;
    if (fscanf(f, "%d %d %d", &b.n_cams, &b.n_points, &b.n_obs) != 3) exit(1);
    b.cam_idx.resize(b.n_obs);
    b.point_idx.resize(b.n_obs);
    b.xy.resize(2 * b.n_obs);
    for (int i = 0; i < b.n_obs; i++) {
        if (fscanf(f, "%d %d %lf %lf", &b.cam_idx[i], &b.point_idx[i],
                   &b.xy[2 * i], &b.xy[2 * i + 1]) != 4) exit(1);
    }
    b.cameras.resize(9 * b.n_cams);
    for (double& v : b.cameras) if (fscanf(f, "%lf", &v) != 1) exit(1);
    b.points.resize(3 * b.n_points);
    for (double& v : b.points) if (fscanf(f, "%lf", &v) != 1) exit(1);
    fclose(f);
    return b;
}

class VertexCameraBAL : public g2o::BaseVertex<9, Eigen::Matrix<double, 9, 1>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Map<const Eigen::Matrix<double, 9, 1>>(update);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Map<const Eigen::Vector3d>(update);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class EdgeObservationBAL
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraBAL, VertexPointBAL> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    // The Snavely reprojection error, identically to the reference cost
    // and the Ceres runner. Angle-axis rotation with the standard
    // small-angle branch (branching on the jet's scalar part, as
    // Ceres's AngleAxisRotatePoint does).
    template <typename T>
    bool operator()(const T* camera, const T* point, T* error) const {
        const T* w = camera;
        T p[3];
        const T theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        if (theta2 > T(1e-24)) {
            const T theta = sqrt(theta2);
            const T c = cos(theta);
            const T s = sin(theta);
            const T k[3] = {w[0] / theta, w[1] / theta, w[2] / theta};
            const T kx = k[1] * point[2] - k[2] * point[1];
            const T ky = k[2] * point[0] - k[0] * point[2];
            const T kz = k[0] * point[1] - k[1] * point[0];
            const T kd = k[0] * point[0] + k[1] * point[1] + k[2] * point[2];
            for (int i = 0; i < 3; i++) {
                const T kxp = (i == 0) ? kx : (i == 1) ? ky : kz;
                p[i] = point[i] * c + kxp * s + k[i] * kd * (T(1.0) - c);
            }
        } else {
            // First order: p = X + w x X.
            p[0] = point[0] + w[1] * point[2] - w[2] * point[1];
            p[1] = point[1] + w[2] * point[0] - w[0] * point[2];
            p[2] = point[2] + w[0] * point[1] - w[1] * point[0];
        }
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        const T xp = -p[0] / p[2];
        const T yp = -p[1] / p[2];
        const T r2 = xp * xp + yp * yp;
        const T distortion = T(1.0) + r2 * (camera[7] + camera[8] * r2);
        const T& focal = camera[6];
        error[0] = focal * distortion * xp - T(measurement()(0));
        error[1] = focal * distortion * yp - T(measurement()(1));
        return true;
    }

    G2O_MAKE_AUTO_AD_FUNCTIONS
};

// 6-DOF camera pose vertex (angle-axis, translation) for the covariance mode:
// intrinsics are held at their solved values (known calibration), so the marginal
// is the 6-DOF pose covariance -- matching arael and the Ceres runner.
class VertexCameraPoseBAL : public g2o::BaseVertex<6, Eigen::Matrix<double, 6, 1>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Map<const Eigen::Matrix<double, 6, 1>>(update);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class EdgeObservationPoseBAL
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraPoseBAL, VertexPointBAL> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double f = 0, k1 = 0, k2 = 0;   // this camera's fixed intrinsics
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

    template <typename T>
    bool operator()(const T* camera, const T* point, T* error) const {
        const T* w = camera;
        T p[3];
        const T theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        if (theta2 > T(1e-24)) {
            const T theta = sqrt(theta2);
            const T c = cos(theta);
            const T s = sin(theta);
            const T k[3] = {w[0] / theta, w[1] / theta, w[2] / theta};
            const T kx = k[1] * point[2] - k[2] * point[1];
            const T ky = k[2] * point[0] - k[0] * point[2];
            const T kz = k[0] * point[1] - k[1] * point[0];
            const T kd = k[0] * point[0] + k[1] * point[1] + k[2] * point[2];
            for (int i = 0; i < 3; i++) {
                const T kxp = (i == 0) ? kx : (i == 1) ? ky : kz;
                p[i] = point[i] * c + kxp * s + k[i] * kd * (T(1.0) - c);
            }
        } else {
            p[0] = point[0] + w[1] * point[2] - w[2] * point[1];
            p[1] = point[1] + w[2] * point[0] - w[0] * point[2];
            p[2] = point[2] + w[0] * point[1] - w[1] * point[0];
        }
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        const T xp = -p[0] / p[2];
        const T yp = -p[1] / p[2];
        const T r2 = xp * xp + yp * yp;
        const T distortion = T(1.0) + r2 * (T(k1) + T(k2) * r2);
        error[0] = T(f) * distortion * xp - T(measurement()(0));
        error[1] = T(f) * distortion * yp - T(measurement()(1));
        return true;
    }

    G2O_MAKE_AUTO_AD_FUNCTIONS
};

// g2o keeps its damping retries inside OptimizationAlgorithmLevenberg::solve()
// and reports the count only for the round just finished (levenbergIteration() is
// reset every round), so summing them from a post-iteration action is the one way
// to see the attempts from outside.
//
// The alternative -- g2o's batch statistics -- costs measurable time, so it
// cannot run inside a timed solve; a probe that has to be timed AND counted (the
// first-iteration purity check needs both) can only be served this way.
struct TrialCounter : public g2o::HyperGraphAction {
    g2o::OptimizationAlgorithmLevenberg* lev = nullptr;
    int trials = 0;
    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*,
                                      Parameters* = 0) override {
        if (lev) trials += lev->levenbergIteration();
        return this;
    }
};

static bench::Result solve(const Bal& b, int max_iters, std::vector<double>* cams_out,
                           std::vector<double>* points_out) {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>>;
    auto linear = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    auto* lev = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
    // g2o's auto lambda heuristic by default -- BA is the problem family
    // it was built for. Env-overridable like the other runners.
    if (const char* li = getenv("G2O_LAMBDA_INIT")) lev->setUserLambdaInit(atof(li));
    auto* counter = new TrialCounter();
    counter->lev = lev;

    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    opt.setAlgorithm(lev);
    opt.addPostIterationAction(counter);

    std::vector<VertexCameraBAL*> cams(b.n_cams);
    for (int i = 0; i < b.n_cams; i++) {
        auto* v = new VertexCameraBAL();
        v->setId(i);
        v->setEstimate(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(&b.cameras[9 * i]));
        opt.addVertex(v);
        cams[i] = v;
    }
    // Schur: marginalize the landmarks so g2o solves the reduced camera
    // system (its bal_example default). G2O_MARGINALIZE=0 turns it off --
    // CHOLMOD then factorizes the full camera+point system (for comparison;
    // far larger, expected to be much slower).
    bool marginalize = true;
    if (const char* m = getenv("G2O_MARGINALIZE")) marginalize = atoi(m) != 0;
    std::vector<VertexPointBAL*> pts(b.n_points);
    for (int i = 0; i < b.n_points; i++) {
        auto* v = new VertexPointBAL();
        v->setId(b.n_cams + i);
        v->setEstimate(Eigen::Map<const Eigen::Vector3d>(&b.points[3 * i]));
        v->setMarginalized(marginalize);
        opt.addVertex(v);
        pts[i] = v;
    }
    for (int i = 0; i < b.n_obs; i++) {
        auto* e = new EdgeObservationBAL();
        e->setVertex(0, cams[b.cam_idx[i]]);
        e->setVertex(1, pts[b.point_idx[i]]);
        e->setMeasurement(Eigen::Vector2d(b.xy[2 * i], b.xy[2 * i + 1]));
        e->setInformation(Eigen::Matrix2d::Identity());
        opt.addEdge(e);
    }

    // Same termination class as the other systems.
    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);

    auto t0 = std::chrono::steady_clock::now();
    opt.initializeOptimization();
    // Unit information: chi2 IS the reference cost; the harness asserts
    // this at the initial estimate.
    opt.computeActiveErrors();
    double initial_chi2 = opt.chi2();
    int iters = opt.optimize(max_iters);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (cams_out) {
        cams_out->resize(9 * b.n_cams);
        for (int i = 0; i < b.n_cams; i++) {
            Eigen::Map<Eigen::Matrix<double, 9, 1>> m(&(*cams_out)[9 * i]);
            m = cams[i]->estimate();
        }
    }
    if (points_out) {
        points_out->resize(3 * b.n_points);
        for (int i = 0; i < b.n_points; i++) {
            Eigen::Map<Eigen::Vector3d> m(&(*points_out)[3 * i]);
            m = pts[i]->estimate();
        }
    }
    return bench::Result{ms, iters, counter->trials, initial_chi2};
}

// Covariance mode, mirroring arael's bal cov_bench and the Ceres runner: known
// calibration (intrinsics fixed -> 6-DOF pose vertices), gauge fix (cameras 0 and
// 1 held constant), landmarks marginalized so g2o solves the Schur-reduced camera
// system, then computeMarginals for the free-camera pose covariances. This is a
// sparse marginal inverse over CHOLMOD -- the same algorithm family as arael's,
// and robust to BAL's weak point depths (no prior needed).
static int cov_mode(const Bal& b) {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
    auto linear = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    auto* lev = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
    if (const char* li = getenv("G2O_LAMBDA_INIT")) lev->setUserLambdaInit(atof(li));

    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    opt.setAlgorithm(lev);

    std::vector<VertexCameraPoseBAL*> cams(b.n_cams);
    for (int i = 0; i < b.n_cams; i++) {
        auto* v = new VertexCameraPoseBAL();
        v->setId(i);
        Eigen::Matrix<double, 6, 1> pose;
        pose << b.cameras[9 * i], b.cameras[9 * i + 1], b.cameras[9 * i + 2],
                b.cameras[9 * i + 3], b.cameras[9 * i + 4], b.cameras[9 * i + 5];
        v->setEstimate(pose);
        v->setFixed(i == 0 || i == 1);   // gauge fix: cameras 0 and 1 held constant
        opt.addVertex(v);
        cams[i] = v;
    }
    std::vector<VertexPointBAL*> pts(b.n_points);
    for (int i = 0; i < b.n_points; i++) {
        auto* v = new VertexPointBAL();
        v->setId(b.n_cams + i);
        v->setEstimate(Eigen::Map<const Eigen::Vector3d>(&b.points[3 * i]));
        v->setMarginalized(true);
        opt.addVertex(v);
        pts[i] = v;
    }
    for (int i = 0; i < b.n_obs; i++) {
        int ci = b.cam_idx[i];
        auto* e = new EdgeObservationPoseBAL();
        e->f = b.cameras[9 * ci + 6];
        e->k1 = b.cameras[9 * ci + 7];
        e->k2 = b.cameras[9 * ci + 8];
        e->setVertex(0, cams[ci]);
        e->setVertex(1, pts[b.point_idx[i]]);
        e->setMeasurement(Eigen::Vector2d(b.xy[2 * i], b.xy[2 * i + 1]));
        e->setInformation(Eigen::Matrix2d::Identity());
        opt.addEdge(e);
    }

    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(bench::full_iters(100));
    opt.addPostIterationAction(terminate);

    opt.initializeOptimization();
    opt.optimize(bench::full_iters(100));

    // Timing: for each N, the cost of computeMarginals over N spread camera
    // poses. g2o reuses the Schur factor from the solve it just ran (warm), and
    // its points are marginalized -- so it recovers camera poses only, no point
    // covariance. Machine-readable COV lines for the Rust harness.
    double budget_s = getenv("COV_BUDGET_S") ? atof(getenv("COV_BUDGET_S")) : 5.0;
    int cap = getenv("COV_CAP") ? atoi(getenv("COV_CAP")) : 200;
    double cell_cap = bench::cov_cell_cap_ms();
    const int free_cams = b.n_cams - 2;
    const int ns[] = {1, 2, 8, 32};
    std::vector<int> queryN(ns, ns + 4);
    queryN.push_back(free_cams);  // "all"
    int prev_n = 0;
    double prev_ms = 0;
    for (int N : queryN) {
        if (N > free_cams) continue;
        if (bench::cov_project_ms(prev_n, prev_ms, N) > cell_cap) {
            printf("COV cam %d toolong 0\n", N);
            fflush(stdout);
            continue;
        }
        std::vector<int> idx = bench::spread(2, free_cams, N);
        g2o::OptimizableGraph::VertexContainer query;
        for (int i : idx) query.push_back(cams[i]);
        bool ok = true;
        int reps = 0;
        double ms = bench::median_ms(budget_s, cap, &reps, [&] {
            g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
            if (!opt.computeMarginals(spinv, query)) ok = false;
        });
        if (!ok)                printf("COV cam %d nan 0\n", N);
        else if (ms > cell_cap) printf("COV cam %d toolong %d\n", N, reps);
        else                    printf("COV cam %d %.3f %d\n", N, ms, reps);
        fflush(stdout);
        prev_n = N;
        prev_ms = ms;
    }

    // One std-dev line for validation: camera 2's 6-DOF pose. Pose layout is
    // [angle-axis(0..2), t(3..5)] -- print translation, then rotation.
    g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
    g2o::OptimizableGraph::VertexContainer one{cams[2]};
    if (opt.computeMarginals(spinv, one)) {
        const Eigen::MatrixXd* c = spinv.block(cams[2]->hessianIndex(), cams[2]->hessianIndex());
        if (c)
            fprintf(stderr, "  g2o camera[2] std dev: t=(%.4f,%.4f,%.4f) rot=(%.5f,%.5f,%.5f)\n",
                    std::sqrt((*c)(3, 3)), std::sqrt((*c)(4, 4)), std::sqrt((*c)(5, 5)),
                    std::sqrt((*c)(0, 0)), std::sqrt((*c)(1, 1)), std::sqrt((*c)(2, 2)));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <problem.txt> <params_out|cov>\n", argv[0]); return 1; }
    Bal b = load(argv[1]);

    if (std::string(argv[2]) == "cov") return cov_mode(b);

    std::vector<double> cams, points;
    bench::report(
        [&](int n) { return solve(b, n, nullptr, nullptr); },
        [&]() { return solve(b, bench::full_iters(100), &cams, &points); });

    std::ofstream out(argv[2]);
    out.precision(17);
    for (int i = 0; i < b.n_cams; i++) {
        for (int k = 0; k < 9; k++) out << cams[9 * i + k] << (k == 8 ? "\n" : " ");
    }
    for (int i = 0; i < b.n_points; i++) {
        for (int k = 0; k < 3; k++) out << points[3 * i + k] << (k == 2 ? "\n" : " ");
    }
    return 0;
}

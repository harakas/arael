// g2o reference for the plane SLAM benchmark, using the SHIPPED g2o types
// (the application is g2o's own plane_slam example): VertexSE3 poses,
// VertexPlane landmarks, EdgeSE3 odometry, EdgeSE3PlaneSensorCalib plane
// observations with the sensor offset vertex fixed at identity.
//
//   g2o_plane <scene.txt> <solution_out>
//
// Probes + protocol line via ../../cpp/bench.h. G2O_LAMBDA_INIT overrides the
// initial damping (default 1e-9, near-Gauss-Newton); G2O_GAIN overrides the
// terminate gain (default 1e-5, the shared termination class).

#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "../../cpp/bench.h"

using namespace g2o;

// Odometry between-edge with the benchmark's shared residual (mirrored
// exactly by the arael runner):
//   err_t = R_i^T (t_j - t_i) - t_m
//   err_r = vee((dR - dR^T)/2),  dR = R_m^T R_i^T R_j
// (The shipped EdgeSE3 uses the error-quaternion vector part instead; a
// custom edge keeps the two runners' costs identical, the same practice as
// the slam benchmark's custom edges.)
class EdgeOdo : public BaseBinaryEdge<6, Eigen::Isometry3d, VertexSE3, VertexSE3> {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void computeError() override {
        const VertexSE3* vi = static_cast<const VertexSE3*>(_vertices[0]);
        const VertexSE3* vj = static_cast<const VertexSE3*>(_vertices[1]);
        Eigen::Matrix3d Ri = vi->estimate().linear();
        Eigen::Matrix3d Rj = vj->estimate().linear();
        Eigen::Vector3d ti = vi->estimate().translation();
        Eigen::Vector3d tj = vj->estimate().translation();
        Eigen::Matrix3d Rm = _measurement.linear();
        Eigen::Vector3d tm = _measurement.translation();
        Eigen::Vector3d et = Ri.transpose() * (tj - ti) - tm;
        Eigen::Matrix3d dR = Rm.transpose() * Ri.transpose() * Rj;
        _error << et.x(), et.y(), et.z(),
            0.5 * (dR(2, 1) - dR(1, 2)),
            0.5 * (dR(0, 2) - dR(2, 0)),
            0.5 * (dR(1, 0) - dR(0, 1));
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

struct ScenePose { Eigen::Vector3d t; Eigen::Quaterniond q; };
struct SceneOdo { int i, j; ScenePose rel; double info[6]; };
struct SceneObs { int p, l; Eigen::Vector4d plane; double info[3]; };

struct Scene {
    std::vector<ScenePose> poses;
    std::vector<Eigen::Vector4d> planes;
    std::vector<SceneOdo> odo;
    std::vector<SceneObs> obs;
};

static Scene load(const char* path) {
    Scene s;
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "pose") {
            ScenePose p; double qw, qx, qy, qz;
            ss >> p.t.x() >> p.t.y() >> p.t.z() >> qw >> qx >> qy >> qz;
            p.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            s.poses.push_back(p);
        } else if (tag == "plane") {
            Eigen::Vector4d v; ss >> v(0) >> v(1) >> v(2) >> v(3);
            s.planes.push_back(v);
        } else if (tag == "odom") {
            SceneOdo o; double qw, qx, qy, qz;
            ss >> o.i >> o.j >> o.rel.t.x() >> o.rel.t.y() >> o.rel.t.z() >> qw >> qx >> qy >> qz;
            for (double& v : o.info) ss >> v;
            o.rel.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            s.odo.push_back(o);
        } else if (tag == "obs") {
            SceneObs b;
            ss >> b.p >> b.l >> b.plane(0) >> b.plane(1) >> b.plane(2) >> b.plane(3);
            for (double& v : b.info) ss >> v;
            s.obs.push_back(b);
        }
    }
    return s;
}

// g2o keeps its damping retries inside OptimizationAlgorithmLevenberg::solve()
// and reports the count only for the round just finished (levenbergIteration()
// is reset every round), so summing them from a post-iteration action is the
// one way to see the attempts from outside.
struct TrialCounter : public g2o::HyperGraphAction {
    g2o::OptimizationAlgorithmLevenberg* lev = nullptr;
    int trials = 0;
    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*,
                                      Parameters* = 0) override {
        if (lev) trials += lev->levenbergIteration();
        return this;
    }
};

static bench::Result solve(const Scene& s, int max_iters,
                           std::vector<double>* pose_out,
                           std::vector<double>* plane_out) {
    const int n = (int)s.poses.size(), m = (int)s.planes.size();
    using LS = LinearSolverEigen<BlockSolverX::PoseMatrixType>;
    auto* lev = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverX>(std::make_unique<LS>()));
    // Problem-appropriate initial damping (near-Gauss-Newton on this
    // well-initialized graph), matching the sibling benchmark policy.
    double lambda0 = 1e-9;
    if (const char* li = getenv("G2O_LAMBDA_INIT")) lambda0 = atof(li);
    lev->setUserLambdaInit(lambda0);

    SparseOptimizer opt;
    opt.setVerbose(getenv("G2O_VERBOSE") != nullptr);
    opt.setAlgorithm(lev);

    for (int i = 0; i < n; i++) {
        auto* v = new VertexSE3();
        v->setId(i);
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = s.poses[i].q.toRotationMatrix();
        T.translation() = s.poses[i].t;
        v->setEstimate(T);
        if (i == 0) v->setFixed(true);
        opt.addVertex(v);
    }
    for (int j = 0; j < m; j++) {
        auto* v = new VertexPlane();
        v->setId(n + j);
        v->setEstimate(Plane3D(s.planes[j]));
        opt.addVertex(v);
    }
    auto* offset = new VertexSE3();
    offset->setId(n + m);
    offset->setEstimate(Eigen::Isometry3d::Identity());
    offset->setFixed(true);
    opt.addVertex(offset);

    for (const auto& o : s.odo) {
        auto* e = new EdgeOdo();
        e->setVertex(0, opt.vertex(o.i));
        e->setVertex(1, opt.vertex(o.j));
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = o.rel.q.toRotationMatrix();
        T.translation() = o.rel.t;
        e->setMeasurement(T);
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
        for (int k = 0; k < 6; k++) info(k, k) = o.info[k];
        e->setInformation(info);
        opt.addEdge(e);
    }
    for (const auto& b : s.obs) {
        auto* e = new EdgeSE3PlaneSensorCalib();
        e->setVertex(0, opt.vertex(b.p));
        e->setVertex(1, opt.vertex(n + b.l));
        e->setVertex(2, offset);
        e->setMeasurement(Plane3D(b.plane));
        Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
        for (int k = 0; k < 3; k++) info(k, k) = b.info[k];
        e->setInformation(info);
        opt.addEdge(e);
    }

    auto* counter = new TrialCounter();
    counter->lev = lev;
    opt.addPostIterationAction(counter);
    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;  // shared termination class
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);

    opt.initializeOptimization();
    opt.computeActiveErrors();
    double chi0 = opt.activeChi2();
    double t0 = bench::now_ms();
    int iters = opt.optimize(max_iters);
    double ms = bench::now_ms() - t0;

    if (pose_out) {
        for (int i = 0; i < n; i++) {
            auto* v = static_cast<VertexSE3*>(opt.vertex(i));
            Eigen::Quaterniond q(v->estimate().linear());
            Eigen::Vector3d t = v->estimate().translation();
            double vals[7] = {t.x(), t.y(), t.z(), q.w(), q.x(), q.y(), q.z()};
            pose_out->insert(pose_out->end(), vals, vals + 7);
        }
        for (int j = 0; j < m; j++) {
            auto* v = static_cast<VertexPlane*>(opt.vertex(n + j));
            Eigen::Vector4d c = v->estimate().toVector();
            double vals[4] = {c(0), c(1), c(2), c(3)};
            plane_out->insert(plane_out->end(), vals, vals + 4);
        }
    }
    return bench::Result{ms, iters, counter->trials, chi0};
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <scene.txt> <solution_out>\n", argv[0]); return 1; }
    Scene s = load(argv[1]);

    std::vector<double> poses, planes;
    bench::report(
        [&](int iters) { return solve(s, iters, nullptr, nullptr); },
        [&]() { return solve(s, bench::full_iters(200), &poses, &planes); });

    std::ofstream out(argv[2]);
    out.precision(17);
    for (size_t i = 0; i + 6 < poses.size(); i += 7)
        out << poses[i] << " " << poses[i + 1] << " " << poses[i + 2] << " "
            << poses[i + 3] << " " << poses[i + 4] << " " << poses[i + 5] << " "
            << poses[i + 6] << "\n";
    for (size_t j = 0; j + 3 < planes.size(); j += 4)
        out << planes[j] << " " << planes[j + 1] << " " << planes[j + 2] << " "
            << planes[j + 3] << "\n";
    return 0;
}

// g2o reference for the plane SLAM benchmark, using the SHIPPED g2o types
// (the application is g2o's own plane_slam example): VertexSE3 poses,
// VertexPlane landmarks, EdgeSE3 odometry, EdgeSE3PlaneSensorCalib plane
// observations with the sensor offset vertex fixed at identity.
//
//   g2o_plane <scene.txt> [solution_out] [max_iters]
//
// Prints initial/final chi2, iterations and wall time; writes the solution
// as N pose lines "tx ty tz qw qx qy qz" then M plane lines "nx ny nz c".

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

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

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <scene.txt> [solution_out] [max_iters]\n", argv[0]); return 1; }
    int max_iters = argc > 3 ? atoi(argv[3]) : 100;

    std::vector<ScenePose> poses;
    std::vector<Eigen::Vector4d> planes;
    std::vector<SceneOdo> odo;
    std::vector<SceneObs> obs;
    {
        std::ifstream in(argv[1]);
        if (!in) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string tag; ss >> tag;
            if (tag == "pose") {
                ScenePose p; double qw, qx, qy, qz;
                ss >> p.t.x() >> p.t.y() >> p.t.z() >> qw >> qx >> qy >> qz;
                p.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                poses.push_back(p);
            } else if (tag == "plane") {
                Eigen::Vector4d v; ss >> v(0) >> v(1) >> v(2) >> v(3);
                planes.push_back(v);
            } else if (tag == "odom") {
                SceneOdo o; double qw, qx, qy, qz;
                ss >> o.i >> o.j >> o.rel.t.x() >> o.rel.t.y() >> o.rel.t.z() >> qw >> qx >> qy >> qz;
                for (double& v : o.info) ss >> v;
                o.rel.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
                odo.push_back(o);
            } else if (tag == "obs") {
                SceneObs b;
                ss >> b.p >> b.l >> b.plane(0) >> b.plane(1) >> b.plane(2) >> b.plane(3);
                for (double& v : b.info) ss >> v;
                obs.push_back(b);
            }
        }
    }
    const int n = (int)poses.size(), m = (int)planes.size();
    fprintf(stderr, "scene: %d poses, %d planes, %zu odom, %zu obs\n", n, m, odo.size(), obs.size());

    SparseOptimizer opt;
    using LS = LinearSolverEigen<BlockSolverX::PoseMatrixType>;
    opt.setAlgorithm(new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverX>(std::make_unique<LS>())));

    for (int i = 0; i < n; i++) {
        auto* v = new VertexSE3();
        v->setId(i);
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = poses[i].q.toRotationMatrix();
        T.translation() = poses[i].t;
        v->setEstimate(T);
        if (i == 0) v->setFixed(true);
        opt.addVertex(v);
    }
    for (int j = 0; j < m; j++) {
        auto* v = new VertexPlane();
        v->setId(n + j);
        v->setEstimate(Plane3D(planes[j]));
        opt.addVertex(v);
    }
    auto* offset = new VertexSE3();
    offset->setId(n + m);
    offset->setEstimate(Eigen::Isometry3d::Identity());
    offset->setFixed(true);
    opt.addVertex(offset);

    for (const auto& o : odo) {
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
    for (const auto& b : obs) {
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

    opt.setVerbose(true);
    opt.initializeOptimization();
    opt.computeActiveErrors();
    double chi0 = opt.activeChi2();
    auto t0 = std::chrono::steady_clock::now();
    int iters = opt.optimize(max_iters);
    auto t1 = std::chrono::steady_clock::now();
    opt.computeActiveErrors();
    printf("chi2 %.6f -> %.6f in %d iterations, %.1f ms\n",
        chi0, opt.activeChi2(), iters,
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    if (argc > 2) {
        std::ofstream out(argv[2]);
        out.precision(17);
        for (int i = 0; i < n; i++) {
            auto* v = static_cast<VertexSE3*>(opt.vertex(i));
            Eigen::Quaterniond q(v->estimate().linear());
            Eigen::Vector3d t = v->estimate().translation();
            out << t.x() << " " << t.y() << " " << t.z() << " "
                << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
        }
        for (int j = 0; j < m; j++) {
            auto* v = static_cast<VertexPlane*>(opt.vertex(n + j));
            Eigen::Vector4d c = v->estimate().toVector();
            out << c(0) << " " << c(1) << " " << c(2) << " " << c(3) << "\n";
        }
    }
    return 0;
}

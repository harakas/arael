// Ceres runner for the 3D pose-graph benchmark, modeled on Ceres's own
// examples/slam/pose_graph_3d (quaternion parameter blocks on
// EigenQuaternionManifold, autodiffed 6-residual between terms, full
// 6x6 sqrt-information). The error-quaternion composition order is the
// benchmark's canonical q_meas^-1 q_a^-1 q_b (Ceres's example composes
// the same family the other way around, which under a coupled
// information matrix would weight the rotation residual in a different
// frame). Protocol:
//   ceres_bench3 <file.g2o> <poses_out>
// prints JSON {solve_ms, first_iter_ms, iterations, accepted,
// initial_cost, cpus_allowed}, writes "x y z qx qy qz qw" lines.
// The 3D datasets always use the file's information matrices.

#include "../../cpp/bench.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
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
    Eigen::Vector3d dt;
    Eigen::Quaterniond dq;
    Eigen::Matrix<double, 6, 6> u; // upper sqrt info, info = u^T u
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
            e.dt = Eigen::Vector3d(x, y, z);
            e.dq = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            Eigen::Matrix<double, 6, 6> info;
            for (int i = 0; i < 6; i++) {
                for (int j = i; j < 6; j++) {
                    double v;
                    ss >> v;
                    info(i, j) = v;
                    info(j, i) = v;
                }
            }
            // info = u^T u with u upper triangular -- the same factor
            // convention as the reference cost.
            e.u = info.llt().matrixU();
            edges.push_back(e);
        }
    }
}

// Canonical between residual (see the benchmark README):
//   r = U * [ R_a^T (t_b - t_a) - t_meas ; 2 * vec(q_meas^-1 q_a^-1 q_b) ]
// with the rotation part on its qw >= 0 branch.
struct BetweenError3 {
    explicit BetweenError3(const EdgeIn& e) : e_(e) {}
    template <typename T>
    bool operator()(const T* pa_ptr, const T* qa_ptr, const T* pb_ptr, const T* qb_ptr,
                    T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pa(pa_ptr), pb(pb_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> qa(qa_ptr), qb(qb_ptr);
        const Eigen::Quaternion<T> qa_inv = qa.conjugate();
        const Eigen::Matrix<T, 3, 1> terr =
            qa_inv * (pb - pa) - e_.dt.template cast<T>();
        Eigen::Quaternion<T> qerr =
            e_.dq.template cast<T>().conjugate() * qa_inv * qb;
        const T two = qerr.w() >= T(0) ? T(2) : T(-2);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        residuals.template head<3>() = terr;
        residuals.template tail<3>() = two * qerr.vec();
        residuals.applyOnTheLeft(e_.u.template cast<T>());
        return true;
    }
    EdgeIn e_;
};

// Unit-weight gauge prior on pose 0 (identical convention to the other
// runners).
struct PriorError3 {
    explicit PriorError3(const PoseIn& p) : p_(p) {}
    template <typename T>
    bool operator()(const T* pa_ptr, const T* qa_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pa(pa_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> qa(qa_ptr);
        Eigen::Quaternion<T> qerr = p_.q.template cast<T>().conjugate() * qa;
        const T two = qerr.w() >= T(0) ? T(2) : T(-2);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        residuals.template head<3>() = pa - p_.t.template cast<T>();
        residuals.template tail<3>() = two * qerr.vec();
        return true;
    }
    PoseIn p_;
};

static bench::Result solve(const std::vector<PoseIn>& poses_in, const std::vector<EdgeIn>& edges,
                       int max_iters, std::vector<PoseIn>* out) {
    std::vector<double> t(poses_in.size() * 3), q(poses_in.size() * 4);
    for (size_t i = 0; i < poses_in.size(); i++) {
        Eigen::Map<Eigen::Vector3d> mt(&t[3 * i]);
        mt = poses_in[i].t;
        Eigen::Map<Eigen::Vector4d> mq(&q[4 * i]);
        mq = poses_in[i].q.coeffs();
    }
    ceres::Problem problem;
    ceres::Manifold* quat = new ceres::EigenQuaternionManifold;
    for (size_t i = 0; i < poses_in.size(); i++) {
        problem.AddParameterBlock(&t[3 * i], 3);
        problem.AddParameterBlock(&q[4 * i], 4, quat);
    }
    for (const EdgeIn& e : edges) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BetweenError3, 6, 3, 4, 3, 4>(new BetweenError3(e)),
            nullptr, &t[3 * e.a], &q[4 * e.a], &t[3 * e.b], &q[4 * e.b]);
    }
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PriorError3, 6, 3, 4>(new PriorError3(poses_in[0])),
        nullptr, &t[0], &q[0]);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_iters;
    options.num_threads = 1;
    // Same termination class as the other systems.
    options.function_tolerance = 1e-5;
    // Problem-appropriate initial trust region (shipped default 1e4
    // over-damps; see the README's initial-damping policy).
    options.initial_trust_region_radius = 1e12;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);
    options.max_trust_region_radius = 1e32;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (out) {
        out->resize(poses_in.size());
        for (size_t i = 0; i < poses_in.size(); i++) {
            (*out)[i].t = Eigen::Map<Eigen::Vector3d>(&t[3 * i]);
            (*out)[i].q = Eigen::Quaterniond(Eigen::Map<Eigen::Vector4d>(&q[4 * i]));
        }
    }
    // Ceres records iteration 0 -- the initial cost evaluation, before any step
    // -- as a successful step. It is not one; discount it so accepted/total
    // mean the same here as in every other runner.
    const int accepted = std::max(0, summary.num_successful_steps - 1);
    return bench::Result{summary.total_time_in_seconds * 1e3, accepted,
                         accepted + summary.num_unsuccessful_steps,
                         2.0 * summary.initial_cost};
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <g2o> <poses_out>\n", argv[0]); return 1; }
    std::vector<PoseIn> poses;
    std::vector<EdgeIn> edges;
    parse_g2o(argv[1], poses, edges);

    std::vector<PoseIn> result;
    bench::report(
        [&](int n) { return solve(poses, edges, n, nullptr); },
        [&]() { return solve(poses, edges, 100, &result); });

    std::ofstream out(argv[2]);
    for (const PoseIn& p : result) {
        out << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
            << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << "\n";
    }

    return 0;
}

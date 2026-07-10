// g2o runner for the localization benchmark. Reads the scene exported by the
// Rust harness (identical problem; same file Ceres reads) and builds the four
// factor types as CUSTOM g2o edges with ANALYTIC Jacobians (linearizeOplus) --
// no numeric differentiation, so g2o runs at full performance. Pose vertices
// are plain 6-vectors [x,y,z,roll,pitch,yaw] with additive oplus (matching
// arael's SimpleEulerAngleParam). Landmarks are FIXED constants, so the
// bearing edge is UNARY over the pose. The euler convention and odometry euler
// extraction reproduce arael's matrix3 twins, so every residual equals
// scene::reference_cost; with identity information the graph chi2 equals the
// reference cost exactly (asserted by the harness via initial_cost). The
// analytic Jacobians are checked against finite differences when
// G2O_VERIFY_JAC=1 (verification only, never in the timed solve).
//
//   g2o_loc <scene.txt> <lm|gn> <solution_out>
// JSON {solve_ms, first_iter_ms, iterations, accepted, initial_cost,
// peak_rss_kb, cpus_allowed} on stdout -- iterations counts every damped
// lambda trial (from an untimed statistics pass), accepted the outer
// iterations; solution_out carries one pose per line (6 values). Landmarks
// are fixed and not written.
// G2O_LAMBDA_INIT overrides the LM initial damping (default 1e-9); G2O_GAIN
// overrides the terminate gain (default 1e-5, the shared termination class).

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/batch_stats.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/jacobian_workspace.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using V6 = Eigen::Matrix<double, 6, 1>;
using V3 = Eigen::Vector3d;
using V2 = Eigen::Vector2d;
using M3 = Eigen::Matrix3d;

static M3 skew(const V3& v) {
    M3 m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

// arael's rotation convention (x=roll, y=pitch, z=yaw; Rz Ry Rx).
static M3 euler_rot(double roll, double pitch, double yaw) {
    double sx = sin(roll), cx = cos(roll), sy = sin(pitch), cy = cos(pitch),
           sz = sin(yaw), cz = cos(yaw);
    M3 R;
    R << cy * cz, -cx * sz + cz * sx * sy, cx * cz * sy + sx * sz,
         cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx,
        -sy, cy * sx, cx * cy;
    return R;
}

// R plus its derivatives wrt (roll, pitch, yaw). dR/dtheta = skew(axis) R.
static void euler_rot_d(double roll, double pitch, double yaw, M3& R, M3 dR[3]) {
    R = euler_rot(roll, pitch, yaw);
    double sy = sin(pitch), cy = cos(pitch), sz = sin(yaw), cz = cos(yaw);
    dR[0] = skew(V3(cz * cy, sz * cy, -sy)) * R;  // d/droll
    dR[1] = skew(V3(-sz, cz, 0)) * R;             // d/dpitch
    dR[2] = skew(V3(0, 0, 1)) * R;                // d/dyaw
}

// arael's get_euler_angles main branch (error rotation near identity).
static V3 rot_to_euler(const M3& m) {
    return V3(atan2(m(2, 1), m(2, 2)), -asin(m(2, 0)), atan2(m(1, 0), m(0, 0)));
}

static V3 rot_to_euler_d(const M3& m, const M3& dm) {
    double dr = (m(2, 2) * dm(2, 1) - m(2, 1) * dm(2, 2)) /
                (m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2));
    double dp = -dm(2, 0) / sqrt(1.0 - m(2, 0) * m(2, 0));
    double dy = (m(0, 0) * dm(1, 0) - m(1, 0) * dm(0, 0)) /
                (m(1, 0) * m(1, 0) + m(0, 0) * m(0, 0));
    return V3(dr, dp, dy);
}

#define IO_STUB \
    bool read(std::istream&) override { return false; } \
    bool write(std::ostream&) const override { return false; }
#define EDGE_IO \
    IO_STUB \
    Eigen::MatrixXd jxi() { return _jacobianOplusXi; }
#define EDGE_IO_B \
    EDGE_IO \
    Eigen::MatrixXd jxj() { return _jacobianOplusXj; }

class VertexPose : public g2o::BaseVertex<6, V6> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate.setZero(); }
    void oplusImpl(const double* u) override {
        for (int i = 0; i < 6; i++) _estimate[i] += u[i];
    }
    IO_STUB
};

// -- the four edges (residual == scene::reference_cost, identity information;
//    analytic linearizeOplus) --

class EdgeDrift : public g2o::BaseUnaryEdge<6, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V6 prior;
    double pi, ei;
    void computeError() override {
        const V6& p = static_cast<VertexPose*>(_vertices[0])->estimate();
        _error.head<3>() = (p.head<3>() - prior.head<3>()) * pi;
        _error.tail<3>() = (p.tail<3>() - prior.tail<3>()) * ei;
    }
    void linearizeOplus() override {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.diagonal() << pi, pi, pi, ei, ei, ei;
    }
    EDGE_IO
};

class EdgeTilt : public g2o::BaseUnaryEdge<2, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double roll_m, pitch_m, isigma;
    void computeError() override {
        const V6& p = static_cast<VertexPose*>(_vertices[0])->estimate();
        _error[0] = (p[3] - roll_m) * isigma;
        _error[1] = (p[4] - pitch_m) * isigma;
    }
    void linearizeOplus() override {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi(0, 3) = isigma;
        _jacobianOplusXi(1, 4) = isigma;
    }
    EDGE_IO
};

// Bearing to a FIXED landmark -- unary edge over the pose.
class EdgeBearing : public g2o::BaseUnaryEdge<2, double, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3 lm;   // fixed landmark
    M3 mf2r;
    V3 camera_pos;
    V2 isigma;
    double scale;
    void computeError() override {
        const V6& pose = static_cast<VertexPose*>(_vertices[0])->estimate();
        M3 R = euler_rot(pose[3], pose[4], pose[5]);
        V3 d = lm - pose.head<3>();
        V3 r_f = mf2r.transpose() * (R.transpose() * d - camera_pos);
        _error[0] = atan2(r_f[1], r_f[0]) * isigma[0] * scale;
        _error[1] = atan2(r_f[2], r_f[0]) * isigma[1] * scale;
    }
    void linearizeOplus() override {
        const V6& pose = static_cast<VertexPose*>(_vertices[0])->estimate();
        M3 R, dR[3];
        euler_rot_d(pose[3], pose[4], pose[5], R, dR);
        V3 d = lm - pose.head<3>();
        V3 r_f = mf2r.transpose() * (R.transpose() * d - camera_pos);
        double s0 = isigma[0] * scale, s1 = isigma[1] * scale;
        double d0 = r_f[0] * r_f[0] + r_f[1] * r_f[1];
        double d1 = r_f[0] * r_f[0] + r_f[2] * r_f[2];
        Eigen::Matrix<double, 2, 3> dr_drf;
        dr_drf << s0 * (-r_f[1] / d0), s0 * (r_f[0] / d0), 0,
                  s1 * (-r_f[2] / d1), 0, s1 * (r_f[0] / d1);
        Eigen::Matrix<double, 2, 3> dr_drr = dr_drf * mf2r.transpose();
        // pose: position (d d / d pose_pos = -I) and euler (via dR^T d)
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<2, 3>(0, 0) = -dr_drr * R.transpose();
        for (int k = 0; k < 3; k++)
            _jacobianOplusXi.block<2, 1>(0, 3 + k) = dr_drr * (dR[k].transpose() * d);
    }
    EDGE_IO
};

// vertices: (0) prev pose, (1) cur pose
class EdgeOdo : public g2o::BaseBinaryEdge<6, double, VertexPose, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3 dpos, dea, pci, eci;
    M3 pcr, ecr;
    void computeError() override {
        const V6& prev = static_cast<VertexPose*>(_vertices[0])->estimate();
        const V6& cur = static_cast<VertexPose*>(_vertices[1])->estimate();
        M3 Rp = euler_rot(prev[3], prev[4], prev[5]);
        M3 Rc = euler_rot(cur[3], cur[4], cur[5]);
        V3 pos_err = Rp.transpose() * (cur.head<3>() - prev.head<3>()) - dpos;
        V3 pos_w = pcr.transpose() * pos_err;
        M3 err_rot = (Rp * euler_rot(dea[0], dea[1], dea[2])).transpose() * Rc;
        V3 ea_w = ecr.transpose() * rot_to_euler(err_rot);
        _error.head<3>() = pos_w.cwiseProduct(pci);
        _error.tail<3>() = ea_w.cwiseProduct(eci);
    }
    void linearizeOplus() override {
        const V6& prev = static_cast<VertexPose*>(_vertices[0])->estimate();
        const V6& cur = static_cast<VertexPose*>(_vertices[1])->estimate();
        M3 Rp, dRp[3], Rc, dRc[3];
        euler_rot_d(prev[3], prev[4], prev[5], Rp, dRp);
        euler_rot_d(cur[3], cur[4], cur[5], Rc, dRc);
        M3 A = euler_rot(dea[0], dea[1], dea[2]).transpose();
        V3 d = cur.head<3>() - prev.head<3>();
        M3 M = A * Rp.transpose() * Rc;  // error rotation
        M3 W = pci.asDiagonal() * pcr.transpose();
        M3 E = eci.asDiagonal() * ecr.transpose();

        _jacobianOplusXi.setZero();  // wrt prev
        _jacobianOplusXj.setZero();  // wrt cur
        _jacobianOplusXi.block<3, 3>(0, 0) = W * (-Rp.transpose());
        _jacobianOplusXj.block<3, 3>(0, 0) = W * Rp.transpose();
        for (int k = 0; k < 3; k++)
            _jacobianOplusXi.block<3, 1>(0, 3 + k) = W * (dRp[k].transpose() * d);
        for (int k = 0; k < 3; k++) {
            M3 dM_prev = A * dRp[k].transpose() * Rc;
            M3 dM_cur = A * Rp.transpose() * dRc[k];
            _jacobianOplusXi.block<3, 1>(3, 3 + k) = E * rot_to_euler_d(M, dM_prev);
            _jacobianOplusXj.block<3, 1>(3, 3 + k) = E * rot_to_euler_d(M, dM_cur);
        }
    }
    EDGE_IO_B
};

// ---------------------------------------------------------------------------
// Scene (identical format to the Ceres runner)
// ---------------------------------------------------------------------------

struct Globals { double drift_pos_isigma, drift_ea_isigma, tilt_isigma, frine_isigma_scale; };
struct Scene {
    int n_poses, n_landmarks, n_frines, n_odo;
    Globals g;
    std::vector<double> pose_init, tilt, lm_pos;
    std::vector<int> f_pose, f_lm;
    std::vector<double> f_mf2r, f_camera_pos, f_isigma;
    std::vector<int> o_prev, o_cur;
    std::vector<double> o_dpos, o_dea, o_pcr, o_pci, o_ecr, o_eci;
};

static void rd(std::ifstream& f, double* d, int n) { for (int i = 0; i < n; i++) f >> d[i]; }

static Scene load(const char* path) {
    std::ifstream f(path);
    Scene s;
    f >> s.n_poses >> s.n_landmarks >> s.n_frines >> s.n_odo;
    f >> s.g.drift_pos_isigma >> s.g.drift_ea_isigma
      >> s.g.tilt_isigma >> s.g.frine_isigma_scale;
    s.pose_init.resize(6 * s.n_poses); s.tilt.resize(2 * s.n_poses);
    for (int i = 0; i < s.n_poses; i++) {
        rd(f, &s.pose_init[6 * i], 6);
        rd(f, &s.tilt[2 * i], 2);
    }
    s.lm_pos.resize(3 * s.n_landmarks);
    for (int i = 0; i < s.n_landmarks; i++) rd(f, &s.lm_pos[3 * i], 3);
    s.f_pose.resize(s.n_frines); s.f_lm.resize(s.n_frines);
    s.f_mf2r.resize(9 * s.n_frines); s.f_camera_pos.resize(3 * s.n_frines);
    s.f_isigma.resize(2 * s.n_frines);
    for (int i = 0; i < s.n_frines; i++) {
        f >> s.f_pose[i] >> s.f_lm[i];
        rd(f, &s.f_mf2r[9 * i], 9); rd(f, &s.f_camera_pos[3 * i], 3); rd(f, &s.f_isigma[2 * i], 2);
    }
    s.o_prev.resize(s.n_odo); s.o_cur.resize(s.n_odo);
    s.o_dpos.resize(3 * s.n_odo); s.o_dea.resize(3 * s.n_odo);
    s.o_pcr.resize(9 * s.n_odo); s.o_pci.resize(3 * s.n_odo);
    s.o_ecr.resize(9 * s.n_odo); s.o_eci.resize(3 * s.n_odo);
    for (int i = 0; i < s.n_odo; i++) {
        f >> s.o_prev[i] >> s.o_cur[i];
        rd(f, &s.o_dpos[3 * i], 3); rd(f, &s.o_dea[3 * i], 3);
        rd(f, &s.o_pcr[9 * i], 9); rd(f, &s.o_pci[3 * i], 3);
        rd(f, &s.o_ecr[9 * i], 9); rd(f, &s.o_eci[3 * i], 3);
    }
    return s;
}

static M3 m3(const double* d) {
    M3 m;
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) m(r, c) = d[3 * r + c];
    return m;
}
static V3 v3(const double* d) { return V3(d[0], d[1], d[2]); }

static double fd_edge(g2o::OptimizableGraph::Edge* e, const Eigen::MatrixXd& Jan, int which_v) {
    auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(e->vertex(which_v));
    int dim = v->dimension(), D = e->dimension();
    const double eps = 1e-6;
    Eigen::MatrixXd Jfd(D, dim);
    std::vector<double> delta(dim, 0.0);
    for (int j = 0; j < dim; j++) {
        delta.assign(dim, 0.0); delta[j] = eps;
        v->push(); v->oplus(delta.data()); e->computeError();
        Eigen::VectorXd ep(D); for (int i = 0; i < D; i++) ep[i] = e->errorData()[i];
        v->pop();
        delta[j] = -eps;
        v->push(); v->oplus(delta.data()); e->computeError();
        Eigen::VectorXd em(D); for (int i = 0; i < D; i++) em[i] = e->errorData()[i];
        v->pop();
        Jfd.col(j) = (ep - em) / (2 * eps);
    }
    return (Jan - Jfd).cwiseAbs().maxCoeff();
}

struct RunResult {
    double ms;
    int iterations; // outer iterations (each ends with an accepted step)
    // Total damped trials: sum of levenbergIterations over the outer
    // iterations (from the untimed statistics pass) -- the count
    // comparable to the other systems' attempts.
    int attempts;
    double initial_cost;
};

static RunResult solve(const Scene& s, bool lm, int max_iters, std::vector<double>* pose_out,
                       bool stats) {
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    auto linear = std::make_unique<g2o::LinearSolverCholmod<BlockSolver::PoseMatrixType>>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    g2o::OptimizationAlgorithm* algo;
    if (lm) {
        auto* lev = new g2o::OptimizationAlgorithmLevenberg(std::move(block));
        double lambda0 = 1e-9;
        if (const char* li = getenv("G2O_LAMBDA_INIT")) lambda0 = atof(li);
        lev->setUserLambdaInit(lambda0);
        algo = lev;
    } else {
        algo = new g2o::OptimizationAlgorithmGaussNewton(std::move(block));
    }
    g2o::SparseOptimizer opt;
    opt.setVerbose(false);
    // Batch statistics cost ~14% of the solve (measured at 300 poses), so
    // they are gathered in a separate UNTIMED solve, never in the timed one.
    opt.setComputeBatchStatistics(stats);
    opt.setAlgorithm(algo);

    for (int i = 0; i < s.n_poses; i++) {
        auto* v = new VertexPose();
        v->setId(i);
        V6 e; for (int k = 0; k < 6; k++) e[k] = s.pose_init[6 * i + k];
        v->setEstimate(e);
        opt.addVertex(v);
    }
    std::vector<g2o::OptimizableGraph::Edge*> checkable;  // one of each type
    auto note = [&](g2o::OptimizableGraph::Edge* e) {
        if (checkable.size() < 4) checkable.push_back(e);
    };
    for (int i = 0; i < s.n_poses; i++) {
        auto* dr = new EdgeDrift();
        dr->setVertex(0, opt.vertex(i));
        for (int k = 0; k < 6; k++) dr->prior[k] = s.pose_init[6 * i + k];
        dr->pi = s.g.drift_pos_isigma; dr->ei = s.g.drift_ea_isigma;
        dr->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        opt.addEdge(dr);
        auto* ti = new EdgeTilt();
        ti->setVertex(0, opt.vertex(i));
        ti->roll_m = s.tilt[2 * i]; ti->pitch_m = s.tilt[2 * i + 1]; ti->isigma = s.g.tilt_isigma;
        ti->setInformation(Eigen::Matrix2d::Identity());
        opt.addEdge(ti);
        if (i == 0) { note(dr); note(ti); }
    }
    for (int k = 0; k < s.n_frines; k++) {
        auto* be = new EdgeBearing();
        be->setVertex(0, opt.vertex(s.f_pose[k]));
        be->lm = v3(&s.lm_pos[3 * s.f_lm[k]]);
        be->mf2r = m3(&s.f_mf2r[9 * k]); be->camera_pos = v3(&s.f_camera_pos[3 * k]);
        be->isigma = V2(s.f_isigma[2 * k], s.f_isigma[2 * k + 1]); be->scale = s.g.frine_isigma_scale;
        be->setInformation(Eigen::Matrix2d::Identity());
        opt.addEdge(be);
        if (k == 0) note(be);
    }
    for (int o = 0; o < s.n_odo; o++) {
        auto* oe = new EdgeOdo();
        oe->setVertex(0, opt.vertex(s.o_prev[o]));
        oe->setVertex(1, opt.vertex(s.o_cur[o]));
        oe->dpos = v3(&s.o_dpos[3 * o]); oe->dea = v3(&s.o_dea[3 * o]);
        oe->pcr = m3(&s.o_pcr[9 * o]); oe->pci = v3(&s.o_pci[3 * o]);
        oe->ecr = m3(&s.o_ecr[9 * o]); oe->eci = v3(&s.o_eci[3 * o]);
        oe->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
        opt.addEdge(oe);
        if (o == 0) note(oe);
    }

    auto* terminate = new g2o::SparseOptimizerTerminateAction();
    double gain = 1e-5;
    if (const char* g = getenv("G2O_GAIN")) gain = atof(g);
    terminate->setGainThreshold(gain);
    terminate->setMaxIterations(max_iters);
    opt.addPostIterationAction(terminate);

    auto t0 = std::chrono::steady_clock::now();
    opt.initializeOptimization();
    opt.computeActiveErrors();
    double initial_cost = opt.chi2();

    if (getenv("G2O_VERIFY_JAC")) {
        const char* names[] = {"drift", "tilt", "bearing", "odo"};
        g2o::JacobianWorkspace jw;
        for (size_t i = 0; i < checkable.size(); i++) {
            auto* e = checkable[i];
            e->computeError();
            jw.updateSize(e);
            jw.allocate();
            e->linearizeOplus(jw);
            double err = 0;
            if (auto* u6 = dynamic_cast<EdgeDrift*>(e)) err = fd_edge(e, u6->jxi(), 0);
            else if (auto* u2 = dynamic_cast<EdgeTilt*>(e)) err = fd_edge(e, u2->jxi(), 0);
            else if (auto* eb = dynamic_cast<EdgeBearing*>(e)) err = fd_edge(e, eb->jxi(), 0);
            else if (auto* eo = dynamic_cast<EdgeOdo*>(e))
                err = std::max(fd_edge(e, eo->jxi(), 0), fd_edge(e, eo->jxj(), 1));
            fprintf(stderr, "jac check %-9s max|analytic-fd| = %.3e\n", names[i], err);
            if (err > 1e-4) { fprintf(stderr, "JACOBIAN MISMATCH\n"); exit(2); }
        }
    }

    int iters = opt.optimize(max_iters);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    int attempts = 0;
    if (stats) {
        for (const auto& st : opt.batchStatistics()) {
            if (st.iteration < 0) continue;
            attempts += st.levenbergIterations;
        }
    }
    if (attempts < iters) attempts = iters; // stats off / GN -- never undercount

    if (pose_out) {
        pose_out->resize(6 * s.n_poses);
        for (int i = 0; i < s.n_poses; i++) {
            const V6& e = static_cast<VertexPose*>(opt.vertex(i))->estimate();
            for (int k = 0; k < 6; k++) (*pose_out)[6 * i + k] = e[k];
        }
    }
    return RunResult{ms, iters, attempts, initial_cost};
}

static long peak_rss_kb() {
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("VmHWM:", 0) == 0) return atol(l.c_str() + 6);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <scene.txt> <lm|gn> <solution_out>\n", argv[0]); return 1; }
    bool lm = std::string(argv[2]) == "lm";
    Scene s = load(argv[1]);

    RunResult first = solve(s, lm, 1, nullptr, false);
    std::vector<double> poses;
    RunResult full = solve(s, lm, 200, &poses, false);
    // Untimed statistics pass: the damped-trial (attempts) count.
    RunResult st_run = solve(s, lm, 200, nullptr, true);
    full.attempts = st_run.attempts;

    std::ofstream out(argv[3]);
    out.precision(17);
    for (int i = 0; i < s.n_poses; i++)
        for (int k = 0; k < 6; k++) out << poses[6 * i + k] << (k == 5 ? "\n" : " ");

    std::string cpus = "?";
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("Cpus_allowed_list:", 0) == 0)
            cpus = l.substr(l.find_last_of(" \t") + 1);

    printf("{\"solve_ms\": %.3f, \"first_iter_ms\": %.3f, \"iterations\": %d, "
           "\"accepted\": %d, \"initial_cost\": %.6f, \"peak_rss_kb\": %ld, \"cpus_allowed\": \"%s\"}\n",
           full.ms, first.ms, full.attempts, full.iterations, full.initial_cost, peak_rss_kb(), cpus.c_str());
    return 0;
}

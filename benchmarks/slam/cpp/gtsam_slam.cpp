// GTSAM runner for the heterogeneous SLAM benchmark. Reads the scene
// exported by the Rust harness (identical problem; same file Ceres,
// SymForce and g2o read) and builds all six factor types as CUSTOM
// NoiseModelFactorN subclasses with ANALYTIC Jacobians -- no numeric
// differentiation, so GTSAM runs at full performance, on parity with the
// autodiff/codegen systems. Pose variables are Vector6 [x,y,z,roll,pitch,
// yaw] and landmarks Vector3, both vector-space (additive) retraction,
// matching arael's SimpleEulerAngleParam and the other systems. The euler
// convention and odometry euler extraction reproduce arael's matrix3
// twins, so every residual equals scene::reference_cost; with unit noise
// GTSAM's error is 0.5 * sum||r||^2, so the harness cross-checks
// initial_cost = 2 * graph.error(initial). Bearings and GPS are plain
// Gaussian (outlier-free problem; no robust kernel). The analytic
// Jacobians are checked against gtsam::numericalDerivative when
// GTSAM_VERIFY_JAC=1 (verification only, never in the timed solve).
//
//   gtsam_slam <scene.txt> <lm> <solution_out>
// JSON {solve_ms, first_iter_ms, iterations, accepted, initial_cost,
// peak_rss_kb, cpus_allowed} on stdout; solution_out carries one pose per
// line (6 values) then one landmark per line (3 values).
// GTSAM_LAMBDA0 overrides the LM initial damping (default 1e-9).

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using gtsam::Key;
using gtsam::Matrix;
using gtsam::NoiseModelFactorN;
using gtsam::Vector;
using V6 = gtsam::Vector6;
using V3 = gtsam::Vector3;
using V2 = gtsam::Vector2;
using M3 = Eigen::Matrix3d;

static Key P(int i) { return gtsam::Symbol('x', i); }
static Key L(int j) { return gtsam::Symbol('l', j); }

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
// R and its derivatives wrt (roll, pitch, yaw): dR/dtheta = skew(axis) R.
static void euler_rot_d(double roll, double pitch, double yaw, M3& R, M3 dR[3]) {
    R = euler_rot(roll, pitch, yaw);
    double sy = sin(pitch), cy = cos(pitch), sz = sin(yaw), cz = cos(yaw);
    dR[0] = skew(V3(cz * cy, sz * cy, -sy)) * R;  // d/droll
    dR[1] = skew(V3(-sz, cz, 0)) * R;             // d/dpitch
    dR[2] = skew(V3(0, 0, 1)) * R;                // d/dyaw
}
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

// -- the six factors (residual == scene::reference_cost, whitening baked
//    in so the noise model is unit; analytic Jacobians) --

struct GpsFactor : public NoiseModelFactorN<V6> {
    V3 gps_pos, isigma; M3 cov_r;
    GpsFactor(Key k, const V3& gp, const M3& cr, const V3& is)
        : NoiseModelFactorN<V6>(gtsam::noiseModel::Unit::Create(3), k),
          gps_pos(gp), isigma(is), cov_r(cr) {}
    Vector evaluateError(const V6& p, boost::optional<Matrix&> H = boost::none) const override {
        V3 rt = cov_r.transpose() * (p.head<3>() - gps_pos);
        if (H) { *H = Matrix::Zero(3, 6); H->block<3, 3>(0, 0) = isigma.asDiagonal() * cov_r.transpose(); }
        return rt.cwiseProduct(isigma);
    }
};

struct DriftFactor : public NoiseModelFactorN<V6> {
    V6 prior; double pi, ei;
    DriftFactor(Key k, const V6& pr, double p_, double e_)
        : NoiseModelFactorN<V6>(gtsam::noiseModel::Unit::Create(6), k), prior(pr), pi(p_), ei(e_) {}
    Vector evaluateError(const V6& p, boost::optional<Matrix&> H = boost::none) const override {
        V6 e;
        e.head<3>() = (p.head<3>() - prior.head<3>()) * pi;
        e.tail<3>() = (p.tail<3>() - prior.tail<3>()) * ei;
        if (H) { *H = Matrix::Zero(6, 6); (*H).diagonal() << pi, pi, pi, ei, ei, ei; }
        return e;
    }
};

struct TiltFactor : public NoiseModelFactorN<V6> {
    double roll_m, pitch_m, isigma;
    TiltFactor(Key k, double r, double p, double is)
        : NoiseModelFactorN<V6>(gtsam::noiseModel::Unit::Create(2), k), roll_m(r), pitch_m(p), isigma(is) {}
    Vector evaluateError(const V6& p, boost::optional<Matrix&> H = boost::none) const override {
        if (H) { *H = Matrix::Zero(2, 6); (*H)(0, 3) = isigma; (*H)(1, 4) = isigma; }
        return V2((p[3] - roll_m) * isigma, (p[4] - pitch_m) * isigma);
    }
};

struct LmDriftFactor : public NoiseModelFactorN<V3> {
    V3 prior; double isigma;
    LmDriftFactor(Key k, const V3& pr, double is)
        : NoiseModelFactorN<V3>(gtsam::noiseModel::Unit::Create(3), k), prior(pr), isigma(is) {}
    Vector evaluateError(const V3& lm, boost::optional<Matrix&> H = boost::none) const override {
        if (H) *H = isigma * M3::Identity();
        return (lm - prior) * isigma;
    }
};

// (0) landmark, (1) pose -- matching ceres BearingCost(lm, pose)
struct BearingFactor : public NoiseModelFactorN<V3, V6> {
    M3 mf2r; V3 camera_pos; V2 isigma; double scale;
    BearingFactor(Key kl, Key kp, const M3& m, const V3& c, const V2& is, double s)
        : NoiseModelFactorN<V3, V6>(gtsam::noiseModel::Unit::Create(2), kl, kp),
          mf2r(m), camera_pos(c), isigma(is), scale(s) {}
    Vector evaluateError(const V3& lm, const V6& pose,
                         boost::optional<Matrix&> H1 = boost::none,
                         boost::optional<Matrix&> H2 = boost::none) const override {
        M3 R, dR[3];
        euler_rot_d(pose[3], pose[4], pose[5], R, dR);
        V3 d = lm - pose.head<3>();
        V3 r_f = mf2r.transpose() * (R.transpose() * d - camera_pos);
        double s0 = isigma[0] * scale, s1 = isigma[1] * scale;
        V2 e(atan2(r_f[1], r_f[0]) * s0, atan2(r_f[2], r_f[0]) * s1);
        if (H1 || H2) {
            double d0 = r_f[0] * r_f[0] + r_f[1] * r_f[1];
            double d1 = r_f[0] * r_f[0] + r_f[2] * r_f[2];
            Eigen::Matrix<double, 2, 3> dr_drf;
            dr_drf << s0 * (-r_f[1] / d0), s0 * (r_f[0] / d0), 0,
                      s1 * (-r_f[2] / d1), 0, s1 * (r_f[0] / d1);
            Eigen::Matrix<double, 2, 3> dr_drr = dr_drf * mf2r.transpose();
            if (H1) *H1 = dr_drr * R.transpose();
            if (H2) {
                *H2 = Matrix::Zero(2, 6);
                H2->block<2, 3>(0, 0) = -dr_drr * R.transpose();
                for (int k = 0; k < 3; k++)
                    H2->block<2, 1>(0, 3 + k) = dr_drr * (dR[k].transpose() * d);
            }
        }
        return e;
    }
};

// (0) prev pose, (1) cur pose
struct OdoFactor : public NoiseModelFactorN<V6, V6> {
    V3 dpos, dea, pci, eci; M3 pcr, ecr;
    OdoFactor(Key kp, Key kc, const V3& dp, const V3& de, const M3& pc, const V3& pi_,
              const M3& ec, const V3& ei_)
        : NoiseModelFactorN<V6, V6>(gtsam::noiseModel::Unit::Create(6), kp, kc),
          dpos(dp), dea(de), pci(pi_), eci(ei_), pcr(pc), ecr(ec) {}
    Vector evaluateError(const V6& prev, const V6& cur,
                         boost::optional<Matrix&> H1 = boost::none,
                         boost::optional<Matrix&> H2 = boost::none) const override {
        M3 Rp, dRp[3], Rc, dRc[3];
        euler_rot_d(prev[3], prev[4], prev[5], Rp, dRp);
        euler_rot_d(cur[3], cur[4], cur[5], Rc, dRc);
        M3 A = euler_rot(dea[0], dea[1], dea[2]).transpose();
        V3 d = cur.head<3>() - prev.head<3>();
        V3 pos_w = pcr.transpose() * (Rp.transpose() * d - dpos);
        M3 M = A * Rp.transpose() * Rc;
        V3 ea_w = ecr.transpose() * rot_to_euler(M);
        V6 e;
        e.head<3>() = pos_w.cwiseProduct(pci);
        e.tail<3>() = ea_w.cwiseProduct(eci);
        if (H1 || H2) {
            M3 W = pci.asDiagonal() * pcr.transpose();
            M3 E = eci.asDiagonal() * ecr.transpose();
            if (H1) {
                *H1 = Matrix::Zero(6, 6);
                H1->block<3, 3>(0, 0) = W * (-Rp.transpose());
                for (int k = 0; k < 3; k++) {
                    H1->block<3, 1>(0, 3 + k) = W * (dRp[k].transpose() * d);
                    H1->block<3, 1>(3, 3 + k) = E * rot_to_euler_d(M, A * dRp[k].transpose() * Rc);
                }
            }
            if (H2) {
                *H2 = Matrix::Zero(6, 6);
                H2->block<3, 3>(0, 0) = W * Rp.transpose();
                for (int k = 0; k < 3; k++)
                    H2->block<3, 1>(3, 3 + k) = E * rot_to_euler_d(M, A * Rp.transpose() * dRc[k]);
            }
        }
        return e;
    }
};

// ---------------------------------------------------------------------------
// Scene (identical format to the Ceres/SymForce/g2o runners)
// ---------------------------------------------------------------------------

struct Globals { double drift_pos_isigma, drift_ea_isigma, drift_lm_isigma, tilt_isigma, frine_isigma_scale; };
struct Scene {
    int n_poses, n_landmarks, n_frines, n_odo;
    Globals g;
    std::vector<double> pose_init, gps_pos, gps_cov_r, gps_cov_isigma, tilt, lm_init;
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
    f >> s.g.drift_pos_isigma >> s.g.drift_ea_isigma >> s.g.drift_lm_isigma
      >> s.g.tilt_isigma >> s.g.frine_isigma_scale;
    s.pose_init.resize(6 * s.n_poses);
    s.gps_pos.resize(3 * s.n_poses); s.gps_cov_r.resize(9 * s.n_poses);
    s.gps_cov_isigma.resize(3 * s.n_poses); s.tilt.resize(2 * s.n_poses);
    for (int i = 0; i < s.n_poses; i++) {
        rd(f, &s.pose_init[6 * i], 6); rd(f, &s.gps_pos[3 * i], 3);
        rd(f, &s.gps_cov_r[9 * i], 9); rd(f, &s.gps_cov_isigma[3 * i], 3);
        rd(f, &s.tilt[2 * i], 2);
    }
    s.lm_init.resize(3 * s.n_landmarks);
    for (int i = 0; i < s.n_landmarks; i++) rd(f, &s.lm_init[3 * i], 3);
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

static void build(const Scene& s, gtsam::NonlinearFactorGraph& graph, gtsam::Values& init) {
    for (int i = 0; i < s.n_poses; i++) {
        V6 p; for (int k = 0; k < 6; k++) p[k] = s.pose_init[6 * i + k];
        init.insert(P(i), p);
        V6 prior = p;
        graph.emplace_shared<GpsFactor>(P(i), v3(&s.gps_pos[3 * i]), m3(&s.gps_cov_r[9 * i]),
                                        v3(&s.gps_cov_isigma[3 * i]));
        graph.emplace_shared<DriftFactor>(P(i), prior, s.g.drift_pos_isigma, s.g.drift_ea_isigma);
        graph.emplace_shared<TiltFactor>(P(i), s.tilt[2 * i], s.tilt[2 * i + 1], s.g.tilt_isigma);
    }
    for (int j = 0; j < s.n_landmarks; j++) {
        init.insert(L(j), v3(&s.lm_init[3 * j]));
        graph.emplace_shared<LmDriftFactor>(L(j), v3(&s.lm_init[3 * j]), s.g.drift_lm_isigma);
    }
    for (int k = 0; k < s.n_frines; k++)
        graph.emplace_shared<BearingFactor>(L(s.f_lm[k]), P(s.f_pose[k]), m3(&s.f_mf2r[9 * k]),
            v3(&s.f_camera_pos[3 * k]), V2(s.f_isigma[2 * k], s.f_isigma[2 * k + 1]),
            s.g.frine_isigma_scale);
    for (int o = 0; o < s.n_odo; o++)
        graph.emplace_shared<OdoFactor>(P(s.o_prev[o]), P(s.o_cur[o]), v3(&s.o_dpos[3 * o]),
            v3(&s.o_dea[3 * o]), m3(&s.o_pcr[9 * o]), v3(&s.o_pci[3 * o]),
            m3(&s.o_ecr[9 * o]), v3(&s.o_eci[3 * o]));
}

// Check every factor's analytic Jacobian against gtsam::numericalDerivative.
static void verify_jacobians(const Scene& s, const gtsam::Values& v) {
    using gtsam::numericalDerivative11;
    using gtsam::numericalDerivative21;
    using gtsam::numericalDerivative22;
    double worst = 0;
    // one representative factor of each type
    {
        GpsFactor f(P(0), v3(&s.gps_pos[0]), m3(&s.gps_cov_r[0]), v3(&s.gps_cov_isigma[0]));
        V6 p = v.at<V6>(P(0)); Matrix H;
        f.evaluateError(p, H);
        Matrix Hn = numericalDerivative11<Vector, V6>(
            [&](const V6& x) { return Vector(f.evaluateError(x)); }, p);
        worst = std::max(worst, (H - Hn).cwiseAbs().maxCoeff());
    }
    {
        V6 prior = v.at<V6>(P(0));
        OdoFactor f(P(s.o_prev[0]), P(s.o_cur[0]), v3(&s.o_dpos[0]), v3(&s.o_dea[0]),
                    m3(&s.o_pcr[0]), v3(&s.o_pci[0]), m3(&s.o_ecr[0]), v3(&s.o_eci[0]));
        V6 a = v.at<V6>(P(s.o_prev[0])), b = v.at<V6>(P(s.o_cur[0]));
        Matrix H1, H2; f.evaluateError(a, b, H1, H2);
        Matrix H1n = numericalDerivative21<Vector, V6, V6>(
            [&](const V6& x, const V6& y) { return Vector(f.evaluateError(x, y)); }, a, b);
        Matrix H2n = numericalDerivative22<Vector, V6, V6>(
            [&](const V6& x, const V6& y) { return Vector(f.evaluateError(x, y)); }, a, b);
        worst = std::max(worst, std::max((H1 - H1n).cwiseAbs().maxCoeff(), (H2 - H2n).cwiseAbs().maxCoeff()));
        (void)prior;
    }
    {
        BearingFactor f(L(s.f_lm[0]), P(s.f_pose[0]), m3(&s.f_mf2r[0]), v3(&s.f_camera_pos[0]),
                        V2(s.f_isigma[0], s.f_isigma[1]), s.g.frine_isigma_scale);
        V3 lm = v.at<V3>(L(s.f_lm[0])); V6 p = v.at<V6>(P(s.f_pose[0]));
        Matrix H1, H2; f.evaluateError(lm, p, H1, H2);
        Matrix H1n = numericalDerivative21<Vector, V3, V6>(
            [&](const V3& x, const V6& y) { return Vector(f.evaluateError(x, y)); }, lm, p);
        Matrix H2n = numericalDerivative22<Vector, V3, V6>(
            [&](const V3& x, const V6& y) { return Vector(f.evaluateError(x, y)); }, lm, p);
        worst = std::max(worst, std::max((H1 - H1n).cwiseAbs().maxCoeff(), (H2 - H2n).cwiseAbs().maxCoeff()));
    }
    fprintf(stderr, "gtsam jac check (gps/odo/bearing) max|analytic-numeric| = %.3e\n", worst);
    if (worst > 1e-4) { fprintf(stderr, "JACOBIAN MISMATCH\n"); exit(2); }
}

struct RunResult { double ms; int iterations; double initial_cost; };

static RunResult solve(const Scene& s, int max_iters, std::vector<double>* pose_out,
                       std::vector<double>* lm_out) {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values init;
    build(s, graph, init);

    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SILENT");
    params.setMaxIterations(max_iters);
    params.setRelativeErrorTol(1e-5);
    params.setAbsoluteErrorTol(1e-5);
    params.lambdaInitial = 1e-9;
    if (const char* l = getenv("GTSAM_LAMBDA0")) params.lambdaInitial = atof(l);

    double initial_cost = 2.0 * graph.error(init);  // GTSAM error is 0.5*sum r^2

    auto t0 = std::chrono::steady_clock::now();
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, init, params);
    gtsam::Values result = optimizer.optimize();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (pose_out) {
        pose_out->resize(6 * s.n_poses);
        for (int i = 0; i < s.n_poses; i++) {
            V6 p = result.at<V6>(P(i));
            for (int k = 0; k < 6; k++) (*pose_out)[6 * i + k] = p[k];
        }
    }
    if (lm_out) {
        lm_out->resize(3 * s.n_landmarks);
        for (int j = 0; j < s.n_landmarks; j++) {
            V3 p = result.at<V3>(L(j));
            for (int k = 0; k < 3; k++) (*lm_out)[3 * j + k] = p[k];
        }
    }
    return RunResult{ms, (int)optimizer.iterations(), initial_cost};
}

static long peak_rss_kb() {
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("VmHWM:", 0) == 0) return atol(l.c_str() + 6);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <scene.txt> <lm> <solution_out>\n", argv[0]); return 1; }
    Scene s = load(argv[1]);

    if (getenv("GTSAM_VERIFY_JAC")) {
        gtsam::NonlinearFactorGraph g; gtsam::Values v; build(s, g, v);
        verify_jacobians(s, v);
    }

    RunResult first = solve(s, 1, nullptr, nullptr);
    std::vector<double> poses, lms;
    RunResult full = solve(s, 200, &poses, &lms);

    std::ofstream out(argv[3]);
    out.precision(17);
    for (int i = 0; i < s.n_poses; i++)
        for (int k = 0; k < 6; k++) out << poses[6 * i + k] << (k == 5 ? "\n" : " ");
    for (int j = 0; j < s.n_landmarks; j++)
        for (int k = 0; k < 3; k++) out << lms[3 * j + k] << (k == 2 ? "\n" : " ");

    std::string cpus = "?";
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("Cpus_allowed_list:", 0) == 0)
            cpus = l.substr(l.find_last_of(" \t") + 1);

    printf("{\"solve_ms\": %.3f, \"first_iter_ms\": %.3f, \"iterations\": %d, "
           "\"accepted\": %d, \"initial_cost\": %.6f, \"peak_rss_kb\": %ld, \"cpus_allowed\": \"%s\"}\n",
           full.ms, first.ms, full.iterations, full.iterations, full.initial_cost, peak_rss_kb(), cpus.c_str());
    return 0;
}

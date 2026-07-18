// Ceres runner for the heterogeneous SLAM benchmark. Reads the scene
// exported by the Rust harness (identical problem), builds all six
// factor types as AutoDiffCostFunctions, and solves. Pose blocks are
// plain 6-vectors [x, y, z, roll, pitch, yaw] with the rotation built
// from the euler angles inside each residual -- matching arael's
// SimpleEulerAngleParam (no manifold) exactly. The euler convention and
// the odometry euler extraction reproduce arael's matrix3 twins, so
// every residual equals scene::reference_cost (asserted by the harness
// via the initial cost). Bearings and GPS are plain Gaussian (outlier-free
// problem; no robust kernel), matching arael.
//
//   ceres_slam <scene.txt> <solution_out> [dense_schur|sparse_schur|
//              sparse_normal_cholesky|iterative_schur]
// JSON {solve_ms, first_iter_ms, iterations, accepted, initial_cost,
// peak_rss_kb, cpus_allowed} on stdout; solution_out carries one pose
// per line (6 values) then one landmark per line (3 values).

#include "../../cpp/bench.h"
#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// -- arael's rotation convention (x=roll, y=pitch, z=yaw; Rz Ry Rx) --
template <typename T>
Eigen::Matrix<T, 3, 3> euler_to_rot(const T* ea) {
    T sx = sin(ea[0]), cx = cos(ea[0]);
    T sy = sin(ea[1]), cy = cos(ea[1]);
    T sz = sin(ea[2]), cz = cos(ea[2]);
    Eigen::Matrix<T, 3, 3> m;
    m << cy * cz,            -cx * sz + cz * sx * sy,  cx * cz * sy + sx * sz,
         cy * sz,             cx * cz + sx * sy * sz,  cx * sy * sz - cz * sx,
        -sy,                  cy * sx,                 cx * cy;
    return m;
}

// arael's get_euler_angles main branch (error rotation is near identity).
template <typename T>
void rot_to_euler(const Eigen::Matrix<T, 3, 3>& m, T* ea) {
    ea[0] = atan2(m(2, 1), m(2, 2));
    ea[1] = -asin(m(2, 0));
    ea[2] = atan2(m(1, 0), m(0, 0));
}

struct Globals {
    double drift_pos_isigma, drift_ea_isigma, drift_lm_isigma,
           tilt_isigma, frine_isigma_scale;
};

struct Scene {
    int n_poses, n_landmarks, n_frines, n_odo;
    Globals g;
    // pose i: init(6), gps_pos(3), gps_cov_r(9), gps_cov_isigma(3), tilt(2)
    std::vector<double> pose_init;    // 6 per pose
    std::vector<double> gps_pos, gps_cov_r, gps_cov_isigma, tilt;
    std::vector<double> lm_init;      // 3 per landmark
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
      >> s.g.drift_lm_isigma >> s.g.tilt_isigma >> s.g.frine_isigma_scale;
    s.pose_init.resize(6 * s.n_poses);
    s.gps_pos.resize(3 * s.n_poses); s.gps_cov_r.resize(9 * s.n_poses);
    s.gps_cov_isigma.resize(3 * s.n_poses); s.tilt.resize(2 * s.n_poses);
    for (int i = 0; i < s.n_poses; i++) {
        rd(f, &s.pose_init[6 * i], 6);
        rd(f, &s.gps_pos[3 * i], 3);
        rd(f, &s.gps_cov_r[9 * i], 9);
        rd(f, &s.gps_cov_isigma[3 * i], 3);
        rd(f, &s.tilt[2 * i], 2);
    }
    s.lm_init.resize(3 * s.n_landmarks);
    for (int i = 0; i < s.n_landmarks; i++) rd(f, &s.lm_init[3 * i], 3);
    s.f_pose.resize(s.n_frines); s.f_lm.resize(s.n_frines);
    s.f_mf2r.resize(9 * s.n_frines); s.f_camera_pos.resize(3 * s.n_frines);
    s.f_isigma.resize(2 * s.n_frines);
    for (int i = 0; i < s.n_frines; i++) {
        f >> s.f_pose[i] >> s.f_lm[i];
        rd(f, &s.f_mf2r[9 * i], 9);
        rd(f, &s.f_camera_pos[3 * i], 3);
        rd(f, &s.f_isigma[2 * i], 2);
    }
    s.o_prev.resize(s.n_odo); s.o_cur.resize(s.n_odo);
    s.o_dpos.resize(3 * s.n_odo); s.o_dea.resize(3 * s.n_odo);
    s.o_pcr.resize(9 * s.n_odo); s.o_pci.resize(3 * s.n_odo);
    s.o_ecr.resize(9 * s.n_odo); s.o_eci.resize(3 * s.n_odo);
    for (int i = 0; i < s.n_odo; i++) {
        f >> s.o_prev[i] >> s.o_cur[i];
        rd(f, &s.o_dpos[3 * i], 3);
        rd(f, &s.o_dea[3 * i], 3);
        rd(f, &s.o_pcr[9 * i], 9);
        rd(f, &s.o_pci[3 * i], 3);
        rd(f, &s.o_ecr[9 * i], 9);
        rd(f, &s.o_eci[3 * i], 3);
    }
    return s;
}

// -- functors --

struct GpsCost {
    const double *pos, *cov_r, *isigma;
    template <typename T> bool operator()(const T* p, T* r) const {
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(cov_r);
        Eigen::Matrix<T, 3, 1> raw(p[0] - pos[0], p[1] - pos[1], p[2] - pos[2]);
        Eigen::Matrix<T, 3, 1> rt = R.transpose().cast<T>() * raw;
        for (int i = 0; i < 3; i++) r[i] = rt[i] * isigma[i];
        return true;
    }
};
struct DriftCost {
    const double *prior; double pi, ei;
    template <typename T> bool operator()(const T* p, T* r) const {
        for (int i = 0; i < 3; i++) r[i] = (p[i] - prior[i]) * pi;
        for (int i = 3; i < 6; i++) r[i] = (p[i] - prior[i]) * ei;
        return true;
    }
};
struct TiltCost {
    double roll, pitch, isigma;
    template <typename T> bool operator()(const T* p, T* r) const {
        r[0] = (p[3] - roll) * isigma;
        r[1] = (p[4] - pitch) * isigma;
        return true;
    }
};
struct LmDriftCost {
    const double* prior; double isigma;
    template <typename T> bool operator()(const T* p, T* r) const {
        for (int i = 0; i < 3; i++) r[i] = (p[i] - prior[i]) * isigma;
        return true;
    }
};
struct BearingCost {
    const double *mf2r, *camera_pos, *isigma; double scale;
    template <typename T> bool operator()(const T* lm, const T* pose, T* r) const {
        Eigen::Matrix<T, 3, 3> mr2w = euler_to_rot(pose + 3);
        Eigen::Matrix<T, 3, 1> d(lm[0] - pose[0], lm[1] - pose[1], lm[2] - pose[2]);
        Eigen::Matrix<T, 3, 1> lm_r = mr2w.transpose() * d;
        Eigen::Matrix<T, 3, 1> r_r(lm_r[0] - camera_pos[0], lm_r[1] - camera_pos[1], lm_r[2] - camera_pos[2]);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> M(mf2r);
        Eigen::Matrix<T, 3, 1> r_f = M.transpose().cast<T>() * r_r;
        r[0] = atan2(r_f[1], r_f[0]) * T(isigma[0] * scale);
        r[1] = atan2(r_f[2], r_f[0]) * T(isigma[1] * scale);
        return true;
    }
};
struct OdoCost {
    const double *dpos, *dea, *pcr, *pci, *ecr, *eci;
    template <typename T> bool operator()(const T* prev, const T* cur, T* r) const {
        Eigen::Matrix<T, 3, 3> mr2w_prev = euler_to_rot(prev + 3);
        Eigen::Matrix<T, 3, 3> mr2w_cur = euler_to_rot(cur + 3);
        Eigen::Matrix<T, 3, 1> d(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]);
        Eigen::Matrix<T, 3, 1> pos_diff = mr2w_prev.transpose() * d;
        Eigen::Matrix<T, 3, 1> pos_err(pos_diff[0] - dpos[0], pos_diff[1] - dpos[1], pos_diff[2] - dpos[2]);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> PCR(pcr), ECR(ecr);
        Eigen::Matrix<T, 3, 1> pos_w = PCR.transpose().cast<T>() * pos_err;
        T dea_t[3] = {T(dea[0]), T(dea[1]), T(dea[2])};
        Eigen::Matrix<T, 3, 3> expected = mr2w_prev * euler_to_rot(dea_t);
        Eigen::Matrix<T, 3, 3> error_rot = expected.transpose() * mr2w_cur;
        T ea_err[3]; rot_to_euler(error_rot, ea_err);
        Eigen::Matrix<T, 3, 1> ea_w = ECR.transpose().cast<T>() * Eigen::Matrix<T, 3, 1>(ea_err[0], ea_err[1], ea_err[2]);
        for (int i = 0; i < 3; i++) r[i] = pos_w[i] * pci[i];
        for (int i = 0; i < 3; i++) r[i + 3] = ea_w[i] * eci[i];
        return true;
    }
};

// Build the six factor types into `problem` over `poses`/`lms` (initialized
// from the scene). Blocks point into those vectors, which must outlive `problem`.
static void build_problem(Scene& s, std::vector<double>& poses, std::vector<double>& lms,
                          ceres::Problem& problem) {
    poses = s.pose_init;
    lms = s.lm_init;
    for (int i = 0; i < s.n_poses; i++) {
        double* p = &poses[6 * i];
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<GpsCost, 3, 6>(
            new GpsCost{&s.gps_pos[3 * i], &s.gps_cov_r[9 * i], &s.gps_cov_isigma[3 * i]}), nullptr, p);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<DriftCost, 6, 6>(
            new DriftCost{&s.pose_init[6 * i], s.g.drift_pos_isigma, s.g.drift_ea_isigma}), nullptr, p);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TiltCost, 2, 6>(
            new TiltCost{s.tilt[2 * i], s.tilt[2 * i + 1], s.g.tilt_isigma}), nullptr, p);
    }
    for (int i = 0; i < s.n_landmarks; i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LmDriftCost, 3, 3>(
            new LmDriftCost{&s.lm_init[3 * i], s.g.drift_lm_isigma}), nullptr, &lms[3 * i]);
    }
    for (int i = 0; i < s.n_frines; i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BearingCost, 2, 3, 6>(
            new BearingCost{&s.f_mf2r[9 * i], &s.f_camera_pos[3 * i], &s.f_isigma[2 * i],
                            s.g.frine_isigma_scale}),
            nullptr, &lms[3 * s.f_lm[i]], &poses[6 * s.f_pose[i]]);
    }
    for (int i = 0; i < s.n_odo; i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<OdoCost, 6, 6, 6>(
            new OdoCost{&s.o_dpos[3 * i], &s.o_dea[3 * i], &s.o_pcr[9 * i], &s.o_pci[3 * i],
                        &s.o_ecr[9 * i], &s.o_eci[3 * i]}),
            nullptr, &poses[6 * s.o_prev[i]], &poses[6 * s.o_cur[i]]);
    }
}

static bench::Result solve(Scene& s, ceres::LinearSolverType linsolver, int max_iters,
                       std::vector<double>* pose_out, std::vector<double>* lm_out) {
    std::vector<double> poses, lms;
    ceres::Problem problem;
    build_problem(s, poses, lms, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = linsolver;
    options.max_num_iterations = max_iters;
    options.num_threads = 1;
    options.function_tolerance = 1e-5;  // shared termination class
    // Problem-appropriate initial trust region (large -> near-Gauss-Newton
    // on this well-initialized graph), matching the pgo benchmark policy.
    options.initial_trust_region_radius = 1e12;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);
    // CERES_VERBOSE=1: per-iteration progress (cost, |gradient|, |step|) and
    // the full report, which names the termination criterion and its tolerance.
    if (getenv("CERES_VERBOSE")) options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (getenv("CERES_VERBOSE")) fprintf(stderr, "%s\n", summary.FullReport().c_str());
    if (pose_out) *pose_out = poses;
    if (lm_out) *lm_out = lms;
    // Count LINEAR SOLVES (factorizations), the unit the parenthesised total
    // reports. num_successful_steps is NOT that count: it includes iteration 0
    // (the initial cost eval, which does no solve) and omits the final
    // convergence-detecting solve. num_linear_solves is the true total; the
    // ones that did not reduce cost are num_unsuccessful_steps.
    const int attempts = summary.num_linear_solves;
    const int accepted = attempts - summary.num_unsuccessful_steps;
    return bench::Result{summary.total_time_in_seconds * 1e3,
                     accepted,
                     attempts,
                     2.0 * summary.initial_cost};
}

static long peak_rss_kb() {
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("VmHWM:", 0) == 0)
            return atol(l.c_str() + 6);
    return 0;
}

// Covariance timing: for each entity (pose 6-DOF, landmark 3-DOF) and each N, the
// cold cost of Covariance::Compute over N spread blocks (a fresh Covariance each
// rep -- Ceres has no separable factor-then-extract). Machine-readable COV lines
// for the Rust harness; a validation std-dev line goes to stderr. SLAM's GPS
// priors anchor the gauge, so no prior is needed.
static void cov_mode(Scene& s) {
    std::vector<double> poses, lms;
    ceres::Problem problem;
    build_problem(s, poses, lms, problem);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 200;
    options.num_threads = 1;
    options.function_tolerance = 1e-5;
    options.initial_trust_region_radius = 1e12;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ceres::Covariance::Options opts;
    opts.algorithm_type = ceres::SPARSE_QR;
    opts.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    opts.num_threads = 1;
    {
        const double* p = &poses[0];
        std::vector<std::pair<const double*, const double*>> b{{p, p}};
        ceres::Covariance probe(opts);
        if (!probe.Compute(b, &problem)) opts.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    }

    double budget = getenv("COV_BUDGET_S") ? atof(getenv("COV_BUDGET_S")) : 5.0;
    int cap = getenv("COV_CAP") ? atoi(getenv("COV_CAP")) : 2000;
    double cell_cap = bench::cov_cell_cap_ms();
    const int ns[] = {1, 2, 8, 32};
    for (int e = 0; e < 2; e++) {
        const char* name = (e == 0) ? "pose" : "landmark";
        int count = (e == 0) ? s.n_poses : s.n_landmarks;
        int stride = (e == 0) ? 6 : 3;
        double* blk = (e == 0) ? poses.data() : lms.data();
        std::vector<int> queryN(ns, ns + 4);
        queryN.push_back(count);  // "all"
        int prev_n = 0;
        double prev_ms = 0;
        for (int N : queryN) {
            if (N > count) continue;
            if (bench::cov_project_ms(prev_n, prev_ms, N) > cell_cap) {
                printf("COV %s %d toolong 0\n", name, N);
                fflush(stdout);
                continue;
            }
            std::vector<int> idx = bench::spread(0, count, N);
            std::vector<std::pair<const double*, const double*>> blocks;
            for (int i : idx) blocks.emplace_back(&blk[stride * i], &blk[stride * i]);
            bool ok = true;
            int reps = 0;
            double ms = bench::median_ms(budget, cap, &reps, [&] {
                ceres::Covariance cov(opts);
                if (!cov.Compute(blocks, &problem)) ok = false;
            });
            if (!ok)                printf("COV %s %d nan 0\n", name, N);
            else if (ms > cell_cap) printf("COV %s %d toolong %d\n", name, N, reps);
            else                    printf("COV %s %d %.3f %d\n", name, N, ms, reps);
            fflush(stdout);
            prev_n = N;
            prev_ms = ms;
        }
    }

    // Validation: middle-pose std dev.
    int mid = s.n_poses / 2;
    ceres::Covariance covm(opts);
    std::pair<const double*, const double*> bm{&poses[6 * mid], &poses[6 * mid]};
    if (covm.Compute({bm}, &problem)) {
        double c[36];
        covm.GetCovarianceBlock(&poses[6 * mid], &poses[6 * mid], c);
        fprintf(stderr, "ceres pose[%d] std dev:", mid);
        for (int d = 0; d < 6; d++) fprintf(stderr, " %.4f", sqrt(c[d * 6 + d]));
        fprintf(stderr, "\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <scene.txt> <solution_out|cov> [linsolver]\n", argv[0]); return 1; }
    Scene s = load(argv[1]);
    if (std::string(argv[2]) == "cov") {
        cov_mode(s);
        return 0;
    }
    ceres::LinearSolverType linsolver = ceres::SPARSE_NORMAL_CHOLESKY;
    if (argc > 3) {
        std::string t = argv[3];
        if (t == "dense_schur") linsolver = ceres::DENSE_SCHUR;
        else if (t == "sparse_schur") linsolver = ceres::SPARSE_SCHUR;
        else if (t == "iterative_schur") linsolver = ceres::ITERATIVE_SCHUR;
        else if (t != "sparse_normal_cholesky") { fprintf(stderr, "unknown linsolver %s\n", t.c_str()); return 1; }
    }

    std::vector<double> poses, lms;
    bench::report(
        [&](int n) { return solve(s, linsolver, n, nullptr, nullptr); },
        [&]() { return solve(s, linsolver, 200, &poses, &lms); });

    std::ofstream out(argv[2]);
    out.precision(17);
    for (int i = 0; i < s.n_poses; i++) {
        for (int k = 0; k < 6; k++) out << poses[6 * i + k] << (k == 5 ? "\n" : " ");
    }
    for (int i = 0; i < s.n_landmarks; i++) {
        for (int k = 0; k < 3; k++) out << lms[3 * i + k] << (k == 2 ? "\n" : " ");
    }

    std::string cpus = "?";
    std::ifstream st("/proc/self/status");
    std::string l;
    while (std::getline(st, l))
        if (l.rfind("Cpus_allowed_list:", 0) == 0)
            cpus = l.substr(l.find_last_of(" \t") + 1);

    return 0;
}

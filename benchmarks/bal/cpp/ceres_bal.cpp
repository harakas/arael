// Ceres runner for the BAL benchmark, modeled on Ceres's own
// examples/simple_bundle_adjuster.cc / bundle_adjuster.cc: the Snavely
// reprojection residual (autodiffed, 2 residuals, 9-param camera block
// + 3-param point block) on BAL's native camera convention. Schur-based
// linear solver -- Ceres's tuned configuration for exactly these
// problems. Protocol:
//   ceres_bal <problem.txt> <params_out> [dense_schur|sparse_schur|
//             iterative_schur|sparse_normal_cholesky]
// prints the shared benchmark protocol line (see ../../cpp/bench.h);
// params_out carries one camera per line (9 values) followed by one
// point per line (3 values).

#include "../../cpp/bench.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
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
    std::vector<double> xy;      // 2 per observation
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

// The Snavely reprojection error: camera = [angle-axis(3), t(3), f, k1,
// k2], negative-z perspective divide, radial distortion.
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double x, double y) : ox(x), oy(y) {}
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        const T xp = -p[0] / p[2];
        const T yp = -p[1] / p[2];
        const T r2 = xp * xp + yp * yp;
        const T distortion = 1.0 + r2 * (camera[7] + camera[8] * r2);
        const T& focal = camera[6];
        residuals[0] = focal * distortion * xp - ox;
        residuals[1] = focal * distortion * yp - oy;
        return true;
    }
    double ox, oy;
};

static bench::Result solve(Bal b, ceres::LinearSolverType linsolver, int max_iters,
                           std::vector<double>* cams_out, std::vector<double>* points_out) {
    ceres::Problem problem;
    for (int i = 0; i < b.n_obs; i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(b.xy[2 * i], b.xy[2 * i + 1])),
            nullptr,
            &b.cameras[9 * b.cam_idx[i]],
            &b.points[3 * b.point_idx[i]]);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = linsolver;
    options.max_num_iterations = max_iters;
    options.num_threads = 1;
    // Same termination class as the other benchmarks. Trust region
    // defaults are Ceres's own -- tuned on these very problems -- with
    // the usual env override.
    options.function_tolerance = 1e-5;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cams_out) *cams_out = b.cameras;
    if (points_out) *points_out = b.points;
    if (getenv("CERES_STATS")) {
        fprintf(stderr, "ceres stats: linear_solver %.1f ms/solve (%d solves), "
                "jacobian %.1f ms, residual %.1f ms, total %.1f s\n",
                1e3 * summary.linear_solver_time_in_seconds / std::max(1, summary.num_linear_solves),
                summary.num_linear_solves,
                1e3 * summary.jacobian_evaluation_time_in_seconds / std::max(1, summary.num_jacobian_evaluations),
                1e3 * summary.residual_evaluation_time_in_seconds / std::max(1, summary.num_residual_evaluations),
                summary.total_time_in_seconds);
    }
    // Ceres records iteration 0 -- the initial cost evaluation, before any step
    // -- as a successful step. It is not one; discount it so accepted and
    // attempts mean the same here as in every other runner.
    const int accepted = std::max(0, summary.num_successful_steps - 1);
    return bench::Result{summary.total_time_in_seconds * 1e3,
                         accepted,
                         accepted + summary.num_unsuccessful_steps,
                         2.0 * summary.initial_cost};
}

// A weak isotropic prior anchoring a point to its solved position. BAL has
// near-degenerate point depths (small-baseline pairs); their exactly-zero
// singular values make SPARSE_QR report rank deficiency and refuse. This tiny
// prior lifts them above the QR rank tolerance without perturbing the
// well-constrained camera directions. arael and g2o (Cholesky-based) need no
// such prior -- they factor through the weak pivots directly.
struct PointPrior {
    PointPrior(double w, const double* p0) : w(w) { p[0] = p0[0]; p[1] = p0[1]; p[2] = p0[2]; }
    template <typename T>
    bool operator()(const T* pt, T* r) const {
        r[0] = T(w) * (pt[0] - T(p[0]));
        r[1] = T(w) * (pt[1] - T(p[1]));
        r[2] = T(w) * (pt[2] - T(p[2]));
        return true;
    }
    double w, p[3];
};

// Covariance, mirroring arael's bal cov_bench. Known calibration: intrinsics
// (f,k1,k2) fixed, recovering 6-DOF pose covariance. Gauge fix (BAL is a
// similarity): cameras 0 and 1 held fully constant. Ceres offers only DENSE_SVD
// and SPARSE_QR -- neither exploits BAL's arrow structure -- and SPARSE_QR needs
// the point prior to reach full rank.
static int cov_mode(Bal b, ceres::LinearSolverType linsolver, const std::string& algo) {
    // Unit prior: negligible next to the per-point observation information
    // (~1e4), but enough to lift every weak point-depth direction above
    // SuiteSparseQR's rank tolerance on all datasets.
    double prior_w = 1.0;
    if (const char* w = getenv("CERES_POINT_PRIOR")) prior_w = atof(w);
    ceres::Problem problem;
    for (int i = 0; i < b.n_obs; i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(b.xy[2 * i], b.xy[2 * i + 1])),
            nullptr,
            &b.cameras[9 * b.cam_idx[i]],
            &b.points[3 * b.point_idx[i]]);
    }
    if (prior_w > 0.0) {
        for (int i = 0; i < b.n_points; i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PointPrior, 3, 3>(new PointPrior(prior_w, &b.points[3 * i])),
                nullptr, &b.points[3 * i]);
        }
    }
    // Known calibration: hold every free camera's intrinsics (f,k1,k2 = idx
    // 6,7,8) constant, recovering 6-DOF pose covariance. Gauge fix: cameras 0 and
    // 1 held fully constant (covers the 7-DOF similarity).
    problem.SetParameterBlockConstant(&b.cameras[0]);
    if (b.n_cams > 1) problem.SetParameterBlockConstant(&b.cameras[9]);
    for (int i = 2; i < b.n_cams; i++) {
        std::vector<int> ci = {6, 7, 8};
        problem.SetManifold(&b.cameras[9 * i], new ceres::SubsetManifold(9, ci));
    }
    ceres::Solver::Options options;
    options.linear_solver_type = linsolver;
    options.max_num_iterations = bench::full_iters(100);
    options.num_threads = 1;
    options.function_tolerance = 1e-5;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ceres::Covariance::Options copts;
    copts.num_threads = 1;
    copts.algorithm_type = (algo == "dense_svd") ? ceres::DENSE_SVD : ceres::SPARSE_QR;

    // Timing: for each entity and each N, the cold cost of Covariance::Compute
    // over N spread blocks (a fresh Covariance each rep -- Ceres has no separable
    // factor-then-extract). Machine-readable COV lines the Rust harness composes
    // into a table. Camera blocks are 6-DOF poses (intrinsics fixed); point blocks
    // are 3-DOF. "all points" (COV_ALL_POINTS) computes the whole point covariance
    // diagonal -- costly, off by default.
    double budget_s = getenv("COV_BUDGET_S") ? atof(getenv("COV_BUDGET_S")) : 5.0;
    int cap = getenv("COV_CAP") ? atoi(getenv("COV_CAP")) : 200;
    double cell_cap = bench::cov_cell_cap_ms();
    const int free_cams = b.n_cams - 2;
    const int ns[] = {1, 2, 8, 32};
    for (int e = 0; e < 2; e++) {
        const char* name = (e == 0) ? "cam" : "point";
        int base = (e == 0) ? 2 : 0;
        int count = (e == 0) ? free_cams : b.n_points;
        int stride = (e == 0) ? 9 : 3;
        const double* blk = (e == 0) ? b.cameras.data() : b.points.data();
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
            std::vector<int> idx = bench::spread(base, count, N);
            std::vector<std::pair<const double*, const double*>> blocks;
            for (int i : idx) blocks.emplace_back(&blk[stride * i], &blk[stride * i]);
            bool ok = true;
            int reps = 0;
            double ms = bench::median_ms(budget_s, cap, &reps, [&] {
                ceres::Covariance cov(copts);
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

    // One std-dev line for validation: camera 2's 6-DOF pose (translation, then
    // rotation). Ceres camera layout is [angle-axis(0..2), t(3..5), f,k1,k2(6..8)].
    ceres::Covariance covariance(copts);
    std::pair<const double*, const double*> b2{&b.cameras[18], &b.cameras[18]};
    if (covariance.Compute({b2}, &problem)) {
        double c[81];
        covariance.GetCovarianceBlock(&b.cameras[18], &b.cameras[18], c);
        fprintf(stderr, "  ceres camera[2] std dev: t=(%.4f,%.4f,%.4f) rot=(%.5f,%.5f,%.5f)\n",
                std::sqrt(c[3 * 9 + 3]), std::sqrt(c[4 * 9 + 4]), std::sqrt(c[5 * 9 + 5]),
                std::sqrt(c[0 * 9 + 0]), std::sqrt(c[1 * 9 + 1]), std::sqrt(c[2 * 9 + 2]));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <problem.txt> <params_out|cov> [linsolver] [cov_algo]\n", argv[0]); return 1; }
    Bal b = load(argv[1]);

    // Covariance feasibility mode: `ceres_bal <problem.txt> cov [linsolver] [algo]`.
    if (std::string(argv[2]) == "cov") {
        ceres::LinearSolverType ls = ceres::SPARSE_SCHUR;
        if (argc > 3) {
            std::string s = argv[3];
            if (s == "dense_schur") ls = ceres::DENSE_SCHUR;
            else if (s == "iterative_schur") ls = ceres::ITERATIVE_SCHUR;
            else if (s == "sparse_normal_cholesky") ls = ceres::SPARSE_NORMAL_CHOLESKY;
        }
        std::string algo = argc > 4 ? argv[4] : "sparse_qr";
        return cov_mode(b, ls, algo);
    }

    ceres::LinearSolverType linsolver = ceres::DENSE_SCHUR;
    if (argc > 3) {
        std::string s = argv[3];
        if (s == "sparse_schur") linsolver = ceres::SPARSE_SCHUR;
        else if (s == "iterative_schur") linsolver = ceres::ITERATIVE_SCHUR;
        else if (s == "sparse_normal_cholesky") linsolver = ceres::SPARSE_NORMAL_CHOLESKY;
        else if (s != "dense_schur") { fprintf(stderr, "unknown linsolver %s\n", s.c_str()); return 1; }
    }

    std::vector<double> cams, points;
    bench::report(
        [&](int n) { return solve(b, linsolver, n, nullptr, nullptr); },
        [&]() { return solve(b, linsolver, bench::full_iters(100), &cams, &points); });

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

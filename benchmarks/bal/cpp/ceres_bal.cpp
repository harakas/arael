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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <problem.txt> <params_out> [linsolver]\n", argv[0]); return 1; }
    Bal b = load(argv[1]);
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

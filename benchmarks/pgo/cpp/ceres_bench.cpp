// Ceres runner for the 2D pose-graph benchmark, modeled on Ceres's own
// examples/slam/pose_graph_2d. Same protocol as the other runners:
//   ceres_bench <file.g2o> <poses_out> <info|unit>
// prints JSON {solve_ms, first_iter_ms, iterations, accepted,
// cpus_allowed} on stdout and writes "x y theta" lines to poses_out.
// Timing wraps only ceres::Solve. Single-threaded by option; the CPU
// pin is inherited from the harness and reported back for the assert.

#include "../../cpp/bench.h"
#include <ceres/ceres.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


struct Pose {
    double x, y, th;
};
struct Edge {
    int a, b;
    double dx, dy, dth;
    double wt, wr; // sqrt information row weights
};

template <typename T>
T NormalizeAngle(const T& a) {
    T two_pi(2.0 * M_PI);
    return a - two_pi * ceres::floor((a + T(M_PI)) / two_pi);
}

struct AngleManifold {
    template <typename T>
    bool Plus(const T* x, const T* delta, T* out) const {
        *out = NormalizeAngle(*x + *delta);
        return true;
    }
    template <typename T>
    bool Minus(const T* y, const T* x, T* out) const {
        *out = NormalizeAngle(*y - *x);
        return true;
    }
};

// Between factor, the residual form of Ceres's pose_graph_2d example
// with sqrt-info row weights (translation info is isotropic in these
// datasets, so the a-frame translation residual has the same norm as
// the reference cost's b-frame form).
struct BetweenError {
    BetweenError(const Edge& e) : e_(e) {}
    template <typename T>
    bool operator()(const T* pa, const T* ya, const T* pb, const T* yb, T* residual) const {
        const T ca = ceres::cos(*ya), sa = ceres::sin(*ya);
        const T gx = pb[0] - pa[0], gy = pb[1] - pa[1];
        residual[0] = (ca * gx + sa * gy - T(e_.dx)) * T(e_.wt);
        residual[1] = (-sa * gx + ca * gy - T(e_.dy)) * T(e_.wt);
        residual[2] = NormalizeAngle(*yb - *ya - T(e_.dth)) * T(e_.wr);
        return true;
    }
    Edge e_;
};

// Unit-weight gauge prior on pose 0 (identical convention to the other
// runners).
struct PriorError {
    PriorError(const Pose& p) : p_(p) {}
    template <typename T>
    bool operator()(const T* pa, const T* ya, T* residual) const {
        residual[0] = pa[0] - T(p_.x);
        residual[1] = pa[1] - T(p_.y);
        residual[2] = *ya - T(p_.th);
        return true;
    }
    Pose p_;
};

static void parse_g2o(const char* path, bool unit, std::vector<Pose>& poses, std::vector<Edge>& edges) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "VERTEX_SE2") {
            int id;
            Pose p;
            ss >> id >> p.x >> p.y >> p.th;
            if (id != (int)poses.size()) { fprintf(stderr, "non-dense vertices\n"); exit(1); }
            poses.push_back(p);
        } else if (tag == "EDGE_SE2") {
            Edge e;
            double i11, i12, i13, i22, i23, i33;
            ss >> e.a >> e.b >> e.dx >> e.dy >> e.dth >> i11 >> i12 >> i13 >> i22 >> i23 >> i33;
            e.wt = unit ? 1.0 : std::sqrt(i11);
            e.wr = unit ? 1.0 : std::sqrt(i33);
            edges.push_back(e);
        }
    }
}

static bench::Result solve(std::vector<Pose> poses, const std::vector<Edge>& edges,
                       const Pose& prior, int max_iters, std::vector<Pose>* out) {
    // Parameter layout follows the Ceres example: separate (x, y) and
    // yaw blocks, yaw on an angle manifold.
    std::vector<double> xy(poses.size() * 2), yaw(poses.size());
    for (size_t i = 0; i < poses.size(); i++) {
        xy[2 * i] = poses[i].x;
        xy[2 * i + 1] = poses[i].y;
        yaw[i] = poses[i].th;
    }
    ceres::Problem problem;
    ceres::Manifold* angle = new ceres::AutoDiffManifold<AngleManifold, 1, 1>;
    for (const Edge& e : edges) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BetweenError, 3, 2, 1, 2, 1>(new BetweenError(e)),
            nullptr, &xy[2 * e.a], &yaw[e.a], &xy[2 * e.b], &yaw[e.b]);
    }
    for (size_t i = 0; i < poses.size(); i++) {
        problem.SetManifold(&yaw[i], angle);
    }
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PriorError, 3, 2, 1>(new PriorError(prior)),
        nullptr, &xy[0], &yaw[0]);

    ceres::Solver::Options options;
    options.max_num_iterations = max_iters;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    // Problem-appropriate initial trust region (the shipped 1e4 default
    // over-damps these well-initialized graphs; see the README's
    // initial-damping policy). Env-overridable for experiments.
    options.initial_trust_region_radius = 1e12;
    if (const char* r = getenv("CERES_RADIUS0")) {
        options.initial_trust_region_radius = atof(r);
    }
    // Same termination class as the other systems.
    options.function_tolerance = 1e-5;
    ceres::Solver::Summary summary;
    auto t0 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (out) {
        out->resize(poses.size());
        for (size_t i = 0; i < poses.size(); i++) {
            (*out)[i] = Pose{xy[2 * i], xy[2 * i + 1], yaw[i]};
        }
    }
    // Count linear solves (factorizations), the unit the parenthesised total
    // reports. num_successful_steps is not that count: it includes iteration 0
    // (the initial eval, which does no solve) and omits the final convergence-
    // detecting solve. num_linear_solves is the true total; num_unsuccessful_steps
    // are the ones that did not reduce cost.
    const int accepted = (int)summary.num_linear_solves - (int)summary.num_unsuccessful_steps;
    return bench::Result{ms, accepted,
                     accepted + (int)summary.num_unsuccessful_steps};
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <g2o> <poses_out> <info|unit>\n", argv[0]); return 1; }
    bool unit = std::string(argv[3]) == "unit";
    std::vector<Pose> poses;
    std::vector<Edge> edges;
    parse_g2o(argv[1], unit, poses, edges);
    Pose prior = poses[0];

    std::vector<Pose> result;
    bench::report(
        [&](int n) { return solve(poses, edges, prior, n, nullptr); },
        [&]() { return solve(poses, edges, prior, 100, &result); });

    std::ofstream out(argv[2]);
    for (const Pose& p : result) out << p.x << " " << p.y << " " << p.th << "\n";

    return 0;
}

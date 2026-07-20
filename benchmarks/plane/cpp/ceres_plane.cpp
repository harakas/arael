// Ceres runner for the plane SLAM benchmark. Poses as t(3) +
// q(4, EigenQuaternionManifold); plane normals on ceres::SphereManifold<3>
// with the distance coefficient as its own scalar block. Residuals are the
// benchmark's shared definitions (see g2o_plane.cpp), autodiffed.
//
//   ceres_plane <scene.txt> <solution_out>
//
// Probes + protocol line via ../../cpp/bench.h; costs reported as chi2
// (2x Ceres's). CERES_RADIUS0 overrides the initial trust region radius;
// CERES_VERBOSE=1 prints per-iteration progress and the full report.

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>

#include "../../cpp/bench.h"

struct Odo {
    double tm[3], rm[9]; // R_m row-major
    double wt, wr;
    template <typename T>
    bool operator()(const T* ti, const T* qi, const T* tj, const T* qj, T* r) const {
        T d[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]};
        T qi_c[4] = {qi[3], -qi[0], -qi[1], -qi[2]}; // w,x,y,z conj (Eigen block is x,y,z,w)
        T local[3];
        ceres::QuaternionRotatePoint(qi_c, d, local);
        r[0] = (local[0] - T(tm[0])) * T(wt);
        r[1] = (local[1] - T(tm[1])) * T(wt);
        r[2] = (local[2] - T(tm[2])) * T(wt);
        // dR = Rm^T Ri^T Rj; vee((dR - dR^T)/2)
        T ri[9], rj[9];
        T qiw[4] = {qi[3], qi[0], qi[1], qi[2]};
        T qjw[4] = {qj[3], qj[0], qj[1], qj[2]};
        ceres::QuaternionToRotation(qiw, ri);
        ceres::QuaternionToRotation(qjw, rj);
        T a[9]; // Ri^T Rj
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                a[3 * c + k] = ri[c] * rj[k] + ri[3 + c] * rj[3 + k] + ri[6 + c] * rj[6 + k];
        T dr[9]; // Rm^T a
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                dr[3 * c + k] = T(rm[c]) * a[k] + T(rm[3 + c]) * a[3 + k] + T(rm[6 + c]) * a[6 + k];
        r[3] = T(0.5) * (dr[7] - dr[5]) * T(wr);
        r[4] = T(0.5) * (dr[2] - dr[6]) * T(wr);
        r[5] = T(0.5) * (dr[3] - dr[1]) * T(wr);
        return true;
    }
};

struct Obs {
    double nm[3], cm;
    double waz, wel, wd;
    template <typename T>
    bool operator()(const T* tp, const T* qp, const T* n, const T* c, T* r) const {
        // predicted local plane
        T qc[4] = {qp[3], -qp[0], -qp[1], -qp[2]};
        T nw[3] = {n[0], n[1], n[2]};
        T nl[3];
        ceres::QuaternionRotatePoint(qc, nw, nl);
        T cl = c[0] + tp[0] * n[0] + tp[1] * n[1] + tp[2] * n[2];
        T h = sqrt(nl[0] * nl[0] + nl[1] * nl[1]);
        T mx = nl[0] * T(nm[0]) + nl[1] * T(nm[1]) + nl[2] * T(nm[2]);
        T my = (T(nm[1]) * nl[0] - T(nm[0]) * nl[1]) / h;
        T mz = (T(nm[2]) * (nl[0] * nl[0] + nl[1] * nl[1])
                - nl[2] * (nl[0] * T(nm[0]) + nl[1] * T(nm[1]))) / h;
        r[0] = atan2(my, mx) * T(waz);
        r[1] = atan2(mz, sqrt(mx * mx + my * my)) * T(wel);
        r[2] = (T(cm) - cl) * T(wd);
        return true;
    }
};

struct Scene {
    std::vector<std::array<double, 3>> t;
    std::vector<std::array<double, 4>> q; // x,y,z,w (Eigen manifold layout)
    std::vector<std::array<double, 3>> pn;
    std::vector<double> pc;
    struct O { int i, j; Odo f; };
    struct B { int p, l; Obs f; };
    std::vector<O> odos;
    std::vector<B> obss;
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
            double x, y, z, qw, qx, qy, qz;
            ss >> x >> y >> z >> qw >> qx >> qy >> qz;
            s.t.push_back({x, y, z});
            s.q.push_back({qx, qy, qz, qw});
        } else if (tag == "plane") {
            double nx, ny, nz, c;
            ss >> nx >> ny >> nz >> c;
            double sc = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            s.pn.push_back({nx * sc, ny * sc, nz * sc});
            s.pc.push_back(c * sc);
        } else if (tag == "odom") {
            Scene::O o; double x, y, z, qw, qx, qy, qz, it_, ir;
            ss >> o.i >> o.j >> x >> y >> z >> qw >> qx >> qy >> qz;
            ss >> it_ >> it_ >> it_ >> ir >> ir >> ir;
            o.f.tm[0] = x; o.f.tm[1] = y; o.f.tm[2] = z;
            Eigen::Matrix3d R = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) o.f.rm[3 * r + c] = R(r, c);
            o.f.wt = sqrt(it_); o.f.wr = sqrt(ir);
            s.odos.push_back(o);
        } else if (tag == "obs") {
            Scene::B b; double nx, ny, nz, c, ia, ie, id;
            ss >> b.p >> b.l >> nx >> ny >> nz >> c >> ia >> ie >> id;
            double sc = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            b.f.nm[0] = nx * sc; b.f.nm[1] = ny * sc; b.f.nm[2] = nz * sc;
            b.f.cm = c * sc;
            b.f.waz = sqrt(ia); b.f.wel = sqrt(ie); b.f.wd = sqrt(id);
            s.obss.push_back(b);
        }
    }
    return s;
}

static bench::Result solve(const Scene& scene, int max_iters,
                           std::vector<std::array<double, 3>>* t_out,
                           std::vector<std::array<double, 4>>* q_out,
                           std::vector<std::array<double, 3>>* pn_out,
                           std::vector<double>* pc_out) {
    // Fresh copies: the probe's reset, outside the timed solve.
    auto t = scene.t;
    auto q = scene.q;
    auto pn = scene.pn;
    auto pc = scene.pc;
    ceres::Problem prob;
    const int n = (int)t.size(), m = (int)pn.size();
    for (int i = 0; i < n; i++) {
        prob.AddParameterBlock(t[i].data(), 3);
        prob.AddParameterBlock(q[i].data(), 4, new ceres::EigenQuaternionManifold);
    }
    for (int j = 0; j < m; j++) {
        prob.AddParameterBlock(pn[j].data(), 3, new ceres::SphereManifold<3>);
        prob.AddParameterBlock(&pc[j], 1);
    }
    prob.SetParameterBlockConstant(t[0].data());
    prob.SetParameterBlockConstant(q[0].data());
    for (const auto& o : scene.odos)
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<Odo, 6, 3, 4, 3, 4>(new Odo(o.f)),
            nullptr, t[o.i].data(), q[o.i].data(), t[o.j].data(), q[o.j].data());
    for (const auto& b : scene.obss)
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<Obs, 3, 3, 4, 3, 1>(new Obs(b.f)),
            nullptr, t[b.p].data(), q[b.p].data(), pn[b.l].data(), &pc[b.l]);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_iters;
    options.num_threads = 1;
    // Shared termination class for this benchmark (see the arael
    // runner's tolerance()); tighter than the sibling benchmarks because
    // these costs are large and the relative test is what bites.
    options.function_tolerance = 1e-7;
    if (const char* f = getenv("PLANE_TOL")) options.function_tolerance = atof(f);
    // Problem-appropriate initial trust region: near-Gauss-Newton on this
    // well-initialized graph, but not so large that the first steps
    // overshoot on the long loops. At 900 poses 1e12 costs 43 iterations
    // and 1e10 rejects five steps; this takes 11 with one rejection.
    // Nothing below 300 poses moves.
    options.initial_trust_region_radius = 1e7;
    if (const char* r = getenv("CERES_RADIUS0")) options.initial_trust_region_radius = atof(r);
    if (getenv("CERES_VERBOSE")) options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);
    if (getenv("CERES_VERBOSE")) fprintf(stderr, "%s\n", summary.FullReport().c_str());
    if (t_out) { *t_out = t; *q_out = q; *pn_out = pn; *pc_out = pc; }
    // Count LINEAR SOLVES (factorizations), the unit the parenthesised total
    // reports; the ones that did not reduce cost are num_unsuccessful_steps.
    const int attempts = (int)summary.num_linear_solves;
    const int accepted = attempts - (int)summary.num_unsuccessful_steps;
    return bench::Result{summary.total_time_in_seconds * 1e3,
                         accepted,
                         attempts,
                         2.0 * summary.initial_cost};
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <scene.txt> <solution_out>\n", argv[0]); return 1; }
    Scene scene = load(argv[1]);

    std::vector<std::array<double, 3>> t, pn;
    std::vector<std::array<double, 4>> q;
    std::vector<double> pc;
    bench::report(
        [&](int iters) { return solve(scene, iters, nullptr, nullptr, nullptr, nullptr); },
        [&]() { return solve(scene, bench::full_iters(200), &t, &q, &pn, &pc); });

    std::ofstream out(argv[2]);
    out.precision(17);
    for (size_t i = 0; i < t.size(); i++)
        out << t[i][0] << " " << t[i][1] << " " << t[i][2] << " "
            << q[i][3] << " " << q[i][0] << " " << q[i][1] << " " << q[i][2] << "\n";
    for (size_t j = 0; j < pn.size(); j++)
        out << pn[j][0] << " " << pn[j][1] << " " << pn[j][2] << " " << pc[j] << "\n";
    return 0;
}

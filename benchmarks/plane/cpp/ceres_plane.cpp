// Ceres runner for the plane SLAM benchmark. Poses as t(3) +
// q(4, EigenQuaternionManifold); plane normals on ceres::SphereManifold<3>
// with the distance coefficient as its own scalar block. Residuals are the
// benchmark's shared definitions (see g2o_plane.cpp), autodiffed.
//
//   ceres_plane <scene.txt> [solution_out] [max_iters]
// Prints one JSON line: {"start_cost":..,"end_cost":..,"iterations":..,
// "accepted":..,"solve_ms":..} with costs as chi2 (2x Ceres's).

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>

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

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <scene.txt> [sol] [iters]\n", argv[0]); return 1; }
    int max_iters = argc > 3 ? atoi(argv[3]) : 100;

    std::vector<std::array<double, 3>> t;
    std::vector<std::array<double, 4>> q; // x,y,z,w (Eigen manifold layout)
    std::vector<std::array<double, 3>> pn;
    std::vector<double> pc;
    ceres::Problem prob;
    std::ifstream in(argv[1]);
    std::string line;
    struct O { int i, j; Odo f; };
    struct B { int p, l; Obs f; };
    std::vector<O> odos; std::vector<B> obss;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "pose") {
            double x, y, z, qw, qx, qy, qz;
            ss >> x >> y >> z >> qw >> qx >> qy >> qz;
            t.push_back({x, y, z});
            q.push_back({qx, qy, qz, qw});
        } else if (tag == "plane") {
            double nx, ny, nz, c;
            ss >> nx >> ny >> nz >> c;
            double s = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            pn.push_back({nx * s, ny * s, nz * s});
            pc.push_back(c * s);
        } else if (tag == "odom") {
            O o; double x, y, z, qw, qx, qy, qz, it_, ir;
            ss >> o.i >> o.j >> x >> y >> z >> qw >> qx >> qy >> qz;
            ss >> it_ >> it_ >> it_ >> ir >> ir >> ir;
            o.f.tm[0] = x; o.f.tm[1] = y; o.f.tm[2] = z;
            Eigen::Matrix3d R = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) o.f.rm[3 * r + c] = R(r, c);
            o.f.wt = sqrt(it_); o.f.wr = sqrt(ir);
            odos.push_back(o);
        } else if (tag == "obs") {
            B b; double nx, ny, nz, c, ia, ie, id;
            ss >> b.p >> b.l >> nx >> ny >> nz >> c >> ia >> ie >> id;
            double s = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            b.f.nm[0] = nx * s; b.f.nm[1] = ny * s; b.f.nm[2] = nz * s;
            b.f.cm = c * s;
            b.f.waz = sqrt(ia); b.f.wel = sqrt(ie); b.f.wd = sqrt(id);
            obss.push_back(b);
        }
    }
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
    for (auto& o : odos)
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<Odo, 6, 3, 4, 3, 4>(new Odo(o.f)),
            nullptr, t[o.i].data(), q[o.i].data(), t[o.j].data(), q[o.j].data());
    for (auto& b : obss)
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<Obs, 3, 3, 4, 3, 1>(new Obs(b.f)),
            nullptr, t[b.p].data(), q[b.p].data(), pn[b.l].data(), &pc[b.l]);

    ceres::Solver::Options opt;
    opt.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    opt.max_num_iterations = max_iters;
    opt.num_threads = 1;
    opt.function_tolerance = 1e-8;
    ceres::Solver::Summary sum;
    ceres::Solve(opt, &prob, &sum);
    int accepted = (int)sum.num_linear_solves - (int)sum.num_unsuccessful_steps;
    printf("{\"start_cost\": %.6f, \"end_cost\": %.6f, \"iterations\": %d, \"accepted\": %d, \"solve_ms\": %.1f}\n",
        2.0 * sum.initial_cost, 2.0 * sum.final_cost,
        accepted + (int)sum.num_unsuccessful_steps, accepted,
        sum.total_time_in_seconds * 1e3);

    if (argc > 2) {
        std::ofstream out(argv[2]);
        out.precision(17);
        for (int i = 0; i < n; i++)
            out << t[i][0] << " " << t[i][1] << " " << t[i][2] << " "
                << q[i][3] << " " << q[i][0] << " " << q[i][1] << " " << q[i][2] << "\n";
        for (int j = 0; j < m; j++)
            out << pn[j][0] << " " << pn[j][1] << " " << pn[j][2] << " " << pc[j] << "\n";
    }
    return 0;
}

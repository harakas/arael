// GTSAM runner for the plane SLAM benchmark. Builds the benchmark's shared
// residuals as CUSTOM NoiseModelFactorN subclasses with ANALYTIC Jacobians
// -- no numeric differentiation, so GTSAM runs at full performance, on
// parity with the autodiff/codegen systems. Poses are gtsam::Pose3, the
// library's own pose type, so its retraction is GTSAM's (the SE(3)
// exponential) and each pose is one variable rather than two. A plane is
// a Vector3 (dy, dz, c) fixed tangent chart around its
// initial normal, the anchor rotation baked into each observation -- the
// same formulation as the factrs and SymForce runners. The gauge is a
// strong prior on pose 0 (zero at the start point). With unit noise
// GTSAM's error is 0.5 * sum||r||^2, so initial_cost = 2 * graph.error.
// The analytic Jacobians are checked against gtsam::numericalDerivative
// when GTSAM_VERIFY_JAC=1 (verification only, never in the timed solve).
//
//   gtsam_plane <scene.txt> <solution_out>
//
// Probes + protocol line via ../../cpp/bench.h. GTSAM_LAMBDA0 overrides
// the LM initial damping (default 1e-9, near-Gauss-Newton).

#include "../../cpp/bench.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using gtsam::Key;
using gtsam::Matrix;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector;
using V3 = gtsam::Vector3;
using M3 = Eigen::Matrix3d;

static Key X(int i) { return gtsam::Symbol('x', i); }
static Key L(int j) { return gtsam::Symbol('l', j); }

static M3 skew(const V3& v) {
    M3 m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
static V3 vee_half(const M3& m) {  // vee((m - m^T)/2)
    return V3(0.5 * (m(2, 1) - m(1, 2)), 0.5 * (m(0, 2) - m(2, 0)),
              0.5 * (m(1, 0) - m(0, 1)));
}

// GTSAM takes its Jacobian outputs as OptionalJacobian, which a ternary
// cannot mix with a raw pointer -- build it explicitly instead. A
// default-constructed one means "not wanted", which is what the
// cost-only evaluations pass.
template <int R, int C>
static gtsam::OptionalJacobian<R, C> opt(bool want, Eigen::Matrix<double, R, C>& m) {
    return want ? gtsam::OptionalJacobian<R, C>(m) : gtsam::OptionalJacobian<R, C>();
}

// The chart embed and its derivative: local direction from (dy, dz),
// l = [1 - u/(2 s2), dz/s2, -dy/s2], s2 = 1 + u/4, u = dy^2 + dz^2.
static V3 chart_local(double dy, double dz, Eigen::Matrix<double, 3, 2>* dl) {
    double u = dy * dy + dz * dz;
    double s2 = 1.0 + u / 4.0;
    if (dl) {
        double s22 = s2 * s2;
        (*dl)(0, 0) = -dy / s22;
        (*dl)(0, 1) = -dz / s22;
        (*dl)(1, 0) = -dy * dz / (2.0 * s22);
        (*dl)(1, 1) = (s2 - dz * dz / 2.0) / s22;
        (*dl)(2, 0) = -(s2 - dy * dy / 2.0) / s22;
        (*dl)(2, 1) = dy * dz / (2.0 * s22);
    }
    return V3(1.0 - u / (2.0 * s2), dz / s2, -dy / s2);
}

// -- the factors (residual == scene_cost, whitening baked in so the noise
//    model is unit; analytic Jacobians) --

// err_t = R_i^T (t_j - t_i) - t_m; err_r = vee((R_m^T R_i^T R_j - .^T)/2).
//
// Both halves are functions of the RELATIVE pose alone -- its translation
// is R_i^T (t_j - t_i) and its rotation R_i^T R_j -- so `between` supplies
// the chain rule back to the two poses and only the small derivative of
// the residual with respect to the relative pose is hand-written.
struct OdoFactor : public gtsam::NoiseModelFactorN<Pose3, Pose3> {
    V3 tm; M3 rm_t; double wt, wr;
    OdoFactor(Key a, Key b, const V3& tm_, const M3& rmt, double wt_, double wr_)
        : NoiseModelFactorN<Pose3, Pose3>(gtsam::noiseModel::Unit::Create(6), a, b),
          tm(tm_), rm_t(rmt), wt(wt_), wr(wr_) {}
    Vector evaluateError(const Pose3& a, const Pose3& b,
                         boost::optional<Matrix&> H1 = boost::none,
                         boost::optional<Matrix&> H2 = boost::none) const override {
        const bool want = H1 || H2;
        gtsam::Matrix6 Ha, Hb;
        Pose3 rel = a.between(b, opt(want, Ha), opt(want, Hb));

        gtsam::Matrix36 Ht, Hr;
        V3 t = rel.translation(opt(want, Ht));
        M3 dR = rm_t * rel.rotation(opt(want, Hr)).matrix();

        gtsam::Vector6 e;
        e.head<3>() = (t - tm) * wt;
        e.tail<3>() = vee_half(dR) * wr;

        if (want) {
            // d vee((A - A^T)/2) for A -> A(I + [w]x) is (tr(A) I - A^T) w / 2.
            gtsam::Matrix6 J;
            J.topRows<3>() = wt * Ht;
            J.bottomRows<3>() =
                0.5 * wr * (dR.trace() * M3::Identity() - dR.transpose()) * Hr;
            if (H1) *H1 = J * Ha;
            if (H2) *H2 = J * Hb;
        }
        return e;
    }
};

// The shared plane observation: azimuth/elevation of the measured normal
// in the frame aligning the predicted local normal with e1, plus the
// distance difference. The plane variable is the (dy, dz, c) chart.
struct ObsFactor : public gtsam::NoiseModelFactorN<Pose3, V3> {
    M3 anchor; V3 nm; double cm, waz, wel, wd;
    ObsFactor(Key p, Key l, const M3& a, const V3& nm_, double cm_,
              double waz_, double wel_, double wd_)
        : NoiseModelFactorN<Pose3, V3>(gtsam::noiseModel::Unit::Create(3), p, l),
          anchor(a), nm(nm_), cm(cm_), waz(waz_), wel(wel_), wd(wd_) {}
    Vector evaluateError(const Pose3& pose, const V3& x,
                         boost::optional<Matrix&> H1 = boost::none,
                         boost::optional<Matrix&> H2 = boost::none) const override {
        const bool want = H1 || H2;
        Eigen::Matrix<double, 3, 2> dl;
        V3 l = chart_local(x[0], x[1], want ? &dl : nullptr);
        V3 nw = anchor * l;

        gtsam::Matrix36 Ht, Hr;
        V3 tp = pose.translation(opt(want, Ht));
        M3 rp = pose.rotation(opt(want, Hr)).matrix();

        V3 nl = rp.transpose() * nw;
        double cl = x[2] + tp.dot(nw);
        double h = sqrt(nl.x() * nl.x() + nl.y() * nl.y());
        double mx = nl.dot(nm);
        double my = (nm.y() * nl.x() - nm.x() * nl.y()) / h;
        double mz = (nm.z() * (nl.x() * nl.x() + nl.y() * nl.y())
                     - nl.z() * (nl.x() * nm.x() + nl.y() * nm.y())) / h;
        double g = sqrt(mx * mx + my * my);
        V3 e(atan2(my, mx) * waz, atan2(mz, g) * wel, (cm - cl) * wd);

        if (want) {
            // Rows of d(mx, my, mz)/d nl, then the two atan2.
            V3 dh(nl.x() / h, nl.y() / h, 0.0);
            V3 dmx = nm;
            V3 dp(nm.y(), -nm.x(), 0.0);
            V3 dmy = (dp - my * dh) / h;
            V3 dq(2.0 * nm.z() * nl.x() - nl.z() * nm.x(),
                  2.0 * nm.z() * nl.y() - nl.z() * nm.y(),
                  -(nl.x() * nm.x() + nl.y() * nm.y()));
            V3 dmz = (dq - mz * dh) / h;
            V3 dr0 = waz * (mx * dmy - my * dmx) / (mx * mx + my * my);
            V3 dg = (mx * dmx + my * dmy) / g;
            V3 dr1 = wel * (g * dmz - mz * dg) / (g * g + mz * mz);

            if (H1) {
                // nl responds to a rotation of the pose as [nl]x; cl to a
                // translation as nw. Both come back through GTSAM's own
                // pose Jacobians.
                *H1 = Matrix::Zero(3, 6);
                M3 sk = skew(nl);
                H1->row(0) = dr0.transpose() * sk * Hr;
                H1->row(1) = dr1.transpose() * sk * Hr;
                H1->row(2) = -wd * nw.transpose() * Ht;
            }
            if (H2) {
                // The chart moves the world normal, which moves both the
                // local normal and the distance.
                Eigen::Matrix<double, 3, 2> dnw = anchor * dl;
                Eigen::Matrix<double, 3, 2> dnl = rp.transpose() * dnw;
                *H2 = Matrix::Zero(3, 3);
                H2->block<1, 2>(0, 0) = dr0.transpose() * dnl;
                H2->block<1, 2>(1, 0) = dr1.transpose() * dnl;
                H2->block<1, 2>(2, 0) = -wd * (tp.transpose() * dnw);
                (*H2)(2, 2) = -wd;
            }
        }
        return e;
    }
};

// Strong gauge prior on pose 0 (zero at the start point).
struct PriorFactor : public gtsam::NoiseModelFactorN<Pose3> {
    Pose3 prior; double w;
    PriorFactor(Key k, const Pose3& p0, double w_)
        : NoiseModelFactorN<Pose3>(gtsam::noiseModel::Unit::Create(6), k),
          prior(p0), w(w_) {}
    Vector evaluateError(const Pose3& p,
                         boost::optional<Matrix&> H = boost::none) const override {
        gtsam::Matrix6 Hb;
        Pose3 rel = prior.between(p, {}, opt(bool(H), Hb));
        gtsam::Matrix36 Ht, Hr;
        V3 t = rel.translation(opt(bool(H), Ht));
        M3 dR = rel.rotation(opt(bool(H), Hr)).matrix();
        gtsam::Vector6 e;
        e.head<3>() = t * w;
        e.tail<3>() = vee_half(dR) * w;
        if (H) {
            gtsam::Matrix6 J;
            J.topRows<3>() = w * Ht;
            J.bottomRows<3>() =
                0.5 * w * (dR.trace() * M3::Identity() - dR.transpose()) * Hr;
            *H = J * Hb;
        }
        return e;
    }
};

struct Scene {
    std::vector<V3> t;
    std::vector<Eigen::Quaterniond> q;
    std::vector<M3> anchor;
    std::vector<double> pc;
    struct O { int i, j; V3 tm; M3 rm_t; double wt, wr; };
    struct B { int p, l; V3 nm; double cm, waz, wel, wd; };
    std::vector<O> odos;
    std::vector<B> obss;
};

static M3 anchor_of(const V3& dir) {
    return Eigen::Quaterniond::FromTwoVectors(V3(1, 0, 0), dir.normalized())
        .toRotationMatrix();
}

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
            s.t.emplace_back(x, y, z);
            s.q.push_back(Eigen::Quaterniond(qw, qx, qy, qz).normalized());
        } else if (tag == "plane") {
            double nx, ny, nz, c;
            ss >> nx >> ny >> nz >> c;
            V3 n(nx, ny, nz);
            s.anchor.push_back(anchor_of(n));
            s.pc.push_back(c / n.norm());
        } else if (tag == "odom") {
            Scene::O o; double x, y, z, qw, qx, qy, qz, it_, ir;
            ss >> o.i >> o.j >> x >> y >> z >> qw >> qx >> qy >> qz;
            ss >> it_ >> it_ >> it_ >> ir >> ir >> ir;
            o.tm = V3(x, y, z);
            o.rm_t = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix().transpose();
            o.wt = sqrt(it_); o.wr = sqrt(ir);
            s.odos.push_back(o);
        } else if (tag == "obs") {
            Scene::B b; double nx, ny, nz, c, ia, ie, id;
            ss >> b.p >> b.l >> nx >> ny >> nz >> c >> ia >> ie >> id;
            V3 n(nx, ny, nz);
            b.nm = n.normalized();
            b.cm = c / n.norm();
            b.waz = sqrt(ia); b.wel = sqrt(ie); b.wd = sqrt(id);
            s.obss.push_back(b);
        }
    }
    return s;
}

static void build(const Scene& s, gtsam::NonlinearFactorGraph& graph, gtsam::Values& init) {
    const int n = (int)s.t.size(), m = (int)s.anchor.size();
    for (int i = 0; i < n; i++) init.insert(X(i), Pose3(Rot3(s.q[i]), s.t[i]));
    for (int j = 0; j < m; j++) init.insert(L(j), V3(0, 0, s.pc[j]));
    for (const auto& o : s.odos)
        graph.emplace_shared<OdoFactor>(X(o.i), X(o.j), o.tm, o.rm_t, o.wt, o.wr);
    for (const auto& b : s.obss)
        graph.emplace_shared<ObsFactor>(X(b.p), L(b.l), s.anchor[b.l], b.nm,
                                        b.cm, b.waz, b.wel, b.wd);
    graph.emplace_shared<PriorFactor>(X(0), Pose3(Rot3(s.q[0]), s.t[0]), 1e6);
}

// GTSAM_VERIFY_JAC=1: check the analytic Jacobians against gtsam's numeric
// differentiation on the first few factors of each kind.
static void verify_jacobians(const Scene& s) {
    using gtsam::numericalDerivative21;
    using gtsam::numericalDerivative22;
    double worst = 0.0;
    for (int k = 0; k < std::min<int>(4, (int)s.odos.size()); k++) {
        const auto& o = s.odos[k];
        OdoFactor f(X(o.i), X(o.j), o.tm, o.rm_t, o.wt, o.wr);
        Pose3 a(Rot3(s.q[o.i]), s.t[o.i]), b(Rot3(s.q[o.j]), s.t[o.j]);
        Matrix H1, H2;
        f.evaluateError(a, b, H1, H2);
        auto fn = [&](const Pose3& x, const Pose3& y) {
            return Vector(f.evaluateError(x, y));
        };
        worst = std::max(worst,
            (H1 - numericalDerivative21<Vector, Pose3, Pose3>(fn, a, b)).cwiseAbs().maxCoeff());
        worst = std::max(worst,
            (H2 - numericalDerivative22<Vector, Pose3, Pose3>(fn, a, b)).cwiseAbs().maxCoeff());
    }
    for (int k = 0; k < std::min<int>(6, (int)s.obss.size()); k++) {
        const auto& b = s.obss[k];
        ObsFactor f(X(b.p), L(b.l), s.anchor[b.l], b.nm, b.cm, b.waz, b.wel, b.wd);
        Pose3 p(Rot3(s.q[b.p]), s.t[b.p]);
        V3 x(0.03, -0.02, s.pc[b.l]);  // off the chart centre
        Matrix H1, H2;
        f.evaluateError(p, x, H1, H2);
        auto fn = [&](const Pose3& a, const V3& c) { return Vector(f.evaluateError(a, c)); };
        worst = std::max(worst,
            (H1 - numericalDerivative21<Vector, Pose3, V3>(fn, p, x)).cwiseAbs().maxCoeff());
        worst = std::max(worst,
            (H2 - numericalDerivative22<Vector, Pose3, V3>(fn, p, x)).cwiseAbs().maxCoeff());
    }
    {
        PriorFactor f(X(0), Pose3(Rot3(s.q[0]), s.t[0]), 1.0);
        Pose3 p(Rot3(s.q[3 % (int)s.q.size()]), s.t[3 % (int)s.t.size()]);
        Matrix H;
        f.evaluateError(p, H);
        auto fn = [&](const Pose3& a) { return Vector(f.evaluateError(a)); };
        worst = std::max(worst,
            (H - gtsam::numericalDerivative11<Vector, Pose3>(fn, p)).cwiseAbs().maxCoeff());
    }
    fprintf(stderr, "gtsam jac check (odo/obs/prior) max|analytic-numeric| = %.3e\n", worst);
    if (worst > 1e-4) { fprintf(stderr, "JACOBIAN MISMATCH\n"); exit(2); }
}

static bench::Result solve(const Scene& s, int max_iters,
                           std::vector<double>* pose_out,
                           std::vector<double>* plane_out) {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values init;
    build(s, graph, init);

    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SILENT");
    params.setMaxIterations(max_iters);
    double tol = 1e-7;  // shared termination class
    if (const char* t = getenv("PLANE_TOL")) tol = atof(t);
    params.setRelativeErrorTol(tol);
    params.setAbsoluteErrorTol(tol);
    // Problem-appropriate initial damping (near-Gauss-Newton on this
    // well-initialized graph), matching the sibling benchmark policy.
    params.lambdaInitial = 1e-9;
    if (const char* l = getenv("GTSAM_LAMBDA0")) params.lambdaInitial = atof(l);

    double initial_cost = 2.0 * graph.error(init);  // GTSAM error is 0.5*sum r^2

    auto t0 = std::chrono::steady_clock::now();
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, init, params);
    gtsam::Values result = optimizer.optimize();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    const int n = (int)s.t.size(), m = (int)s.anchor.size();
    if (pose_out) {
        pose_out->clear();
        for (int i = 0; i < n; i++) {
            Pose3 p = result.at<Pose3>(X(i));
            Eigen::Quaterniond q(p.rotation().matrix());
            V3 t = p.translation();
            double vals[7] = {t.x(), t.y(), t.z(), q.w(), q.x(), q.y(), q.z()};
            pose_out->insert(pose_out->end(), vals, vals + 7);
        }
        plane_out->clear();
        for (int j = 0; j < m; j++) {
            V3 x = result.at<V3>(L(j));
            V3 nw = s.anchor[j] * chart_local(x[0], x[1], nullptr);
            double vals[4] = {nw.x(), nw.y(), nw.z(), x[2]};
            plane_out->insert(plane_out->end(), vals, vals + 4);
        }
    }
    // GTSAM keeps its damping retries inside iterate() and its API exposes
    // no way to count them: accepted steps are all it reports.
    const int iters = (int)optimizer.iterations();
    return bench::Result{ms, iters, iters, initial_cost};
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <scene.txt> <solution_out>\n", argv[0]); return 1; }
    Scene s = load(argv[1]);
    if (getenv("GTSAM_VERIFY_JAC")) verify_jacobians(s);

    std::vector<double> poses, planes;
    bench::report(
        [&](int iters) { return solve(s, iters, nullptr, nullptr); },
        [&]() { return solve(s, bench::full_iters(200), &poses, &planes); });

    std::ofstream out(argv[2]);
    out.precision(17);
    for (size_t i = 0; i + 6 < poses.size(); i += 7)
        out << poses[i] << " " << poses[i + 1] << " " << poses[i + 2] << " "
            << poses[i + 3] << " " << poses[i + 4] << " " << poses[i + 5] << " "
            << poses[i + 6] << "\n";
    for (size_t j = 0; j + 3 < planes.size(); j += 4)
        out << planes[j] << " " << planes[j + 1] << " " << planes[j + 2] << " "
            << planes[j + 3] << "\n";
    return 0;
}

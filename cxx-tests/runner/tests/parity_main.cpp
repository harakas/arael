// Builds the fixture problem through the generated C++ interface,
// solves, and prints "name value" lines for the Rust side to compare.
#include <fit.hpp>
#include <cstdio>
#include <cstring>
#include <string>

using namespace arael;

static void p(const char* n, double v) { std::printf("%s %.17e\n", n, v); }
static void pi(const char* n, long v) { std::printf("%s %ld\n", n, v); }

static void fill(Fit& fit) {
    for (int i = 0; i < 6; i++) {
        auto o = fit.obs().push();
        o.set_x(double(i));
        o.set_y(2.0 * i + 1.0 + (i % 2 == 0 ? 0.05 : -0.05));
    }
    double t[3] = {1.5, -0.3, 0.7};
    double w[3] = {1.0, 2.0, 0.5};
    for (int i = 0; i < 3; i++) {
        auto n = fit.items().push();
        n.set_t(t[i]);
        n.set_w(w[i]);
    }
}

int main() {
    Fit fit;
    fill(fit);
    pi("clean", std::strlen(fit.validate()) == 0 ? 1 : 0);
    pi("n_obs", fit.obs().size());
    pi("n_items", fit.items().size());
    p("obs3_y", fit.obs()[3].y());
    p("item1_t", fit.items()[1].t());

    LmConfig cfg;
    cfg.max_iters = 50;
    LmResult r = fit.solve_dense(cfg);
    pi("dense_status", long(r.status));
    p("dense_start", r.start_cost);
    p("dense_end", r.end_cost);
    pi("dense_iters", r.iterations);
    p("dense_m", fit.m());
    p("dense_c", fit.c());
    for (int i = 0; i < 3; i++) {
        char name[16];
        std::snprintf(name, sizeof name, "dense_v%d", i);
        p(name, fit.items()[i].v());
    }

    Fit fit2;
    fill(fit2);
    LmResult r2 = fit2.solve_sparse(cfg);
    pi("sparse_status", long(r2.status));
    p("sparse_end", r2.end_cost);
    p("sparse_m", fit2.m());
    p("sparse_c", fit2.c());

    // Stage 3 surface: math types, deque pose chain with ties through
    // refs, arena with a removal, nested Info with an Option entity,
    // fixed euler param, param optimize flags.
    Fit f3;
    fill(f3);
    f3.set_cal({0.25, -0.5});

    vect3d targets[3] = {{0, 0, 0}, {1, 0.5, 0}, {2, 1, 0}};
    auto p1 = f3.poses().push_back();
    auto p2 = f3.poses().push_back();
    auto p0 = f3.poses().push_front();
    PoseRef ps[3] = {p0, p1, p2};
    for (int i = 0; i < 3; i++) {
        ps[i].set_target(targets[i]);
        ps[i].set_pos({0.1 * i, -0.1 * i, 0.05});
        ps[i].set_ea({0.1, 0.2, 0.3 * i});
        ps[i].set_ea_optimize(false);
    }
    auto gps = ps[0].info().make_gps();
    gps.set_pos({7.0, 8.0, 9.0});
    gps.set_isigma(2.5f);

    auto t01 = f3.ties().push();
    t01.set_a(f3.poses().ref_at(0));
    t01.set_b(f3.poses().ref_at(1));
    t01.set_d({1.0, 0.4, 0.0});
    t01.set_w(3.0);
    auto t12 = f3.ties().push();
    t12.set_a(f3.poses().ref_at(1));
    t12.set_b(f3.poses().ref_at(2));
    t12.set_d({1.0, 0.6, 0.0});
    t12.set_w(3.0);

    auto m0 = f3.marks().push();
    auto m1 = f3.marks().push();
    auto m2 = f3.marks().push();
    f3.marks().get(m0).set_t(0.4);
    f3.marks().get(m0).set_w(1.0);
    f3.marks().get(m1).set_t(9.0);
    f3.marks().get(m1).set_w(1.0);
    f3.marks().get(m2).set_t(-0.6);
    f3.marks().get(m2).set_w(2.0);
    f3.marks().remove(m1);

    pi("s3_clean", std::strlen(f3.validate()) == 0 ? 1 : 0);
    LmResult r3 = f3.solve_dense(cfg);
    pi("s3_status", long(r3.status));
    p("s3_end", r3.end_cost);
    p("s3_cal_x", f3.cal().x);
    p("s3_cal_y", f3.cal().y);
    for (int i = 0; i < 3; i++) {
        vect3d q = f3.poses()[i].pos();
        char name[24];
        std::snprintf(name, sizeof name, "s3_p%d", i);
        std::string base = name;
        p((base + "_x").c_str(), q.x);
        p((base + "_y").c_str(), q.y);
        p((base + "_z").c_str(), q.z);
    }
    vect3d ea0 = f3.poses()[0].ea();
    p("s3_ea0_z", ea0.z);
    pi("s3_has_gps0", ps[0].info().has_gps() ? 1 : 0);
    pi("s3_has_gps1", ps[1].info().has_gps() ? 1 : 0);
    p("s3_gps0_y", ps[0].info().gps().pos().y);
    p("s3_gps0_isigma", double(ps[0].info().gps().isigma()));
    pi("s3_marks_len", f3.marks().size());
    p("s3_mark0_v", f3.marks().get(m0).v());
    p("s3_mark2_v", f3.marks().get(m2).v());

    // Degenerate model: the root's m/c stay unconstrained (no obs)
    // while one item gives a nonzero cost, so assembly reaches the
    // zero diagonal. The failure comes back as a status code plus
    // text, not a crash.
    Fit bad;
    auto n = bad.items().push();
    n.set_t(1.0);
    n.set_w(1.0);
    LmResult rb = bad.solve_dense(cfg);
    pi("bad_status", long(rb.status));
    pi("bad_has_error", std::strlen(bad.last_error()) > 0 ? 1 : 0);

    return 0;
}

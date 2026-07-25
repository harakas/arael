// Builds the fixture problem through the generated C++ interface,
// solves, and prints "name value" lines for the Rust side to compare.
#include <fit.hpp>
#include <cstdio>
#include <cstring>
#include <string>

using namespace cxx_fit;

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

    // The config holds the preset's actual Rust values -- print every
    // field so the Rust side can verify the layout end to end.
    LmConfig defs;
    p("cfg_abs", defs.abs_precision);
    p("cfg_rel", defs.rel_precision);
    pi("cfg_max_iters", defs.max_iters);
    pi("cfg_min_iters", defs.min_iters);
    pi("cfg_patience", defs.patience);
    pi("cfg_threads", defs.num_threads);
    pi("cfg_verbose", defs.verbose ? 1 : 0);
    p("cfg_lambda", defs.initial_lambda);
    p("cfg_cost_threshold", defs.cost_threshold);
    p("cfg_lambda_floor", defs.lambda_floor);
    pi("cfg_grad_has", defs.gradient_tolerance.has_value() ? 1 : 0);
    pi("cfg_time_has", defs.time_limit_seconds.has_value() ? 1 : 0);
    p("cfg_wc_lambda", LmConfig::well_conditioned().initial_lambda);

    LmConfig cfg;
    cfg.max_iters = 50;
    LmResult r = fit.solve_dense(cfg).value();
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

    // Covariance at the solution: the 1x1 marginal of the first item.
    auto cov = fit.assemble_covariance(CovMode::AllMarginals);
    pi("cov_ok", cov.is_ok() ? 1 : 0);
    if (cov.is_ok()) {
        auto m = cov->marginal(fit.items()[0]);
        pi("cov_item0_ok", m.is_ok() ? 1 : 0);
        p("cov_item0", m.value());
    }

    Fit fit2;
    fill(fit2);
    LmResult r2 = fit2.solve_sparse(cfg).value();
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
    Pose ps[3] = {p0, p1, p2};
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
    LmResult r3 = f3.solve_dense(cfg).value();
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
    p("s3_gps0_y", ps[0].info().gps()->pos().y);
    p("s3_gps0_isigma", double(ps[0].info().gps()->isigma()));
    pi("s3_marks_len", f3.marks().size());
    // Range-for over every container kind: vec, deque, and the arena
    // cursor (which must skip the removed slot).
    double it_obs = 0;
    for (auto o : f3.obs()) it_obs += o.y();
    p("it_obs_sum", it_obs);
    double it_pose = 0;
    for (auto q : f3.poses()) it_pose += q.pos().x;
    p("it_pose_sum", it_pose);
    double it_marks = 0;
    uint32_t it_marks_n = 0;
    for (auto mk : f3.marks()) { it_marks += mk.t(); it_marks_n++; }
    p("it_marks_sum", it_marks);
    pi("it_marks_n", it_marks_n);
    // Manual iterators with operator->.
    double it_arrow = 0;
    for (auto it = f3.obs().begin(); it != f3.obs().end(); ++it)
        it_arrow += it->x();
    for (auto it = f3.marks().begin(); it != f3.marks().end(); ++it)
        it_arrow += it->w();
    p("it_arrow_sum", it_arrow);
    // Backward walks: position-weighted sums pin the ORDER, not just
    // the membership.
    double back_obs = 0;
    int k = 1;
    for (auto it = f3.obs().end(); it != f3.obs().begin();) {
        --it;
        back_obs += k * it->y();
        k++;
    }
    p("back_obs", back_obs);
    double back_marks = 0;
    k = 1;
    for (auto it = f3.marks().end(); it != f3.marks().begin();) {
        --it;
        back_marks += k * it->t();
        k++;
    }
    p("back_marks", back_marks);
    // rbegin/rend with the arrow proxy, over vec and the holed arena.
    double r_obs = 0;
    k = 1;
    for (auto rit = f3.obs().rbegin(); rit != f3.obs().rend(); ++rit) {
        r_obs += k * rit->y();
        k++;
    }
    p("r_obs", r_obs);
    double r_marks = 0;
    k = 1;
    for (auto rit = f3.marks().rbegin(); rit != f3.marks().rend(); ++rit) {
        r_marks += k * rit->t();
        k++;
    }
    p("r_marks", r_marks);
    p("s3_mark0_v", f3.marks().get(m0).v());
    p("s3_mark2_v", f3.marks().get(m2).v());

    // Container removal ops on a scratch model: vec pop/truncate/clear,
    // deque pops from both ends, arena clear.
    Fit f4;
    fill(f4);
    f4.obs().pop();
    pi("ops_obs_after_pop", f4.obs().size());
    f4.obs().truncate(2);
    pi("ops_obs_after_trunc", f4.obs().size());
    f4.obs().clear();
    pi("ops_obs_after_clear", f4.obs().size());
    f4.poses().push_back();
    f4.poses().push_back();
    f4.poses().push_front();
    f4.poses().pop_front();
    f4.poses().pop_back();
    pi("ops_poses_left", f4.poses().size());
    pi("ops_pop_empty", f4.obs().pop() ? 1 : 0);
    f4.marks().push();
    f4.marks().push();
    f4.marks().clear();
    pi("ops_marks_after_clear", f4.marks().size());

    // reserve/empty/contains/try_get/front/back on a scratch model.
    Fit f5;
    f5.obs().reserve(64);
    f5.items().reserve(64);
    f5.poses().reserve(64);
    f5.marks().reserve(64);
    pi("cap_obs_empty", f5.obs().empty() ? 1 : 0);
    auto i5 = f5.items().push();
    i5.set_t(0.25);
    auto d5 = f5.poses().push_back();
    d5.set_pos(vect3d{1.5, 0, 0});
    f5.poses().push_back().set_pos(vect3d{2.5, 0, 0});
    auto a5 = f5.marks().push();
    f5.marks().get(a5).set_t(0.75);
    auto a5b = f5.marks().push();
    pi("cap_obs_still_empty", f5.obs().empty() ? 1 : 0);
    pi("cap_items_nonempty", f5.items().empty() ? 1 : 0);
    NRef i5r = f5.items().ref_at(0);
    pi("cap_items_contains", f5.items().contains(i5r) ? 1 : 0);
    pi("cap_items_contains_default", f5.items().contains(NRef{}) ? 1 : 0);
    p("cap_items_try_get", f5.items().try_get(i5r).value().t());
    pi("cap_poses_contains", f5.poses().contains(f5.poses().ref_at(1)) ? 1 : 0);
    p("cap_poses_front_x", f5.poses().front().pos().x);
    p("cap_poses_back_x", f5.poses().back().pos().x);
    pi("cap_marks_contains", f5.marks().contains(a5) ? 1 : 0);
    p("cap_marks_try_get", f5.marks().try_get(a5).value().t());
    f5.marks().remove(a5b);
    pi("cap_marks_stale_contains", f5.marks().contains(a5b) ? 1 : 0);
    pi("cap_marks_stale_try_get", f5.marks().try_get(a5b).has_value() ? 1 : 0);

    // Degenerate model: the root's m/c stay unconstrained (no obs)
    // while one item gives a nonzero cost, so assembly reaches the
    // zero diagonal. The failure comes back as a status code plus
    // text, not a crash.
    Fit bad;
    auto n = bad.items().push();
    n.set_t(1.0);
    n.set_w(1.0);
    SolveResult rb = bad.solve_dense(cfg);
    pi("bad_status", long(rb.is_err() ? rb.error().status : rb.value().status));
    pi("bad_has_error", rb.is_err() && std::strlen(rb.error().message) > 0 ? 1 : 0);

    return 0;
}

// Builds the fixture problem through the generated C++ interface,
// solves, and prints "name value" lines for the Rust side to compare.
#include <fit.hpp>
#include <cstdio>
#include <cstring>

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

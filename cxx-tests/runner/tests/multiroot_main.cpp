// One translation unit using BOTH roots of the multi-root fixture:
// nested namespaces keep the two solver surfaces apart (the roots
// solve at different precisions), and the single capi staticlib
// carries both symbol sets. Prints "name value" lines for the Rust
// side to compare.
#include <line.hpp>
#include <decay.hpp>
#include <cstdio>

static void p(const char* n, double v) { std::printf("%s %.17e\n", n, v); }

int main() {
    cxx_mr::line::Line line;
    for (int i = 1; i <= 4; i++) {
        auto ob = line.obs().push();
        ob.set_x(double(i));
        ob.set_y(3.0 * i + (i % 2 == 0 ? 0.25 : -0.25));
    }
    cxx_mr::line::LmConfig lc;
    lc.max_iters = 30;
    auto lr = line.solve_dense(lc).value();
    p("line_status", double(lr.status));
    p("line_end", double(lr.end_cost));
    p("line_k", line.k());

    cxx_mr::decay::Decay decay;
    const float t[3] = {0.5f, -1.5f, 2.0f};
    for (int i = 0; i < 3; i++) {
        auto c = decay.cells().push();
        c.set_t(t[i]);
        c.set_w(1.0f + float(i));
    }
    cxx_mr::decay::LmConfig dc;
    dc.max_iters = 30;
    auto dr = decay.solve_dense(dc).value();
    p("decay_status", double(dr.status));
    p("decay_end", double(dr.end_cost));
    for (int i = 0; i < 3; i++) {
        char name[16];
        std::snprintf(name, sizeof name, "cell%d", i);
        p(name, double(decay.cells()[uint32_t(i)].v()));
    }
    return 0;
}

// Line fit through the generated C++ interface -- the smallest
// realistic consumer, doubling as the CMake integration smoke test.
#include <fit.hpp>
#include <cstdio>

using namespace cxx_fit;

int main() {
    Fit fit;
    // y = 2x + 1 with alternating +-0.05 noise.
    for (int i = 0; i < 6; i++) {
        auto o = fit.obs().push();
        o.set_x(double(i));
        o.set_y(2.0 * i + 1.0 + (i % 2 == 0 ? 0.05 : -0.05));
    }

    LmConfig cfg;
    cfg.max_iters = 50;
    SolveResult r = fit.solve_dense(cfg);
    if (r.is_err()) {
        std::fprintf(stderr, "solve failed: %s\n", r.error().message);
        return 1;
    }
    std::printf("m %.12f\nc %.12f\nend_cost %.3e\n",
        fit.m(), fit.c(), r->end_cost);

    // The fit must land near the generating line.
    if (fit.m() < 1.9 || fit.m() > 2.1 || fit.c() < 0.8 || fit.c() > 1.2) {
        std::fprintf(stderr, "fit off: m=%f c=%f\n", fit.m(), fit.c());
        return 1;
    }
    return 0;
}

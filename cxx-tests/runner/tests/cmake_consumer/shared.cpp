// A shared library built on the generated interface: the capi staticlib is
// linked into it, and only this library's own function may be exported.
#include <fit.hpp>

extern "C" double consumer_fit_slope() {
    cxx_fit::Fit fit;
    for (int i = 0; i < 6; i++) {
        auto o = fit.obs().push();
        o.set_x(double(i));
        o.set_y(2.0 * i + 1.0);
    }
    cxx_fit::LmConfig cfg;
    cfg.max_iters = 50;
    cxx_fit::SolveResult r = fit.solve_dense(cfg);
    return r.is_err() ? -1.0 : fit.m();
}

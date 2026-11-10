// The line fit on Metal through the generated C++ interface, built with
// ARAEL_GPU=metal: the block supernodal route on the device's f32 factor,
// refined, lands where the CPU's dense solve does.
#include <fit.hpp>
#include <cmath>
#include <cstdio>

using namespace cxx_fit;

static void fill(Fit& fit) {
    for (int i = 0; i < 6; i++) {
        auto o = fit.obs().push();
        o.set_x(double(i));
        o.set_y(2.0 * i + 1.0 + (i % 2 == 0 ? 0.05 : -0.05));
    }
}

int main() {
    LmConfig cfg;
    cfg.max_iters = 50;
    Fit cpu;
    fill(cpu);
    SolveResult rc = cpu.solve_dense(cfg);
    if (rc.is_err()) {
        std::fprintf(stderr, "cpu solve failed: %s\n", rc.error().message);
        return 1;
    }

    Fit dev;
    fill(dev);
    SparseOptions so;
    so.block_supernodal = BlockSupernodalMode::Always;
    so.gpu = GpuBackend::Metal;
    so.mixed_precision = !arael::has_f64(so.gpu);   // an f64 model needs it there
    SolveResult rd = dev.solve_sparse(cfg, so);
    if (rd.is_err()) {
        std::fprintf(stderr, "metal solve failed: %s\n", rd.error().message);
        return 1;
    }
    SchurPlan p = rd->plan().value();
    std::printf("gpu_factor %d\nf32_factor %d\nend_cost %.17e\ncpu_end_cost %.17e\n",
        p.gpu_factor ? 1 : 0, p.f32_factor ? 1 : 0, rd->end_cost, rc->end_cost);
    if (!p.gpu_factor || !p.f32_factor) {
        std::fprintf(stderr, "not on the device's f32 factor\n");
        return 1;
    }
    if (std::fabs(rd->end_cost - rc->end_cost) > 1e-9 * rc->end_cost) {
        std::fprintf(stderr, "metal ends at %.17e, the cpu at %.17e\n", rd->end_cost, rc->end_cost);
        return 1;
    }
    return 0;
}

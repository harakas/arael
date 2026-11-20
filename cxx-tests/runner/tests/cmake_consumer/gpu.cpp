// The line fit on the CPU device through the generated C++ interface,
// built with ARAEL_GPU=cpu: what LmResult::gpu() reports with the device
// assembly on and off, and without a GPU.
#include <fit.hpp>
#include <cstdio>
#include <cstring>

using namespace cxx_fit;

static void fill(Fit& fit) {
    for (int i = 0; i < 6; i++) {
        auto o = fit.obs().push();
        o.set_x(double(i));
        o.set_y(2.0 * i + 1.0 + (i % 2 == 0 ? 0.05 : -0.05));
    }
}

static SolveResult solve(GpuBackend gpu, bool assembly) {
    LmConfig cfg;
    cfg.max_iters = 50;
    Fit fit;
    fill(fit);
    SparseOptions so;
    so.block_supernodal = BlockSupernodalMode::Always;
    so.gpu = gpu;
    so.gpu_assembly = assembly;
    return fit.solve_sparse(cfg, so);
}

static void show(const char* tag, const GpuReport& g) {
    std::printf("%s_backend %u\n%s_index %u\n%s_named %d\n%s_wide %u\n", tag, unsigned(g.backend), tag, g.index,
        tag, (std::strlen(g.name) > 0 && std::strlen(g.target) > 0) ? 1 : 0, tag, unsigned(g.wide));
    std::printf("%s_f32_factor %d\n%s_f32_system %d\n%s_work %u\n%s_cpu_assembly %u\n%s_why %zu\n", tag, g.f32_factor ? 1 : 0,
        tag, g.f32_system ? 1 : 0, tag, unsigned(g.work), tag, unsigned(g.cpu_assembly), tag, std::strlen(g.no_device_form));
    std::printf("%s_reserved %d\n%s_buffers %d\n", tag, g.reserved_bytes > 0 ? 1 : 0, tag, g.buffers_created > 0 ? 1 : 0);
}

int main() {
    SolveResult none = solve(GpuBackend::None, true);
    SolveResult on = solve(GpuBackend::Cpu, true);
    SolveResult off = solve(GpuBackend::Cpu, false);
    if (none.is_err() || on.is_err() || off.is_err()) {
        std::fprintf(stderr, "a solve failed: %s\n", on.is_err() ? on.error().message : "");
        return 1;
    }
    std::printf("none %d\n", none->gpu() ? 0 : 1);
    show("on", on->gpu().value());
    std::printf("on_report %d\n", std::strstr(on->report(), "  gpu       cpu:0 ") ? 1 : 0);
    show("off", off->gpu().value());
    return 0;
}

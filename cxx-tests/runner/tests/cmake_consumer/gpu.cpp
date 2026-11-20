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
    cfg.gather_timing = true;
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

// The timed solve's setup parts and the device's own time.
static void timing(const char* tag, const LmResult& r) {
    const LmTiming& t = r.timing;
    std::printf("%s_setup %d\n", tag, t.setup ? 1 : 0);
    if (t.setup) {
        const SetupTiming& s = *t.setup;
        double parts = s.blocks + s.structure + s.gpu_open + s.ordering + s.weighing + s.factor + s.gpu_plan + s.gpu_layout;
        std::printf("%s_setup_within %d\n%s_setup_gpu %d\n", tag, (parts > 0.0 && parts <= t.analysis) ? 1 : 0, tag, s.gpu_plan > 0.0 ? 1 : 0);
    }
    std::printf("%s_gpu_time %d\n", tag, t.gpu ? 1 : 0);
    if (t.gpu) {
        std::printf("%s_attempts_timed %d\n%s_assembly_timed %u\n", tag,
            (t.gpu->attempt_count == r.iterations && t.gpu->attempts > 0.0) ? 1 : 0, tag, t.gpu->assembly_count);
    }
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
    timing("none", none.value());
    show("on", on->gpu().value());
    timing("on", on.value());
    std::printf("on_report %d\n", std::strstr(on->report(), "  gpu       cpu:0 ") ? 1 : 0);
    show("off", off->gpu().value());
    return 0;
}

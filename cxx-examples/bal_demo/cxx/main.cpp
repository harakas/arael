// Bundle adjustment of a BAL ("Bundle Adjustment in the Large") problem
// over the generated arael C++ interface, on the CPU or a GPU -- the C++
// twin of examples/bal_demo.rs and the in-house test bed of the GPU
// routes through the exports. The model and the solver are Rust (model/);
// loading the file, composing the scene, choosing the route and
// reporting are C++.
//
//   ./bal_demo [path/to/problem.txt] [--gpu cuda|hip] [--device N]
//              [--f32-factor] [--no-gpu-assembly] [--repeat N]
//              [--compare] [--threads N]
//
// Every run solves the reduced (Schur) system with the block supernodal
// Cholesky, the route the GPU runs, so a CPU and a GPU run compute the
// same thing. --repeat solves the problem N times from its initial state
// (the device state and the compiled kernels carry over); --compare adds
// a CPU run and prints the relative difference of the final costs.
// VERBOSE=1 prints the solver's iteration lines. The exit code is 0 on a
// converged solve, 1 otherwise.
#include <scene.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace bal_demo;

#ifndef BAL_DEFAULT_DATASET
#define BAL_DEFAULT_DATASET "problem-49-7776-pre.txt"
#endif

/// A BAL file as read: per observation (camera, point, pixel), per camera
/// (Rodrigues rotation, translation, focal length, k1, k2), per point its
/// position.
struct Bal {
    std::vector<uint32_t> obs_cam, obs_pt;
    std::vector<vect2d> obs_xy;
    std::vector<vect3d> cam_rot, cam_t, cam_intr;
    std::vector<vect3d> pts;
};

static Bal load(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot read %s (benchmarks/bal/fetch_datasets.sh fetches the larger problems)\n", path);
        std::exit(2);
    }
    Bal b;
    size_t nc = 0, np = 0, no = 0;
    in >> nc >> np >> no;
    b.obs_cam.resize(no);
    b.obs_pt.resize(no);
    b.obs_xy.resize(no);
    for (size_t i = 0; i < no; i++) {
        in >> b.obs_cam[i] >> b.obs_pt[i] >> b.obs_xy[i].x >> b.obs_xy[i].y;
    }
    auto v3 = [&in]() {
        vect3d v;
        in >> v.x >> v.y >> v.z;
        return v;
    };
    for (size_t i = 0; i < nc; i++) {
        b.cam_rot.push_back(v3());
        b.cam_t.push_back(v3());
        b.cam_intr.push_back(v3());
    }
    for (size_t i = 0; i < np; i++) {
        b.pts.push_back(v3());
    }
    if (!in) {
        std::fprintf(stderr, "%s: truncated BAL file\n", path);
        std::exit(2);
    }
    return b;
}

/// The world-to-camera rotation of a Rodrigues vector, as Euler angles.
static vect3d euler_of_rodrigues(vect3d w) {
    double t2 = w.x * w.x + w.y * w.y + w.z * w.z;
    matrix3d m;
    if (t2 > 1e-24) {
        double th = std::sqrt(t2);
        m = matrix3d::rotation_from_axis_angle(w * (1.0 / th), th);
    } else {
        m = matrix3d::from_elements(1.0, -w.z, w.y, w.z, 1.0, -w.x, -w.y, w.x, 1.0);
    }
    return m.get_euler_angles();
}

/// The scene of `b` at its initial state.
static void compose(const Bal& b, Scene& s) {
    // Reserve before filling: a push that grows a collection moves its
    // elements, which invalidates the handles taken into it.
    s.cameras().reserve(b.cam_t.size());
    s.points().reserve(b.pts.size());
    s.observations().reserve(b.obs_xy.size());
    for (size_t i = 0; i < b.cam_t.size(); i++) {
        auto c = s.cameras().push();
        c.set_t(b.cam_t[i]);
        c.set_ea(euler_of_rodrigues(b.cam_rot[i]));
        c.set_intr(b.cam_intr[i]);
    }
    for (const auto& p : b.pts) {
        s.points().push().set_pos(p);
    }
    for (size_t i = 0; i < b.obs_xy.size(); i++) {
        auto o = s.observations().push();
        o.set_cam(s.cameras().ref_at(b.obs_cam[i]));
        o.set_pt(s.points().ref_at(b.obs_pt[i]));
        o.set_xy(b.obs_xy[i]);
    }
}

struct Run {
    bool failed = false;
    bool ok = false;
    double start = 0.0, end = 0.0, ms = 0.0;
    uint32_t iters = 0, accepted = 0;
    bool reduced = false, gpu_factor = false, gpu_assembly = false, f32_factor = false;
};

static Run solve(const Bal& b, const LmConfig& cfg, const SparseOptions& so) {
    Scene s;
    compose(b, s);
    auto t0 = std::chrono::steady_clock::now();
    SolveResult r = s.solve_sparse(cfg, so);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    Run run;
    run.ms = ms;
    if (r.is_err()) {
        std::fprintf(stderr, "solve failed: %s\n", r.error().message);
        run.failed = true;
        return run;
    }
    const LmResult& v = r.value();
    run.ok = v.status == LmStatus::Converged;
    run.start = v.start_cost;
    run.end = v.end_cost;
    run.iters = v.iterations;
    run.accepted = v.accepted_iterations;
    if (v.plan().has_value()) {
        SchurPlan p = v.plan().value();
        run.reduced = p.reduced;
        run.gpu_factor = p.gpu_factor;
        run.gpu_assembly = p.gpu_assembly;
        run.f32_factor = p.f32_factor;
    }
    return run;
}

static void report(const char* label, const Run& r, size_t n_obs) {
    if (r.failed) {
        std::printf("%-5s failed\n", label);
        return;
    }
    auto rms = [n_obs](double c) { return std::sqrt(c / double(n_obs)); };
    std::printf("%-5s %9.1f ms  %2u iterations (%2u accepted)  cost %.4f -> %.4f  RMS %.4f -> %.4f px  [%s%s%s%s]\n",
                label, r.ms, r.iters, r.accepted, r.start, r.end, rms(r.start), rms(r.end),
                r.reduced ? "reduced" : "whole system",
                r.gpu_factor ? ", GPU factor" : "",
                r.gpu_assembly ? ", GPU assembly and loop" : "",
                r.f32_factor ? ", f32 factor" : "");
}

int main(int argc, char** argv) {
    const char* path = BAL_DEFAULT_DATASET;
    GpuBackend gpu = GpuBackend::None;
    uint32_t device = 0, repeat = 1, threads = 0;
    bool f32_factor = false, gpu_assembly = true, compare = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gpu") {
            std::string g = next();
            if (g == "cuda") {
                gpu = GpuBackend::Cuda;
            } else if (g == "hip") {
                gpu = GpuBackend::Hip;
            } else {
                std::fprintf(stderr, "--gpu takes cuda or hip\n");
                return 2;
            }
        } else if (a == "--device") {
            device = uint32_t(std::atoi(next()));
        } else if (a == "--f32-factor") {
            f32_factor = true;
        } else if (a == "--no-gpu-assembly") {
            gpu_assembly = false;
        } else if (a == "--repeat") {
            repeat = uint32_t(std::atoi(next()));
        } else if (a == "--compare") {
            compare = true;
        } else if (a == "--threads") {
            threads = uint32_t(std::atoi(next()));
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else {
            path = argv[i];
        }
    }
    Bal b = load(path);
    std::printf("BAL problem: %zu cameras, %zu points, %zu observations\n", b.cam_t.size(), b.pts.size(), b.obs_xy.size());

    // The gain-ratio driver of the ill-conditioned preset, tightened for
    // a short run, as in the Rust example.
    LmConfig cfg = LmConfig::ill_conditioned();
    cfg.max_iters = 30;
    cfg.initial_lambda = 1e-4;
    cfg.abs_precision = 1e-5;
    cfg.rel_precision = 1e-5;
    cfg.patience = 1;
    cfg.verbose = std::getenv("VERBOSE") != nullptr;
    if (threads > 0) {
        cfg.num_threads = threads;
    }

    // The reduced system, factorized by the block supernodal Cholesky.
    SparseOptions so;
    so.schur = SchurPolicy::Force;
    so.envelope = EnvelopeMode::Never;
    so.block_supernodal = BlockSupernodalMode::Always;
    so.mixed_precision = f32_factor;
    so.gpu = gpu;
    so.gpu_index = device;
    so.gpu_assembly = gpu_assembly;
    const char* label = gpu == GpuBackend::None ? "cpu" : "gpu";

    bool ok = true;
    Run last;
    for (uint32_t k = 0; k < repeat; k++) {
        last = solve(b, cfg, so);
        report(label, last, b.obs_xy.size());
        ok = ok && last.ok;
    }
    if (compare) {
        SparseOptions cpu = so;
        cpu.gpu = GpuBackend::None;
        Run c = solve(b, cfg, cpu);
        report("cpu", c, b.obs_xy.size());
        ok = ok && c.ok;
        std::printf("final cost relative difference: %.2e\n", std::fabs(last.end - c.end) / c.end);
    }
    return ok ? 0 : 1;
}

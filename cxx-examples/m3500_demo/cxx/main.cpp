// M3500 2D pose-graph optimization over the generated arael C++
// interface -- the C++ twin of examples/m3500_demo.rs. The model and
// solver are Rust (model/); loading the g2o file, composing the graph,
// and reporting are C++.
//
// Reads a g2o file with VERTEX_SE2 / EDGE_SE2 entries and solves the
// classic pose-graph problem: 3500 poses (x, y, theta), ~5450 relative
// SE2 measurements, gauge fixed by a soft prior on pose 0. Unit
// weights by default (--weighted uses the dataset's sqrt-info).
//
//   ./m3500_demo [path/to/file.g2o] [--weighted] [--dump out.txt]
//   VERBOSE=1 for solver iteration lines.
#include <graph.hpp>
#include <arael/g2o.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace m3500_demo;

#ifndef M3500_DEFAULT_DATASET
#define M3500_DEFAULT_DATASET "input_M3500_g2o.g2o"
#endif

static double rad_diff(double a, double b) {
    double d = a - b;
    while (d > M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

static void load_g2o(const char* path, bool weighted, Graph& graph) {
    arael::g2o::Dataset2 ds = arael::g2o::Dataset2::load(path);
    // Reserve before filling: a push that grows the collection moves its
    // elements, which invalidates any handle already taken into it.
    graph.poses().reserve(ds.poses.size());
    graph.edges().reserve(ds.deltas.size());
    for (const auto& p : ds.poses) {
        auto pose = graph.poses().push();
        pose.set_pos(p.t);
        pose.set_rot_angle(p.th);
        // The first pose anchors the gauge.
        if (!graph.has_prior()) {
            auto prior = graph.make_prior();
            prior.set_p(graph.poses().ref_at(0));
            prior.set_pos(p.t);
            prior.set_th(p.th);
        }
    }
    for (const auto& d : ds.deltas) {
        // M3500 is diagonal with I11 == I22; sqrt-info weighting then
        // reduces to two per-edge row scales.
        double wt = 1.0, wr = 1.0;
        if (weighted)
            arael_assert_true(d.iso_sqrt_info(wt, wr));
        auto e = graph.edges().push();
        e.set_a(graph.poses().ref_at(d.a));
        e.set_b(graph.poses().ref_at(d.b));
        e.set_delta(d.dt);
        e.set_dth(d.dth);
        e.set_wt(wt);
        e.set_wr(wr);
    }
}

// Reference metrics computed directly from the data, independent of
// the generated solver code: plain least-squares cost and the
// Huber(1.0) block metric other minimal solvers report.
static void metrics(Graph& graph, double& ls, double& huber) {
    ls = 0.0;
    huber = 0.0;
    auto block = [&](double r0, double r1, double r2) {
        double s = r0 * r0 + r1 * r1 + r2 * r2;
        ls += s;
        huber += s > 1.0 ? 2.0 * std::sqrt(s) - 1.0 : s;
    };
    for (auto e : graph.edges()) {
        auto a = graph.poses().get(e.a());
        auto b = graph.poses().get(e.b());
        double sa = std::sin(a.rot_angle()), ca = std::cos(a.rot_angle());
        double sb = std::sin(b.rot_angle()), cb = std::cos(b.rot_angle());
        vect2d delta = e.delta();
        double gx = a.pos().x + ca * delta.x - sa * delta.y - b.pos().x;
        double gy = a.pos().y + sa * delta.x + ca * delta.y - b.pos().y;
        block((cb * gx + sb * gy) * e.wt(),
              (-sb * gx + cb * gy) * e.wt(),
              rad_diff(a.rot_angle() + e.dth(), b.rot_angle()) * e.wr());
    }
    if (graph.has_prior()) {
        auto prior = graph.prior().value();
        auto p = graph.poses().get(prior.p());
        block(p.pos().x - prior.pos().x,
              p.pos().y - prior.pos().y,
              p.rot_angle() - prior.th());
    }
}

// Minimal EPS scatter of pose positions (before = light gray, after =
// black), raw PostScript with no dependencies.
static void write_eps(const std::vector<vect2d>& before,
                      const std::vector<vect2d>& after, const char* out) {
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    for (const auto* pts : {&before, &after}) {
        for (const auto& p : *pts) {
            xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        }
    }
    const double size = 500.0;
    double scale = size / std::max(xmax - xmin, ymax - ymin);
    std::FILE* f = std::fopen(out, "w");
    arael_assert_true(f != nullptr);
    std::fprintf(f, "%%!PS-Adobe-3.0 EPSF-3.0\n");
    std::fprintf(f, "%%%%BoundingBox: 0 0 %d %d\n", int(size) + 20, int(size) + 20);
    const std::vector<vect2d>* sets[2] = {&before, &after};
    const double grays[2] = {0.75, 0.0};
    for (int k = 0; k < 2; k++) {
        std::fprintf(f, "%g setgray\n", grays[k]);
        for (const auto& p : *sets[k])
            std::fprintf(f, "%.1f %.1f 1.2 0 360 arc fill\n",
                10.0 + (p.x - xmin) * scale, 10.0 + (p.y - ymin) * scale);
    }
    std::fprintf(f, "showpage\n");
    std::fclose(f);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    bool weighted = false;
    const char* path = M3500_DEFAULT_DATASET;
    const char* dump = nullptr;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--weighted") == 0) weighted = true;
        else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else if (argv[i][0] != '-') path = argv[i];
    }

    Graph graph;
    load_g2o(path, weighted, graph);
    if (weighted) std::printf("using information-matrix (sqrt-info) weighting\n");
    std::printf("%s: %u poses, %u edges\n", path, graph.poses().size(), graph.edges().size());

    double ls0, huber0;
    metrics(graph, ls0, huber0);
    std::printf("initial cost: LS=%.6f huber=%.6f\n", ls0, huber0);
    std::vector<vect2d> before;
    for (auto p : graph.poses()) before.push_back(p.pos());

    std::printf("parameters: %u\n", graph.poses().size() * Pose2::param_count);

    LmConfig cfg = LmConfig::well_conditioned();
    cfg.verbose = std::getenv("VERBOSE") != nullptr;
    auto start = std::chrono::steady_clock::now();
    SolveResult r = graph.solve_sparse(cfg);
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    if (r.is_err()) {
        std::fprintf(stderr, "solve failed: %s\n", r.error().message);
        return 1;
    }

    double ls1, huber1;
    metrics(graph, ls1, huber1);
    std::printf("%u iterations, cost %.6f -> %.6f\n",
        r->iterations, r->start_cost, r->end_cost);
    std::printf("final cost:   LS=%.6f huber=%.6f\n", ls1, huber1);
    std::printf("solve time: %.3fs\n", elapsed.count());

    std::vector<vect2d> after;
    for (auto p : graph.poses()) after.push_back(p.pos());
    const char* out = weighted ? "m3500_weighted.eps" : "m3500.eps";
    write_eps(before, after, out);
    std::printf("wrote %s\n", out);

    if (dump) {
        std::FILE* f = std::fopen(dump, "w");
        arael_assert_true(f != nullptr);
        for (auto p : graph.poses())
            std::fprintf(f, "%.17g %.17g %.17g\n", p.pos().x, p.pos().y, p.rot_angle());
        std::fclose(f);
        std::printf("dumped poses to %s\n", dump);
    }

    const char* labels[3] = {"x0", "x1", "x3499"};
    const uint32_t idxs[3] = {0, 1, 3499};
    for (int k = 0; k < 3; k++) {
        if (idxs[k] < graph.poses().size()) {
            auto p = graph.poses()[idxs[k]];
            std::printf("%s: theta=%.6f x=%.6f y=%.6f\n",
                labels[k], p.rot_angle(), p.pos().x, p.pos().y);
        }
    }
    return 0;
}

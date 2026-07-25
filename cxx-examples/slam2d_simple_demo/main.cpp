// 2D SLAM from C++ over the generated arael interface -- the C++ twin
// of examples/slam2d_simple_demo.rs. The model and solver are Rust
// (model/); this file does everything else: synthesize the world,
// compose the problem, solve, report errors against ground truth, and
// plot the map to an EPS file.
//
// The robot drives an arc; odometry drifts; a forward camera reports
// bearings to building corners. The first pose is held fixed at the
// origin (set_*_optimize(false)) so the map cannot slide or rotate.
//
// Build and run:
//   cmake -S . -B build && cmake --build build && ./build/slam2d_simple_demo

#include <path.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace arael;

struct Cfg {
    int n_poses = 20;
    int n_landmarks = 30;
    unsigned seed = 42;
    float step = 1.5f;
    float turn = 0.10f;
    float fov_half = 60.0f * float(M_PI) / 180.0f;
    float range_min = 4.0f;
    float range_max = 50.0f;
    float odo_pos_sigma = 0.05f;
    float odo_gamma_sigma = 0.3f * float(M_PI) / 180.0f;
    float bearing_sigma = 1.0f * float(M_PI) / 180.0f;
    float init_range = 20.0f;
};

static float wrap_angle(float a) { return std::atan2(std::sin(a), std::cos(a)); }

// Gentle left-turning arc starting at the origin facing east.
static std::vector<std::pair<vect2f, float>> truth_poses(const Cfg& cfg) {
    std::vector<std::pair<vect2f, float>> out;
    vect2f pos{0, 0};
    float gamma = 0;
    out.push_back({pos, gamma});
    for (int i = 1; i < cfg.n_poses; i++) {
        pos = pos + vect2f{cfg.step * std::cos(gamma), cfg.step * std::sin(gamma)};
        gamma += cfg.turn;
        out.push_back({pos, gamma});
    }
    return out;
}

// Corners scattered around the trajectory, at observable distance.
static std::vector<vect2f> truth_landmarks(
    const Cfg& cfg, std::mt19937& rng,
    const std::vector<std::pair<vect2f, float>>& poses)
{
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> pick(0, poses.size() - 1);
    std::vector<vect2f> out;
    while (int(out.size()) < cfg.n_landmarks) {
        vect2f anchor = poses[pick(rng)].first;
        float theta = uni(rng) * 2.0f * float(M_PI);
        float r = cfg.range_min + uni(rng) * (cfg.range_max - cfg.range_min);
        vect2f lm = anchor + vect2f{r * std::cos(theta), r * std::sin(theta)};
        bool visible = false;
        for (auto& [p, g] : poses) {
            float d = (lm - p).norm();
            if (d >= cfg.range_min && d <= cfg.range_max) { visible = true; break; }
        }
        if (visible) out.push_back(lm);
    }
    return out;
}

// Bearing from (pos, gamma) to lm, with FOV / range gating; NAN = unseen.
static float observe(const Cfg& cfg, vect2f pos, float gamma, vect2f lm) {
    vect2f d = lm - pos;
    float dist = d.norm();
    if (dist < cfg.range_min || dist > cfg.range_max) return NAN;
    vect2f local = matrix2f::rotation(gamma).transpose() * d;
    float bearing = std::atan2(local.y, local.x);
    if (std::fabs(bearing) > cfg.fov_half) return NAN;
    return bearing;
}

// ---------------------------------------------------------------------------
// EPS plot: ground truth vs estimate.
// ---------------------------------------------------------------------------
static void write_eps(
    const char* file,
    const std::vector<std::pair<vect2f, float>>& gt_poses,
    const std::vector<vect2f>& gt_lms,
    const std::vector<vect2f>& est_poses,
    const std::vector<vect2f>& est_lms,
    const std::vector<int>& lm_to_gt)
{
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    auto grow = [&](vect2f p) {
        x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    };
    for (auto& [p, g] : gt_poses) grow(p);
    for (auto& p : gt_lms) grow(p);
    for (auto& p : est_lms) grow(p);
    float pad = 0.05f * std::max(x1 - x0, y1 - y0);
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
    const float w = 500.0f;
    float s = w / std::max(x1 - x0, y1 - y0);
    float h = (y1 - y0) * s;
    auto X = [&](vect2f p) { return (p.x - x0) * s; };
    auto Y = [&](vect2f p) { return (p.y - y0) * s; };

    FILE* f = std::fopen(file, "w");
    if (!f) { std::perror(file); return; }
    std::fprintf(f, "%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: 0 0 %d %d\n",
        int(w) + 1, int(h) + 21);
    std::fprintf(f, "/l { lineto } def /m { moveto } def\n");

    auto polyline = [&](const std::vector<vect2f>& pts) {
        for (size_t i = 0; i < pts.size(); i++)
            std::fprintf(f, "%.1f %.1f %s\n", X(pts[i]), Y(pts[i]), i ? "l" : "m");
        std::fprintf(f, "stroke\n");
    };
    std::vector<vect2f> gtp, estp;
    for (auto& [p, g] : gt_poses) gtp.push_back(p);
    estp = est_poses;

    // Estimated landmark -> its ground truth, as faint links.
    std::fprintf(f, "0.85 setgray 0.5 setlinewidth\n");
    for (size_t i = 0; i < est_lms.size(); i++) {
        vect2f a = est_lms[i], b = gt_lms[lm_to_gt[i]];
        std::fprintf(f, "%.1f %.1f m %.1f %.1f l stroke\n", X(a), Y(a), X(b), Y(b));
    }
    // Ground-truth path (gray) and landmarks (crosses).
    std::fprintf(f, "0.6 setgray 1 setlinewidth\n");
    polyline(gtp);
    for (auto& p : gt_lms)
        std::fprintf(f, "%.1f %.1f m -3 -3 rmoveto 6 6 rlineto -6 0 rmoveto 6 -6 rlineto stroke\n",
            X(p), Y(p));
    // Estimated path (blue) and landmarks (red dots).
    std::fprintf(f, "0 0 0.8 setrgbcolor 1.5 setlinewidth\n");
    polyline(estp);
    std::fprintf(f, "0.8 0 0 setrgbcolor\n");
    for (auto& p : est_lms)
        std::fprintf(f, "%.1f %.1f 2.5 0 360 arc fill\n", X(p), Y(p));
    std::fprintf(f, "0 setgray /Helvetica findfont 10 scalefont setfont\n");
    std::fprintf(f, "4 %.1f m (gray: ground truth   blue: solved path   red: solved corners) show\n",
        h + 8.0f);
    std::fprintf(f, "showpage\n");
    std::fclose(f);
}

int main() {
    Cfg cfg;
    std::mt19937 rng(cfg.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    auto gt_poses = truth_poses(cfg);
    auto gt_lms = truth_landmarks(cfg, rng, gt_poses);

    Path path;

    // Dead-reckoned initial estimates from noisy odometry.
    vect2f est_pos{0, 0};
    float est_gamma = 0;

    for (size_t pi = 0; pi < gt_poses.size(); pi++) {
        auto pose = path.poses().push_back();
        if (pi == 0) {
            // Hold the first pose fixed: every measurement is relative,
            // so without an anchor the whole map could slide or rotate.
            pose.set_pos_optimize(false);
            pose.set_gamma_optimize(false);
            continue;
        }
        auto [gt_p, gt_g] = gt_poses[pi];
        auto [prev_p, prev_g] = gt_poses[pi - 1];
        vect2f true_delta = matrix2f::rotation(prev_g).transpose() * (gt_p - prev_p);
        float true_dg = gt_g - prev_g;
        vect2f noisy_delta = true_delta
            + vect2f{cfg.odo_pos_sigma * nd(rng), cfg.odo_pos_sigma * nd(rng)};
        float noisy_dg = true_dg + cfg.odo_gamma_sigma * nd(rng);

        est_pos = est_pos + matrix2f::rotation(est_gamma) * noisy_delta;
        est_gamma += noisy_dg;

        pose.set_pos(est_pos);
        pose.set_gamma(est_gamma);
        pose.set_delta_pos(noisy_delta);
        pose.set_delta_gamma(noisy_dg);
        pose.set_delta_pos_isigma(1.0f / cfg.odo_pos_sigma);
        pose.set_delta_gamma_isigma(1.0f / cfg.odo_gamma_sigma);

        auto pair = path.pose_pairs().push();
        pair.set_prev(path.poses().ref_at(uint32_t(pi - 1)));
        pair.set_cur(path.poses().ref_at(uint32_t(pi)));
    }

    // Landmarks with at least two sightings; init on the first ray.
    std::vector<Ref_Landmark> lm_refs;
    std::vector<int> lm_to_gt;
    int n_frines = 0;
    for (size_t li = 0; li < gt_lms.size(); li++) {
        std::vector<std::pair<size_t, float>> sightings;
        for (size_t pi = 0; pi < gt_poses.size(); pi++) {
            float b = observe(cfg, gt_poses[pi].first, gt_poses[pi].second, gt_lms[li]);
            if (std::isnan(b)) continue;
            sightings.push_back({pi, b + cfg.bearing_sigma * nd(rng)});
        }
        if (sightings.size() < 2) continue;   // two rays to triangulate

        Ref_Landmark r = path.landmarks().push();
        auto lm = path.landmarks().get(r);
        auto [first_pi, first_b] = sightings[0];
        auto p0 = path.poses()[uint32_t(first_pi)];
        float world_b = p0.gamma() + first_b;
        lm.set_pos(p0.pos() + vect2f{cfg.init_range * std::cos(world_b),
                                     cfg.init_range * std::sin(world_b)});
        for (auto [pi, b] : sightings) {
            auto fr = lm.frines().push();
            fr.set_pose(path.poses().ref_at(uint32_t(pi)));
            fr.set_bearing(b);
            fr.set_isigma(1.0f / cfg.bearing_sigma);
            n_frines++;
        }
        lm_refs.push_back(r);
        lm_to_gt.push_back(int(li));
    }

    std::printf("Path: %u poses, %u pose_pairs, %u landmarks, %d frines\n",
        path.poses().size(), path.pose_pairs().size(), path.landmarks().size(), n_frines);

    LmConfig cfg_lm = LmConfig::well_conditioned();
    SolveResult r = path.solve_sparse(cfg_lm);
    if (r.is_err()) {
        std::fprintf(stderr, "solve failed: %s\n", r.error().message);
        return 1;
    }
    std::printf("Solved: status %d, cost %.3f -> %.3f in %u iterations\n",
        int(r->status), double(r->start_cost), double(r->end_cost), r->iterations);

    std::printf("\n-- Pose errors vs GT --\n");
    std::vector<vect2f> est_poses;
    float pos_sum = 0, g_sum = 0;
    for (size_t i = 0; i < gt_poses.size(); i++) {
        vect2f p = path.poses()[uint32_t(i)].pos();
        float g = path.poses()[uint32_t(i)].gamma();
        est_poses.push_back(p);
        float pe = (p - gt_poses[i].first).norm();
        float ge = std::fabs(wrap_angle(g - gt_poses[i].second));
        std::printf("  pose %2zu: |dp|=%.3fm  |dgamma|=%.3fdeg\n",
            i, double(pe), double(ge) * 180.0 / M_PI);
        pos_sum += pe;
        g_sum += ge;
    }
    std::printf("  mean: pos=%.4fm  gamma=%.3fdeg\n",
        double(pos_sum) / gt_poses.size(),
        double(g_sum) / gt_poses.size() * 180.0 / M_PI);

    std::printf("\n-- Landmark errors vs GT --\n");
    std::vector<vect2f> est_lms;
    float lm_sum = 0;
    for (size_t i = 0; i < lm_refs.size(); i++) {
        auto lm = path.landmarks().get(lm_refs[i]);
        vect2f p = lm.pos();
        est_lms.push_back(p);
        float e = (p - gt_lms[lm_to_gt[i]]).norm();
        std::printf("  lm %2zu: |d|=%.3fm  frines=%u\n", i, double(e), lm.frines().size());
        lm_sum += e;
    }
    if (!lm_refs.empty())
        std::printf("  mean: |d|=%.4fm\n", double(lm_sum) / lm_refs.size());

    const char* out = "slam2d_simple_cxx.eps";
    write_eps(out, gt_poses, gt_lms, est_poses, est_lms, lm_to_gt);
    std::printf("\nMap plotted to %s\n", out);
    return 0;
}

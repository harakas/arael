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

using namespace slam2d_simple;

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

// Per-landmark 95% confidence ellipse from the 2x2 position covariance:
// chi^2(0.95, df=2) = 5.991, semi-axes are sqrt(5.991 * eigenvalue).
struct Ellipse {
    vect2f center;
    float semi_major, semi_minor, angle;
};

static Ellipse ellipse_from_cov(vect2f center, matrix2d c) {
    auto [r, d] = c.symmetric_eigen(); // ascending eigenvalues
    const double chi2_95 = 5.991;
    vect2d major = r.col(1);
    return {center,
            float(std::sqrt(std::max(d.y, 0.0) * chi2_95)),
            float(std::sqrt(std::max(d.x, 0.0) * chi2_95)),
            float(std::atan2(major.y, major.x))};
}

// ---------------------------------------------------------------------------
// EPS plot -- the same layout as the Rust example's:
//   * ground truth poses (gray dashed chain + gray triangles) first;
//   * bearing rays from each optimized pose, tinted per landmark;
//   * optimized pose chain (dashed) + dark filled triangles along gamma;
//   * per-landmark 95% ellipses in the landmark's own hue;
//   * landmark error lines + GT landmark dots above the ray bundles;
//   * optimized landmarks as hued dots.
// ---------------------------------------------------------------------------

static void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    float h6 = (h - std::floor(h)) * 6.0f;
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h6, 2.0f) - 1.0f));
    switch (int(h6)) {
        case 0: r = c; g = x; b = 0; break;
        case 1: r = x; g = c; b = 0; break;
        case 2: r = 0; g = c; b = x; break;
        case 3: r = 0; g = x; b = c; break;
        case 4: r = x; g = 0; b = c; break;
        default: r = c; g = 0; b = x; break;
    }
    float m = v - c;
    r += m; g += m; b += m;
}

// Evenly-spaced hues; rays get a washed-out tint of the landmark's hue.
static void landmark_color(size_t i, size_t n, bool ray, float& r, float& g, float& b) {
    float h = n == 0 ? 0.0f : float(i) / float(n);
    hsv_to_rgb(h, ray ? 0.40f : 0.85f, ray ? 0.97f : 0.78f, r, g, b);
}

struct Sighting { size_t pose; float bearing; };

static void write_eps(
    const char* file,
    const std::vector<std::pair<vect2f, float>>& gt_poses,
    const std::vector<vect2f>& gt_lms,
    const std::vector<std::pair<vect2f, float>>& est_poses,
    const std::vector<vect2f>& est_lms,
    const std::vector<std::vector<Sighting>>& lm_sightings,
    const std::vector<int>& lm_to_gt,
    const std::vector<Ellipse>& ellipses)
{
    // Bounding box across everything we plan to draw.
    float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
    auto grow = [&](vect2f p) {
        xmin = std::min(xmin, p.x); ymin = std::min(ymin, p.y);
        xmax = std::max(xmax, p.x); ymax = std::max(ymax, p.y);
    };
    for (auto& [p, g] : est_poses) grow(p);
    for (auto& p : est_lms) grow(p);
    for (auto& [p, g] : gt_poses) grow(p);
    for (int gi : lm_to_gt) grow(gt_lms[gi]);
    xmin -= 3; xmax += 3; ymin -= 3; ymax += 3;

    const float page_w = 540.0f, page_h = 420.0f, pad = 18.0f;
    float s = std::min((page_w - 2 * pad) / (xmax - xmin),
                       (page_h - 2 * pad) / (ymax - ymin));
    float dx = (page_w - s * (xmax - xmin)) * 0.5f;
    float dy = (page_h - s * (ymax - ymin)) * 0.5f;
    auto X = [&](vect2f p) { return dx + (p.x - xmin) * s; };
    auto Y = [&](vect2f p) { return dy + (p.y - ymin) * s; };

    FILE* f = std::fopen(file, "w");
    if (!f) { std::perror(file); return; }
    std::fprintf(f, "%%!PS-Adobe-3.0 EPSF-3.0\n");
    std::fprintf(f, "%%%%BoundingBox: 0 0 %d %d\n", int(page_w), int(page_h));
    std::fprintf(f, "%%%%Creator: slam2d_simple_demo (C++)\n%%%%EndComments\n");
    // Triangle marker `x y angle_deg size tri`: forward tip at (size, 0).
    std::fprintf(f, "/tri { gsave 4 2 roll translate exch rotate "
        "dup 0 moveto "
        "dup -0.55 mul 1 index 0.45 mul lineto "
        "dup -0.55 mul exch -0.45 mul lineto "
        "closepath fill grestore } def\n");
    std::fprintf(f, "/dot { newpath 0 360 arc fill } def\n");

    auto polyline = [&](const std::vector<std::pair<vect2f, float>>& pts) {
        std::fprintf(f, "newpath ");
        for (size_t i = 0; i < pts.size(); i++)
            std::fprintf(f, "%.2f %.2f %s ", X(pts[i].first), Y(pts[i].first),
                i ? "lineto" : "moveto");
        std::fprintf(f, "stroke\n");
    };

    // Ground-truth pose shadow (behind everything).
    std::fprintf(f, "0.62 0.62 0.62 setrgbcolor 0.8 setlinewidth [3 2] 0 setdash\n");
    polyline(gt_poses);
    std::fprintf(f, "[] 0 setdash\n");
    for (auto& [p, g] : gt_poses)
        std::fprintf(f, "%.2f %.2f %.2f 8 tri\n", X(p), Y(p),
            g * 180.0f / float(M_PI));

    // Bearing rays from each optimized pose, world frame, 110%% of the
    // pose->landmark distance so each ray reaches its landmark.
    std::fprintf(f, "0.25 setlinewidth\n");
    size_t n_lm = est_lms.size();
    for (size_t li = 0; li < n_lm; li++) {
        float r, g, b;
        landmark_color(li, n_lm, true, r, g, b);
        std::fprintf(f, "%.3f %.3f %.3f setrgbcolor\n", r, g, b);
        for (auto& sight : lm_sightings[li]) {
            vect2f pp = est_poses[sight.pose].first;
            float world_dir = est_poses[sight.pose].second + sight.bearing;
            float dist = (est_lms[li] - pp).norm() * 1.10f;
            vect2f tip = pp + vect2f{dist * std::cos(world_dir),
                                     dist * std::sin(world_dir)};
            std::fprintf(f, "newpath %.2f %.2f moveto %.2f %.2f lineto stroke\n",
                X(pp), Y(pp), X(tip), Y(tip));
        }
    }

    // Optimized pose chain (dashed) + filled triangles along gamma.
    std::fprintf(f, "0.08 0.15 0.30 setrgbcolor 1.0 setlinewidth [4 2] 0 setdash\n");
    polyline(est_poses);
    std::fprintf(f, "[] 0 setdash 0.10 0.18 0.40 setrgbcolor\n");
    for (auto& [p, g] : est_poses)
        std::fprintf(f, "%.2f %.2f %.2f 6.5 tri\n", X(p), Y(p),
            g * 180.0f / float(M_PI));

    // 95%% confidence ellipses, each in its landmark's hue.
    std::fprintf(f, "0.6 setlinewidth\n");
    for (size_t i = 0; i < ellipses.size(); i++) {
        auto& e = ellipses[i];
        if (e.semi_major <= 0 || e.semi_minor <= 0) continue;
        float r, g, b;
        landmark_color(i, n_lm, false, r, g, b);
        std::fprintf(f, "%.3f %.3f %.3f setrgbcolor\nnewpath ", r, g, b);
        const int segs = 48;
        float ct = std::cos(e.angle), st = std::sin(e.angle);
        for (int j = 0; j <= segs; j++) {
            float phi = 2.0f * float(M_PI) * float(j) / segs;
            float lx = e.semi_major * std::cos(phi);
            float ly = e.semi_minor * std::sin(phi);
            vect2f w{e.center.x + ct * lx - st * ly,
                     e.center.y + st * lx + ct * ly};
            std::fprintf(f, "%.2f %.2f %s ", X(w), Y(w), j ? "lineto" : "moveto");
        }
        std::fprintf(f, "closepath stroke\n");
    }

    // Landmark error lines + GT landmark dots (above the ray bundles).
    std::fprintf(f, "0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth\n");
    for (size_t i = 0; i < n_lm; i++) {
        vect2f gt = gt_lms[lm_to_gt[i]];
        std::fprintf(f, "newpath %.2f %.2f moveto %.2f %.2f lineto stroke\n",
            X(est_lms[i]), Y(est_lms[i]), X(gt), Y(gt));
    }
    for (int gi : lm_to_gt)
        std::fprintf(f, "%.2f %.2f 2.2 dot\n", X(gt_lms[gi]), Y(gt_lms[gi]));

    // Optimized landmarks, one hue per landmark.
    for (size_t i = 0; i < n_lm; i++) {
        float r, g, b;
        landmark_color(i, n_lm, false, r, g, b);
        std::fprintf(f, "%.3f %.3f %.3f setrgbcolor %.2f %.2f 2.8 dot\n",
            r, g, b, X(est_lms[i]), Y(est_lms[i]));
    }

    std::fprintf(f, "%%%%EOF\n");
    std::fclose(f);
}

int main() {
    // Line-buffer stdout so our lines interleave correctly with the
    // solver's verbose output (the Rust side flushes per line).
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Cfg cfg;
    std::mt19937 rng(cfg.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    auto gt_poses = truth_poses(cfg);
    auto gt_lms = truth_landmarks(cfg, rng, gt_poses);

    Path path;

    // Dead-reckoned initial estimates from noisy odometry.
    vect2f est_pos{0, 0};
    float est_gamma = 0;

    // Reserve before filling: a push that grows the collection moves its
    // elements, which invalidates any handle already taken into it.
    path.poses().reserve(gt_poses.size());
    path.pose_pairs().reserve(gt_poses.size());
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
    std::vector<LandmarkRef> lm_refs;
    std::vector<int> lm_to_gt;
    std::vector<std::vector<Sighting>> lm_sightings;
    int n_frines = 0;
    path.landmarks().reserve(gt_lms.size());
    for (size_t li = 0; li < gt_lms.size(); li++) {
        std::vector<Sighting> sightings;
        for (size_t pi = 0; pi < gt_poses.size(); pi++) {
            float b = observe(cfg, gt_poses[pi].first, gt_poses[pi].second, gt_lms[li]);
            if (std::isnan(b)) continue;
            sightings.push_back({pi, b + cfg.bearing_sigma * nd(rng)});
        }
        if (sightings.size() < 2) continue;   // two rays to triangulate

        LandmarkRef r = path.landmarks().push();
        auto lm = path.landmarks().get(r);
        lm.frines().reserve(sightings.size());
        size_t first_pi = sightings[0].pose;
        float first_b = sightings[0].bearing;
        auto p0 = path.poses()[uint32_t(first_pi)];
        float world_b = p0.gamma() + first_b;
        lm.set_pos(p0.pos() + vect2f{cfg.init_range * std::cos(world_b),
                                     cfg.init_range * std::sin(world_b)});
        for (auto& sight : sightings) {
            auto fr = lm.frines().push();
            fr.set_pose(path.poses().ref_at(uint32_t(sight.pose)));
            fr.set_bearing(sight.bearing);
            fr.set_isigma(1.0f / cfg.bearing_sigma);
            n_frines++;
        }
        lm_refs.push_back(r);
        lm_to_gt.push_back(int(li));
        lm_sightings.push_back(std::move(sightings));
    }

    // Range-for works on every container view (arena walks live slots).
    int frines_in_model = 0;
    for (auto lm : path.landmarks())
        frines_in_model += int(lm.frines().size());
    std::printf("Path: %u poses, %u pose_pairs, %u landmarks, %d frines (%d wired)\n",
        path.poses().size(), path.pose_pairs().size(), path.landmarks().size(),
        n_frines, frines_in_model);

    // gather_timing fills the result's timing block, which the report
    // below breaks down per phase.
    LmConfig cfg_lm = LmConfig::well_conditioned();
    cfg_lm.verbose = true;
    cfg_lm.gather_timing = true;
    SolveResult r = path.solve_sparse(cfg_lm);
    if (r.is_err()) {
        std::fprintf(stderr, "solve failed: %s\n", r.error().message);
        return 1;
    }
    // The result prints itself: status, cost, where the time went --
    // rendered by the Rust side from the full solve result.
    // last_report() is the same text in plain ASCII, for a log.
    std::printf("\n%s\n", path.last_pretty_report());

    std::printf("\n-- Pose errors vs GT --\n");
    std::vector<std::pair<vect2f, float>> est_poses;
    float pos_sum = 0, g_sum = 0;
    for (size_t i = 0; i < gt_poses.size(); i++) {
        vect2f p = path.poses()[uint32_t(i)].pos();
        float g = path.poses()[uint32_t(i)].gamma();
        est_poses.push_back({p, g});
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

    // Per-landmark uncertainty from the parameter covariance. The first
    // pose is held fixed, so the Hessian is invertible; each landmark's
    // 2x2 block is its own positional uncertainty.
    std::vector<Ellipse> ellipses;
    auto cov = path.assemble_covariance(CovMode::AllMarginals);
    if (cov.is_err()) {
        std::fprintf(stderr, "covariance: %s -- skipping uncertainty\n",
            cov.error().message);
    } else {
        for (size_t i = 0; i < lm_refs.size(); i++) {
            auto lm = path.landmarks().get(lm_refs[i]);
            auto m = cov->marginal(lm);
            if (m.is_err()) continue;
            ellipses.push_back(ellipse_from_cov(lm.pos(), m.value()));
        }
        std::printf("\n%zu landmark uncertainty ellipses (95%%)\n", ellipses.size());
    }

    const char* out = "slam2d_simple_cxx.eps";
    write_eps(out, gt_poses, gt_lms, est_poses, est_lms, lm_sightings, lm_to_gt, ellipses);
    std::printf("\nMap plotted to %s\n", out);
    return 0;
}

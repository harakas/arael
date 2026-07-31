// Synthetic visual-inertial SLAM demo over the generated arael C++
// interface -- the C++ twin of examples/slam_demo_gm.rs. The model and
// solver are Rust (model/); composing the problem, the graduated
// optimization ramp, and the error reports are C++. See the Rust
// example for the full model walkthrough.
//
// An S-curve trajectory (default 60 poses, 240 point landmarks) at
// 5-30m distance, 5 cameras with 360-degree coverage, GPS + wheel
// odometry + accelerometer tilt, 50% outlier feature associations at
// 30x pixel noise. Feature and GPS residuals robustified by a
// Geman-McClure block loss; landmarks use anchored inverse-depth.
#include <path.hpp>
#include <arael/geometry.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace slam_demo_gm;
using arael::camerad;

// ---------------------------------------------------------------------------
// Random numbers (mt19937_64; the numbers differ from the Rust
// example's StdRng -- same shape, same behavior)
// ---------------------------------------------------------------------------

struct Rng {
    std::mt19937_64 mt;
    std::uniform_real_distribution<double> unif{0.0, 1.0};
    std::normal_distribution<double> norm{0.0, 1.0};
    explicit Rng(uint64_t seed) : mt(seed) {}
    double uniform() { return unif(mt); }
    double normal() { return double(norm(mt)); }
    size_t index(size_t n) {
        return std::uniform_int_distribution<size_t>(0, n - 1)(mt);
    }
};

static std::vector<camerad> create_cameras() {
    // 5 cameras at 72-degree intervals around the robot, looking
    // toward the horizon.
    std::vector<camerad> cameras;
    const uint32_t w = 1024, h = 768;
    const double fov_deg = 80.0;
    const double fx = (double(w) / 2.0) / std::tan(fov_deg / 2.0 * M_PI / 180.0);
    const double fy = fx;
    const int n = 5;
    for (int i = 0; i < n; i++) {
        double yaw = double(i) * (360.0 / n) * M_PI / 180.0;
        double sy = std::sin(yaw), cy_ = std::cos(yaw);
        // Camera Z looks outward; image Y looks down.
        matrix3d mc2r = matrix3d::from_cols(
            {-sy, cy_, 0.0}, {0.0, 0.0, -1.0}, {cy_, sy, 0.0});
        cameras.push_back(camerad{fx, fy, double(w) / 2.0, double(h) / 2.0, w, h,
            {cy_ * 0.1, sy * 0.1, 0.3}, mc2r});
    }
    return cameras;
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

struct SceneConfig {
    size_t num_poses = 60;
    size_t num_landmarks = 240;
    uint64_t seed = 42;
    double outlier_fraction = 0.5; // fraction of invalid associations
    double outlier_scale = 30.0;   // outlier pixel noise is 30x normal
    double s_amplitude = 1.5;      // S-curve lateral amplitude (meters)
    double s_frequency = 0.8;      // S-curve angular frequency
    double step_size = 0.25;       // distance between poses (meters)
    double gps_sigma = 0.3;        // GPS per-fix noise sigma (meters)
    double gps_sigma_inflate = 2.0; // constraint models an inflated GPS
                                    // covariance so it does not compete
                                    // with odometry/features locally
    double odo_pos_k = 0.10;       // position noise fraction of distance
    double odo_pos_base = 0.03;    // base position noise (meters)
    double odo_ea_k = 0.01;
    double odo_ea_base = 0.001;
    size_t lm_visibility_range = 15;
    double lm_visibility_prob = 0.75;
};

// Covariance split into R and 1/sqrt(d), the form the constraints
// consume.
static void decompose_cov(matrix3d cov, matrix3d& r, vect3d& isigma) {
    auto [rr, d] = cov.symmetric_eigen();
    r = rr;
    isigma = {1.0 / std::sqrt(d.x), 1.0 / std::sqrt(d.y), 1.0 / std::sqrt(d.z)};
}

static matrix3d diagonal_cov(vect3d sigma) {
    return matrix3d::from_elements(
        sigma.x * sigma.x, 0.0, 0.0,
        0.0, sigma.y * sigma.y, 0.0,
        0.0, 0.0, sigma.z * sigma.z);
}

struct GtPose { vect3d pos; vect3d ea; };

static std::vector<GtPose> generate_ground_truth_poses(const SceneConfig& cfg) {
    std::vector<GtPose> poses;
    double t = 0.0;
    for (size_t i = 0; i < cfg.num_poses; i++) {
        double x = t;
        double y = cfg.s_amplitude * std::sin(cfg.s_frequency * t);
        // Yaw follows the tangent direction.
        double dy = cfg.s_amplitude * cfg.s_frequency * std::cos(cfg.s_frequency * t);
        poses.push_back({{x, y, 0.0}, {0.0, 0.0, std::atan2(dy, 1.0)}});
        t += cfg.step_size;
    }
    return poses;
}

struct GtLandmark { vect3d pos; size_t anchor_idx; };

static std::vector<GtLandmark> generate_ground_truth_landmarks(
    const SceneConfig& cfg, Rng& rng, const std::vector<GtPose>& poses) {
    std::vector<GtLandmark> landmarks;
    for (size_t i = 0; i < cfg.num_landmarks; i++) {
        for (;;) {
            size_t anchor_idx = rng.index(poses.size());
            vect3d anchor = poses[anchor_idx].pos;
            double angle = rng.uniform() * 2.0 * double(M_PI);
            double dist = 5.0 + rng.uniform() * 25.0;
            vect3d lm{anchor.x + dist * std::cos(angle),
                      anchor.y + dist * std::sin(angle),
                      rng.uniform() * 2.0};
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& p : poses)
                min_dist = std::min(min_dist, (lm - p.pos).norm());
            if (min_dist >= 5.0 && min_dist <= 30.0) {
                landmarks.push_back({lm, anchor_idx});
                break;
            }
        }
    }
    return landmarks;
}

// Composes the problem into `path`. Returns the ground-truth index of
// each pushed landmark (landmarks nobody observes are skipped).
static std::vector<size_t> build_path(const SceneConfig& cfg, Rng& rng, Path& path,
    const std::vector<GtPose>& gt_poses, const std::vector<GtLandmark>& gt_landmarks) {
    auto cameras = create_cameras();

    // Weak prior on each landmark's inverse range (1/m units).
    const double drift_rho_sigma = 1.0;
    const double tilt_sigma_deg = 0.25; // accelerometer accuracy
    const double tilt_sigma_rad = tilt_sigma_deg * double(M_PI) / 180.0;

    path.set_drift_rho_isigma(1.0 / drift_rho_sigma);
    path.set_tilt_isigma(1.0 / tilt_sigma_rad);
    path.set_frine_isigma_scale(1.0);
    path.set_frine_c2(2.99);
    path.set_frine_cauchy(-1.0);
    path.set_gps_c2(7.815);

    // (landmark index, observing-pose index, feature ref); pose refs
    // resolve after all poses are built.
    struct FrineData { size_t li; size_t pi; PointFeatureRef feature; };
    std::vector<FrineData> frine_data;

    // Reserve before filling: a push that grows the collection moves its
    // elements, which invalidates any handle already taken into it.
    path.poses().reserve(gt_poses.size());
    for (size_t pi = 0; pi < gt_poses.size(); pi++) {
        vect3d pos = gt_poses[pi].pos;
        vect3d ea = gt_poses[pi].ea;
        matrix3d mr2w = matrix3d::rotation_from_euler_angles(ea);

        // Odometry deltas from ground truth.
        vect3d delta_pos{0.0, 0.0, 0.0};
        matrix3d delta_rot = matrix3d::identity();
        if (pi > 0) {
            matrix3d prev_mw2r =
                matrix3d::rotation_from_euler_angles(gt_poses[pi - 1].ea).transpose();
            delta_pos = prev_mw2r * (pos - gt_poses[pi - 1].pos);
            delta_rot = prev_mw2r * mr2w;
        }

        // Odometry covariance proportional to motion; rotation angle
        // from the trace identity cos(theta) = (tr - 1) / 2.
        double dp_norm = std::max(delta_pos.norm(), 0.01);
        double tr = delta_rot[0].x + delta_rot[1].y + delta_rot[2].z;
        double de_norm = std::max(std::acos(std::min(std::max((tr - 1.0) * 0.5, -1.0), 1.0)), 0.001);
        double ps = cfg.odo_pos_k * dp_norm + cfg.odo_pos_base;
        vect3d pos_sigma{ps, ps * 0.5, ps * 0.5}; // lateral less noisy
        double rs = cfg.odo_ea_k * de_norm + cfg.odo_ea_base;
        vect3d rot_sigma{rs, rs, rs};

        auto pose = path.poses().push_back();
        auto info = pose.info();

        // Features for landmarks visible from this pose.
        for (size_t li = 0; li < gt_landmarks.size(); li++) {
            vect3d lm_pos = gt_landmarks[li].pos;
            size_t anchor_idx = gt_landmarks[li].anchor_idx;
            size_t dist_to_anchor = pi >= anchor_idx ? pi - anchor_idx : anchor_idx - pi;
            if (dist_to_anchor > cfg.lm_visibility_range) continue;
            if (rng.uniform() > cfg.lm_visibility_prob) continue;
            for (const auto& cam : cameras) {
                vect3d p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if (p_cam.z < 0.5) continue; // behind camera or too close
                vect2d pixel = cam.project(p_cam);
                if (!cam.is_visible(pixel)) continue;

                // Pixel noise (uniform +-1 pixel, outliers scaled up).
                bool is_outlier = rng.uniform() < cfg.outlier_fraction;
                double noise_scale = is_outlier ? cfg.outlier_scale : 1.0;
                vect2d noisy_pixel{
                    pixel.x + noise_scale * (rng.uniform() * 2.0 - 1.0),
                    pixel.y + noise_scale * (rng.uniform() * 2.0 - 1.0)};

                // Feature-to-robot frame: col0 = view direction, col1/2 =
                // perpendicular axes for the angular error components.
                vect3d dir = cam.unproject_to_robot(noisy_pixel);
                vect3d cam_up = -(cam.mc2r.col(1));
                vect3d up_proj = cam_up - dir * (cam_up * dir);
                double up_norm = up_proj.norm();
                if (up_norm < 1e-6) continue;
                vect3d col2 = up_proj * (1.0 / up_norm);
                vect3d col1 = col2 % dir;

                vect2d sigma = cam.pixel_angular_size(noisy_pixel);

                auto feat = info.features().push();
                feat.set_pixel(noisy_pixel);
                feat.set_mf2r(matrix3d::from_cols(dir, col1, col2));
                feat.set_camera_pos(cam.camera_pos);
                feat.set_isigma({1.0 / sigma.x, 1.0 / sigma.y});
                frine_data.push_back({li, pi, info.features().last_ref()});
            }
        }

        // GPS: iid per-fix noise; the constraint covariance is inflated.
        vect3d gps_pos{pos.x + cfg.gps_sigma * rng.normal(),
                       pos.y + cfg.gps_sigma * rng.normal(),
                       pos.z + cfg.gps_sigma * rng.normal()};
        double ms = cfg.gps_sigma * cfg.gps_sigma_inflate;

        // Noisy initial pose estimate.
        const double init_noise_pos = 0.1;  // meters
        const double init_noise_ea = 0.02;  // radians
        vect3d noisy_pos{pos.x + init_noise_pos * rng.normal(),
                         pos.y + init_noise_pos * rng.normal(),
                         pos.z + init_noise_pos * rng.normal()};
        vect3d noisy_ea{ea.x + init_noise_ea * rng.normal(),
                        ea.y + init_noise_ea * rng.normal(),
                        ea.z + init_noise_ea * rng.normal()};

        pose.set_r2w_translation(noisy_pos);
        pose.set_r2w_rotation(quaternd::from_euler_angles(noisy_ea));

        info.set_delta_pos(delta_pos);
        info.set_delta_rot(delta_rot);
        matrix3d cov_r; vect3d cov_isigma;
        decompose_cov(diagonal_cov(pos_sigma), cov_r, cov_isigma);
        info.set_delta_pos_cov_r(cov_r);
        info.set_delta_pos_cov_isigma(cov_isigma);
        decompose_cov(diagonal_cov(rot_sigma), cov_r, cov_isigma);
        info.set_delta_rot_cov_r(cov_r);
        info.set_delta_rot_cov_isigma(cov_isigma);

        auto gps = info.make_gps();
        gps.set_pos(gps_pos);
        decompose_cov(diagonal_cov({ms, ms, ms}), cov_r, cov_isigma);
        gps.set_cov_r(cov_r);
        gps.set_cov_isigma(cov_isigma);

        // Tilt: noise lives in angle space (roll/pitch); the reading is
        // the up direction it implies (row 2 of R).
        double r = ea.x + tilt_sigma_rad * rng.normal();
        double p = ea.y + tilt_sigma_rad * rng.normal();
        info.set_tilt_g({-std::sin(p), std::cos(p) * std::sin(r),
                         std::cos(p) * std::cos(r)});
    }

    // Landmarks with frines: the anchor is the middlest observing pose,
    // snapshotted at its initial position; direction and inverse range
    // initialize from the noisy landmark guess.
    std::vector<size_t> kept_gt;
    path.landmarks().reserve(gt_landmarks.size());
    for (size_t li = 0; li < gt_landmarks.size(); li++) {
        vect3d lm_pos = gt_landmarks[li].pos;
        vect3d noisy_lm{lm_pos.x + 0.5 * rng.normal(),
                        lm_pos.y + 0.5 * rng.normal(),
                        lm_pos.z + 0.3 * rng.normal()};
        std::vector<const FrineData*> obs;
        for (const auto& fd : frine_data)
            if (fd.li == li) obs.push_back(&fd);
        if (obs.empty()) continue; // skip landmarks with no observations
        kept_gt.push_back(li);

        PoseRef anchor_pose = path.poses().ref_at(uint32_t(obs[obs.size() / 2]->pi));
        vect3d anchor = path.poses().get(anchor_pose).r2w_translation();
        vect3d d = noisy_lm - anchor;

        auto lm = path.landmarks().get(path.landmarks().push());
        lm.set_anchor(anchor);
        lm.set_anchor_pose(anchor_pose);
        lm.set_dir_unit(d * (1.0 / d.norm()));
        lm.set_rho(1.0 / d.norm());
        lm.frines().reserve(obs.size());
        for (const auto* fd : obs) {
            auto fr = lm.frines().push();
            fr.set_pose(path.poses().ref_at(uint32_t(fd->pi)));
            fr.set_feature(fd->feature);
        }
    }

    // Pose pairs for odometry.
    for (uint32_t i = 1; i < path.poses().size(); i++) {
        auto pp = path.pose_pairs().push();
        pp.set_prev(path.poses().ref_at(i - 1));
        pp.set_cur(path.poses().ref_at(i));
    }
    return kept_gt;
}

struct Stats { double mean, median, min, max; };
static Stats stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    return {sum / double(v.size()), v[v.size() / 2], v.front(), v.back()};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_usage() {
    std::fprintf(stderr, "Usage: slam_demo_gm [OPTIONS]\n");
    std::fprintf(stderr, "  --solver <dense|sparse>  (default: sparse)\n");
    std::fprintf(stderr, "  --loss <gm|cauchy>       (default: gm)\n");
    std::fprintf(stderr, "  --poses <N>              (default: 60)\n");
    std::fprintf(stderr, "  --landmarks <N>          (default: 240)\n");
    std::fprintf(stderr, "  --seed <N>               (default: 42)\n");
}

int main(int argc, char** argv) {
    // Line-buffer stdout so our lines interleave correctly with the
    // solver's verbose output (the Rust side flushes per line).
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string solver_name = "sparse";
    std::string loss_name = "gm";
    SceneConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--solver") solver_name = next();
        else if (a == "--loss") loss_name = next();
        else if (a == "--poses") cfg.num_poses = std::strtoull(next(), nullptr, 10);
        else if (a == "--landmarks") cfg.num_landmarks = std::strtoull(next(), nullptr, 10);
        else if (a == "--seed") cfg.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage();
            return 1;
        }
    }
    if (solver_name == "faer") solver_name = "sparse"; // faer is the sparse backend
    if (solver_name != "dense" && solver_name != "sparse") {
        std::fprintf(stderr, "Unknown solver: %s. Available: dense, sparse\n",
            solver_name.c_str());
        return 1;
    }

    std::printf("Solver: %s  Loss: %s  Poses: %zu  Landmarks: %zu  Seed: %llu\n",
        solver_name.c_str(), loss_name.c_str(), cfg.num_poses, cfg.num_landmarks,
        (unsigned long long)cfg.seed);

    Rng rng(cfg.seed);
    auto gt_poses = generate_ground_truth_poses(cfg);
    auto gt_landmarks = generate_ground_truth_landmarks(cfg, rng, gt_poses);

    Path path;
    auto kept_gt = build_path(cfg, rng, path, gt_poses, gt_landmarks);

    // Each family's measured-best threshold.
    if (loss_name == "gm") {
        path.set_frine_cauchy(-1.0);
        path.set_frine_c2(2.99);
    } else if (loss_name == "cauchy") {
        path.set_frine_cauchy(1.0);
        path.set_frine_c2(1.5);
    } else {
        std::fprintf(stderr, "Unknown loss: %s. Available: gm, cauchy\n", loss_name.c_str());
        return 1;
    }

    size_t n_frines = 0;
    for (auto lm : path.landmarks()) n_frines += lm.frines().size();
    std::printf("Path: %u poses, %u landmarks, %zu frines, %u pose_pairs\n",
        path.poses().size(), path.landmarks().size(), n_frines, path.pose_pairs().size());
    std::printf("Parameters: %u (Pose=%u, Landmark=%u)\n\n",
        path.poses().size() * Pose::param_count
            + path.landmarks().size() * PointLandmark::param_count,
        Pose::param_count, PointLandmark::param_count);

    // Print a few poses.
    for (size_t i : {size_t(0), cfg.num_poses / 2, cfg.num_poses - 1}) {
        if (i >= path.poses().size()) continue;
        auto pose = path.poses()[uint32_t(i)];
        vect3d t = pose.r2w_translation();
        vect3d ea = pose.r2w_rotation().get_euler_angles();
        std::printf("Pose %2zu: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)\n",
            i, t.x, t.y, t.z, ea.x, ea.y, ea.z);
        std::printf("      gt: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)\n",
            gt_poses[i].pos.x, gt_poses[i].pos.y, gt_poses[i].pos.z,
            gt_poses[i].ea.x, gt_poses[i].ea.y, gt_poses[i].ea.z);
    }
    std::printf("\n");

    // Graduated optimization: start with loose feature constraints,
    // tighten. Landmark anchors re-snapshot between passes -- values
    // only, so one LmSession carries every pass and the sparsity
    // analysis is reused warm, like the Rust example.
    std::printf("--- Optimization ---\n");
    std::vector<double> isigma_scales =
        std::getenv("SINGLE_PASS") ? std::vector<double>{1.0}
                                   : std::vector<double>{0.01, 0.1, 1.0};
    LmSession session;
    for (size_t pass = 0; pass < isigma_scales.size(); pass++) {
        path.set_frine_isigma_scale(isigma_scales[pass]);
        std::printf("\nPass %zu (isigma scale=%g):\n", pass + 1, isigma_scales[pass]);
        LmConfig lm_cfg = LmConfig::well_conditioned();
        lm_cfg.rel_precision = 1e-6;
        lm_cfg.verbose = true;
        SolveResult r = solver_name == "dense" ? path.solve_dense(lm_cfg)
                                               : session.solve(path, lm_cfg);
        if (r.is_err()) {
            std::fprintf(stderr, "solve failed: %s\n", r.error().message);
            return 1;
        }
        std::printf("  %u iterations, cost %.4f -> %.4f\n",
            r->iterations, r->start_cost, r->end_cost);

        // Move each landmark's anchor to its anchor pose's CURRENT
        // position and re-express direction + inverse range there, so
        // the anchor stays near the rays as the poses converge.
        // Landmarks at (near) infinity keep their ray.
        if (pass + 1 < isigma_scales.size()) {
            for (auto lm : path.landmarks()) {
                if (std::abs(lm.rho()) < 1e-4) continue;
                vect3d world = lm.anchor() + lm.dir_unit() * (1.0 / lm.rho());
                vect3d c_new = path.poses().get(lm.anchor_pose()).r2w_translation();
                vect3d d = world - c_new;
                double n = d.norm();
                if (n < 1e-3) continue;
                lm.set_anchor(c_new);
                lm.set_dir_unit(d * (1.0 / n));
                lm.set_rho(1.0 / n);
            }
        }
    }

    // Mean absolute pose error vs GT.
    {
        double pos_err_sum = 0.0, ea_err_sum = 0.0;
        size_t n = std::min(gt_poses.size(), size_t(path.poses().size()));
        for (size_t i = 0; i < n; i++) {
            auto pose = path.poses()[uint32_t(i)];
            pos_err_sum += (pose.r2w_translation() - gt_poses[i].pos).norm();
            ea_err_sum += (pose.r2w_rotation().get_euler_angles() - gt_poses[i].ea).norm();
        }
        std::printf("\nFinal cost: %.4f\n", path.cost());
        std::printf("Mean pose error vs GT: pos=%.4fm  ea=%.3fdeg\n",
            pos_err_sum / double(n), ea_err_sum / double(n) * 180.0 / double(M_PI));
    }

    // Relative pose errors: consecutive deltas in the local frame.
    std::printf("\n--- Relative pose errors ---\n");
    std::vector<double> dpos_errs, dpos_rel_errs, dea_errs_deg, dea_rel_errs;
    for (size_t i = 1; i < std::min(gt_poses.size(), size_t(path.poses().size())); i++) {
        auto prev = path.poses()[uint32_t(i - 1)];
        auto pose = path.poses()[uint32_t(i)];

        matrix3d gt_mr2w = matrix3d::rotation_from_euler_angles(gt_poses[i - 1].ea);
        vect3d gt_delta_pos = gt_mr2w.transpose() * (gt_poses[i].pos - gt_poses[i - 1].pos);

        matrix3d opt_mr2w_prev = prev.r2w_rotation().rotation_matrix();
        vect3d opt_delta_pos = opt_mr2w_prev.transpose()
            * (pose.r2w_translation() - prev.r2w_translation());

        double dpos_err = (opt_delta_pos - gt_delta_pos).norm();
        double gt_step = gt_delta_pos.norm();
        double dpos_rel = gt_step > 1e-6 ? 100.0 * dpos_err / gt_step : 0.0;

        matrix3d gt_mr2w_cur = matrix3d::rotation_from_euler_angles(gt_poses[i].ea);
        vect3d gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles();

        matrix3d opt_mr2w_cur = pose.r2w_rotation().rotation_matrix();
        vect3d opt_delta_ea = (opt_mr2w_prev.transpose() * opt_mr2w_cur).get_euler_angles();

        double dea_err = (opt_delta_ea - gt_delta_ea).norm();
        double dea_err_deg = dea_err * 180.0 / double(M_PI);
        double gt_rot = gt_delta_ea.norm();
        double dea_rel = gt_rot > 1e-6 ? 100.0 * dea_err / gt_rot : 0.0;

        std::printf("Pair %2zu-%2zu: dpos=%.4fm (%.1f%%)  dea=%.3fdeg (%.1f%%)\n",
            i - 1, i, dpos_err, dpos_rel, dea_err_deg, dea_rel);
        dpos_errs.push_back(dpos_err);
        dpos_rel_errs.push_back(dpos_rel);
        dea_errs_deg.push_back(dea_err_deg);
        dea_rel_errs.push_back(dea_rel);
    }
    if (!dpos_errs.empty()) {
        Stats s = stats(dpos_errs);
        std::printf("Delta pos: mean=%.4fm  median=%.4fm  min=%.4fm  max=%.4fm\n",
            s.mean, s.median, s.min, s.max);
        s = stats(dpos_rel_errs);
        std::printf("Delta pos: mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%\n",
            s.mean, s.median, s.min, s.max);
        s = stats(dea_errs_deg);
        std::printf("Delta ea:  mean=%.3fdeg  median=%.3fdeg  min=%.3fdeg  max=%.3fdeg\n",
            s.mean, s.median, s.min, s.max);
        s = stats(dea_rel_errs);
        std::printf("Delta ea:  mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%\n",
            s.mean, s.median, s.min, s.max);
    }

    // Landmark uncertainty from the parameter covariance. The relative
    // covariance C_ll + C_pp - C_lp - C_pl over the landmark and pose
    // POSITION blocks cancels the shared gauge uncertainty; ellipsoid
    // semi-axes are the sqrt of its eigenvalues.
    auto cov_r = path.assemble_covariance(CovMode::AllMarginals);
    if (cov_r.is_err())
        std::printf("Covariance unavailable: %s\n", cov_r.error().message);

    // Landmark errors: landmark-to-closest-pose vector, opt vs GT.
    std::printf("\n--- Landmark errors (relative to closest pose) ---\n");
    std::vector<double> lm_errs, lm_rel_errs;
    std::vector<double> max_sigmas;
    size_t li_out = 0;
    for (auto lm : path.landmarks()) {
        vect3d gt_lm = gt_landmarks[kept_gt[li_out]].pos;

        // Closest GT pose.
        size_t closest_idx = 0;
        double best = std::numeric_limits<double>::max();
        for (size_t j = 0; j < gt_poses.size(); j++) {
            double d = (gt_lm - gt_poses[j].pos).norm();
            if (d < best) { best = d; closest_idx = j; }
        }
        matrix3d gt_mr2w = matrix3d::rotation_from_euler_angles(gt_poses[closest_idx].ea);
        vect3d gt_vec = gt_mr2w.transpose() * (gt_lm - gt_poses[closest_idx].pos);

        auto opt_pose = path.poses()[uint32_t(closest_idx)];
        matrix3d opt_mr2w = opt_pose.r2w_rotation().rotation_matrix();
        vect3d lm_world = lm.anchor() + lm.dir_unit() * (1.0 / lm.rho());
        vect3d opt_vec = opt_mr2w.transpose() * (lm_world - opt_pose.r2w_translation());
        double err = (opt_vec - gt_vec).norm();
        double gt_dist = gt_vec.norm();
        double rel_pct = 100.0 * err / gt_dist;

        // The landmark marginal is [dir chart (2); rho]; map it to
        // world position covariance with J = [unit_d / rho, -unit /
        // rho^2] (the anchor is constant data). The pose marginal is
        // [w (rotation); d (translation)], d in the reference rotation
        // frame: the world translation block is R * C_dd * R^T.
        // Near-infinity landmarks (rho ~ 0) and failed queries print
        // without a sigma.
        bool have_sigma = false;
        double sg[3] = {0, 0, 0};
        if (cov_r.is_ok() && std::abs(lm.rho()) >= 1e-4) {
            auto& cov = cov_r.value();
            double rho = lm.rho();
            vect3d u = lm.dir_unit();
            vect3d ud0 = lm.dir_unit_d0(), ud1 = lm.dir_unit_d1();
            matrix3d j = matrix3d::from_elements(
                ud0.x / rho, ud1.x / rho, -double(u.x) / (rho * rho),
                ud0.y / rho, ud1.y / rho, -double(u.y) / (rho * rho),
                ud0.z / rho, ud1.z / rho, -double(u.z) / (rho * rho));
            auto marg = cov.marginal(lm);
            double pose_buf[36], cross_buf[18];
            if (marg.is_ok()
                && cov.marginal(opt_pose, pose_buf, 36) == 6
                && cov.cross(lm, opt_pose, cross_buf, 18) == 3) {
                matrix3d c_ll = j * marg.value() * j.transpose();
                matrix3d r = opt_mr2w.cast<double>();
                // The pose marginal is 6x6 [w; d]: take the d block.
                matrix3d c_dd = matrix3d::from_elements(
                    pose_buf[21], pose_buf[22], pose_buf[23],
                    pose_buf[27], pose_buf[28], pose_buf[29],
                    pose_buf[33], pose_buf[34], pose_buf[35]);
                matrix3d c_pp = r * c_dd * r.transpose();
                // The 3x6 landmark x pose cross: its d columns.
                matrix3d x = matrix3d::from_elements(
                    cross_buf[3], cross_buf[4], cross_buf[5],
                    cross_buf[9], cross_buf[10], cross_buf[11],
                    cross_buf[15], cross_buf[16], cross_buf[17]);
                matrix3d c_lp = j * x * r.transpose();
                matrix3d cov_rel = c_ll + c_pp - c_lp - c_lp.transpose();
                auto [evec, eval] = cov_rel.symmetric_eigen();
                (void)evec;
                sg[0] = std::sqrt(std::max(eval.z, 0.0)); // ascending -> desc
                sg[1] = std::sqrt(std::max(eval.y, 0.0));
                sg[2] = std::sqrt(std::max(eval.x, 0.0));
                have_sigma = true;
            }
        }
        if (have_sigma) {
            std::printf("LM %3zu: |d|=%.3fm  rel=%.2f%%  dist=%.1fm  sigma=(%.3f,%.3f,%.3f)m  frines=%u\n",
                li_out, err, rel_pct, gt_dist, sg[0], sg[1], sg[2], lm.frines().size());
            max_sigmas.push_back(double(sg[0]));
        } else {
            std::printf("LM %3zu: |d|=%.3fm  rel=%.2f%%  dist=%.1fm  frines=%u\n",
                li_out, err, rel_pct, gt_dist, lm.frines().size());
        }
        lm_errs.push_back(err);
        lm_rel_errs.push_back(rel_pct);
        li_out++;
    }
    if (!lm_errs.empty()) {
        Stats s = stats(lm_errs);
        std::printf("LM pos:  mean=%.3fm  median=%.3fm  min=%.3fm  max=%.3fm\n",
            s.mean, s.median, s.min, s.max);
        s = stats(lm_rel_errs);
        std::printf("LM rel:  mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%\n",
            s.mean, s.median, s.min, s.max);
    }
    if (!max_sigmas.empty()) {
        Stats s = stats(max_sigmas);
        std::printf("Max principal sigma: mean=%.3fm  median=%.3fm  min=%.3fm  max=%.3fm\n",
            s.mean, s.median, s.min, s.max);
    }
    return 0;
}

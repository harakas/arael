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
using arael::Camera;

// ---------------------------------------------------------------------------
// Random numbers (mt19937_64; the numbers differ from the Rust
// example's StdRng -- same shape, same behavior)
// ---------------------------------------------------------------------------

struct Rng {
    std::mt19937_64 mt;
    std::uniform_real_distribution<float> unif{0.0f, 1.0f};
    std::normal_distribution<double> norm{0.0, 1.0};
    explicit Rng(uint64_t seed) : mt(seed) {}
    float uniform() { return unif(mt); }
    float normal() { return float(norm(mt)); }
    size_t index(size_t n) {
        return std::uniform_int_distribution<size_t>(0, n - 1)(mt);
    }
};

static std::vector<Camera> create_cameras() {
    // 5 cameras at 72-degree intervals around the robot, looking
    // toward the horizon.
    std::vector<Camera> cameras;
    const uint32_t w = 1024, h = 768;
    const float fov_deg = 80.0f;
    const float fx = (float(w) / 2.0f) / std::tan(fov_deg / 2.0f * float(M_PI) / 180.0f);
    const float fy = fx;
    const int n = 5;
    for (int i = 0; i < n; i++) {
        float yaw = float(i) * (360.0f / n) * float(M_PI) / 180.0f;
        float sy = std::sin(yaw), cy_ = std::cos(yaw);
        // Camera Z looks outward; image Y looks down.
        matrix3f mc2r = matrix3f::from_cols(
            {-sy, cy_, 0.0f}, {0.0f, 0.0f, -1.0f}, {cy_, sy, 0.0f});
        cameras.push_back(Camera{fx, fy, float(w) / 2.0f, float(h) / 2.0f, w, h,
            {cy_ * 0.1f, sy * 0.1f, 0.3f}, mc2r});
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
    float outlier_fraction = 0.5f; // fraction of invalid associations
    float outlier_scale = 30.0f;   // outlier pixel noise is 30x normal
    float s_amplitude = 1.5f;      // S-curve lateral amplitude (meters)
    float s_frequency = 0.8f;      // S-curve angular frequency
    float step_size = 0.25f;       // distance between poses (meters)
    float gps_sigma = 0.3f;        // GPS per-fix noise sigma (meters)
    float gps_sigma_inflate = 2.0f; // constraint models an inflated GPS
                                    // covariance so it does not compete
                                    // with odometry/features locally
    float odo_pos_k = 0.10f;       // position noise fraction of distance
    float odo_pos_base = 0.03f;    // base position noise (meters)
    float odo_ea_k = 0.01f;
    float odo_ea_base = 0.001f;
    size_t lm_visibility_range = 15;
    float lm_visibility_prob = 0.75f;
};

// Covariance split into R and 1/sqrt(d), the form the constraints
// consume.
static void decompose_cov(matrix3f cov, matrix3f& r, vect3f& isigma) {
    auto [rr, d] = cov.symmetric_eigen();
    r = rr;
    isigma = {1.0f / std::sqrt(d.x), 1.0f / std::sqrt(d.y), 1.0f / std::sqrt(d.z)};
}

static matrix3f diagonal_cov(vect3f sigma) {
    return matrix3f::from_elements(
        sigma.x * sigma.x, 0.0f, 0.0f,
        0.0f, sigma.y * sigma.y, 0.0f,
        0.0f, 0.0f, sigma.z * sigma.z);
}

struct GtPose { vect3f pos; vect3f ea; };

static std::vector<GtPose> generate_ground_truth_poses(const SceneConfig& cfg) {
    std::vector<GtPose> poses;
    float t = 0.0f;
    for (size_t i = 0; i < cfg.num_poses; i++) {
        float x = t;
        float y = cfg.s_amplitude * std::sin(cfg.s_frequency * t);
        // Yaw follows the tangent direction.
        float dy = cfg.s_amplitude * cfg.s_frequency * std::cos(cfg.s_frequency * t);
        poses.push_back({{x, y, 0.0f}, {0.0f, 0.0f, std::atan2(dy, 1.0f)}});
        t += cfg.step_size;
    }
    return poses;
}

struct GtLandmark { vect3f pos; size_t anchor_idx; };

static std::vector<GtLandmark> generate_ground_truth_landmarks(
    const SceneConfig& cfg, Rng& rng, const std::vector<GtPose>& poses) {
    std::vector<GtLandmark> landmarks;
    for (size_t i = 0; i < cfg.num_landmarks; i++) {
        for (;;) {
            size_t anchor_idx = rng.index(poses.size());
            vect3f anchor = poses[anchor_idx].pos;
            float angle = rng.uniform() * 2.0f * float(M_PI);
            float dist = 5.0f + rng.uniform() * 25.0f;
            vect3f lm{anchor.x + dist * std::cos(angle),
                      anchor.y + dist * std::sin(angle),
                      rng.uniform() * 2.0f};
            float min_dist = std::numeric_limits<float>::max();
            for (const auto& p : poses)
                min_dist = std::min(min_dist, (lm - p.pos).norm());
            if (min_dist >= 5.0f && min_dist <= 30.0f) {
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
    const float drift_rho_sigma = 1.0f;
    const float tilt_sigma_deg = 0.25f; // accelerometer accuracy
    const float tilt_sigma_rad = tilt_sigma_deg * float(M_PI) / 180.0f;

    path.set_drift_rho_isigma(1.0f / drift_rho_sigma);
    path.set_tilt_isigma(1.0f / tilt_sigma_rad);
    path.set_frine_isigma_scale(1.0f);
    path.set_frine_c2(2.99f);
    path.set_frine_cauchy(-1.0f);
    path.set_gps_c2(7.815f);

    // (landmark index, observing-pose index, feature ref); pose refs
    // resolve after all poses are built.
    struct FrineData { size_t li; size_t pi; PointFeatureRef feature; };
    std::vector<FrineData> frine_data;

    // Reserve before filling: a push that grows the collection moves its
    // elements, which invalidates any handle already taken into it.
    path.poses().reserve(gt_poses.size());
    for (size_t pi = 0; pi < gt_poses.size(); pi++) {
        vect3f pos = gt_poses[pi].pos;
        vect3f ea = gt_poses[pi].ea;
        matrix3f mr2w = matrix3f::rotation_from_euler_angles(ea);

        // Odometry deltas from ground truth.
        vect3f delta_pos{0.0f, 0.0f, 0.0f};
        matrix3f delta_rot = matrix3f::identity();
        if (pi > 0) {
            matrix3f prev_mw2r =
                matrix3f::rotation_from_euler_angles(gt_poses[pi - 1].ea).transpose();
            delta_pos = prev_mw2r * (pos - gt_poses[pi - 1].pos);
            delta_rot = prev_mw2r * mr2w;
        }

        // Odometry covariance proportional to motion; rotation angle
        // from the trace identity cos(theta) = (tr - 1) / 2.
        float dp_norm = std::max(delta_pos.norm(), 0.01f);
        float tr = delta_rot[0].x + delta_rot[1].y + delta_rot[2].z;
        float de_norm = std::max(std::acos(std::min(std::max((tr - 1.0f) * 0.5f, -1.0f), 1.0f)), 0.001f);
        float ps = cfg.odo_pos_k * dp_norm + cfg.odo_pos_base;
        vect3f pos_sigma{ps, ps * 0.5f, ps * 0.5f}; // lateral less noisy
        float rs = cfg.odo_ea_k * de_norm + cfg.odo_ea_base;
        vect3f rot_sigma{rs, rs, rs};

        auto pose = path.poses().push_back();
        auto info = pose.info();

        // Features for landmarks visible from this pose.
        for (size_t li = 0; li < gt_landmarks.size(); li++) {
            vect3f lm_pos = gt_landmarks[li].pos;
            size_t anchor_idx = gt_landmarks[li].anchor_idx;
            size_t dist_to_anchor = pi >= anchor_idx ? pi - anchor_idx : anchor_idx - pi;
            if (dist_to_anchor > cfg.lm_visibility_range) continue;
            if (rng.uniform() > cfg.lm_visibility_prob) continue;
            for (const auto& cam : cameras) {
                vect3f p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if (p_cam.z < 0.5f) continue; // behind camera or too close
                vect2f pixel = cam.project(p_cam);
                if (!cam.is_visible(pixel)) continue;

                // Pixel noise (uniform +-1 pixel, outliers scaled up).
                bool is_outlier = rng.uniform() < cfg.outlier_fraction;
                float noise_scale = is_outlier ? cfg.outlier_scale : 1.0f;
                vect2f noisy_pixel{
                    pixel.x + noise_scale * (rng.uniform() * 2.0f - 1.0f),
                    pixel.y + noise_scale * (rng.uniform() * 2.0f - 1.0f)};

                // Feature-to-robot frame: col0 = view direction, col1/2 =
                // perpendicular axes for the angular error components.
                vect3f dir = cam.unproject_to_robot(noisy_pixel);
                vect3f cam_up = -(cam.mc2r.col(1));
                vect3f up_proj = cam_up - dir * (cam_up * dir);
                float up_norm = up_proj.norm();
                if (up_norm < 1e-6f) continue;
                vect3f col2 = up_proj * (1.0f / up_norm);
                vect3f col1 = col2 % dir;

                vect2f sigma = cam.pixel_angular_size(noisy_pixel);

                auto feat = info.features().push();
                feat.set_pixel(noisy_pixel);
                feat.set_mf2r(matrix3f::from_cols(dir, col1, col2));
                feat.set_camera_pos(cam.camera_pos);
                feat.set_isigma({1.0f / sigma.x, 1.0f / sigma.y});
                frine_data.push_back({li, pi, info.features().last_ref()});
            }
        }

        // GPS: iid per-fix noise; the constraint covariance is inflated.
        vect3f gps_pos{pos.x + cfg.gps_sigma * rng.normal(),
                       pos.y + cfg.gps_sigma * rng.normal(),
                       pos.z + cfg.gps_sigma * rng.normal()};
        float ms = cfg.gps_sigma * cfg.gps_sigma_inflate;

        // Noisy initial pose estimate.
        const float init_noise_pos = 0.1f;  // meters
        const float init_noise_ea = 0.02f;  // radians
        vect3f noisy_pos{pos.x + init_noise_pos * rng.normal(),
                         pos.y + init_noise_pos * rng.normal(),
                         pos.z + init_noise_pos * rng.normal()};
        vect3f noisy_ea{ea.x + init_noise_ea * rng.normal(),
                        ea.y + init_noise_ea * rng.normal(),
                        ea.z + init_noise_ea * rng.normal()};

        pose.set_r2w_translation(noisy_pos);
        pose.set_r2w_rotation(quaternf::from_euler_angles(noisy_ea));

        info.set_delta_pos(delta_pos);
        info.set_delta_rot(delta_rot);
        matrix3f cov_r; vect3f cov_isigma;
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
        float r = ea.x + tilt_sigma_rad * rng.normal();
        float p = ea.y + tilt_sigma_rad * rng.normal();
        info.set_tilt_g({-std::sin(p), std::cos(p) * std::sin(r),
                         std::cos(p) * std::cos(r)});
    }

    // Landmarks with frines: the anchor is the middlest observing pose,
    // snapshotted at its initial position; direction and inverse range
    // initialize from the noisy landmark guess.
    std::vector<size_t> kept_gt;
    path.landmarks().reserve(gt_landmarks.size());
    for (size_t li = 0; li < gt_landmarks.size(); li++) {
        vect3f lm_pos = gt_landmarks[li].pos;
        vect3f noisy_lm{lm_pos.x + 0.5f * rng.normal(),
                        lm_pos.y + 0.5f * rng.normal(),
                        lm_pos.z + 0.3f * rng.normal()};
        std::vector<const FrineData*> obs;
        for (const auto& fd : frine_data)
            if (fd.li == li) obs.push_back(&fd);
        if (obs.empty()) continue; // skip landmarks with no observations
        kept_gt.push_back(li);

        PoseRef anchor_pose = path.poses().ref_at(uint32_t(obs[obs.size() / 2]->pi));
        vect3f anchor = path.poses().get(anchor_pose).r2w_translation();
        vect3f d = noisy_lm - anchor;

        auto lm = path.landmarks().get(path.landmarks().push());
        lm.set_anchor(anchor);
        lm.set_anchor_pose(anchor_pose);
        lm.set_dir_unit(d * (1.0f / d.norm()));
        lm.set_rho(1.0f / d.norm());
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

struct Stats { float mean, median, min, max; };
static Stats stats(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    float sum = 0;
    for (float x : v) sum += x;
    return {sum / float(v.size()), v[v.size() / 2], v.front(), v.back()};
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
        path.set_frine_cauchy(-1.0f);
        path.set_frine_c2(2.99f);
    } else if (loss_name == "cauchy") {
        path.set_frine_cauchy(1.0f);
        path.set_frine_c2(1.5f);
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
        vect3f t = pose.r2w_translation();
        vect3f ea = pose.r2w_rotation().get_euler_angles();
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
    std::vector<float> isigma_scales =
        std::getenv("SINGLE_PASS") ? std::vector<float>{1.0f}
                                   : std::vector<float>{0.01f, 0.1f, 1.0f};
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
                if (std::abs(lm.rho()) < 1e-4f) continue;
                vect3f world = lm.anchor() + lm.dir_unit() * (1.0f / lm.rho());
                vect3f c_new = path.poses().get(lm.anchor_pose()).r2w_translation();
                vect3f d = world - c_new;
                float n = d.norm();
                if (n < 1e-3f) continue;
                lm.set_anchor(c_new);
                lm.set_dir_unit(d * (1.0f / n));
                lm.set_rho(1.0f / n);
            }
        }
    }

    // Mean absolute pose error vs GT.
    {
        float pos_err_sum = 0.0f, ea_err_sum = 0.0f;
        size_t n = std::min(gt_poses.size(), size_t(path.poses().size()));
        for (size_t i = 0; i < n; i++) {
            auto pose = path.poses()[uint32_t(i)];
            pos_err_sum += (pose.r2w_translation() - gt_poses[i].pos).norm();
            ea_err_sum += (pose.r2w_rotation().get_euler_angles() - gt_poses[i].ea).norm();
        }
        std::printf("\nFinal cost: %.4f\n", path.cost());
        std::printf("Mean pose error vs GT: pos=%.4fm  ea=%.3fdeg\n",
            pos_err_sum / float(n), ea_err_sum / float(n) * 180.0f / float(M_PI));
    }

    // Relative pose errors: consecutive deltas in the local frame.
    std::printf("\n--- Relative pose errors ---\n");
    std::vector<float> dpos_errs, dpos_rel_errs, dea_errs_deg, dea_rel_errs;
    for (size_t i = 1; i < std::min(gt_poses.size(), size_t(path.poses().size())); i++) {
        auto prev = path.poses()[uint32_t(i - 1)];
        auto pose = path.poses()[uint32_t(i)];

        matrix3f gt_mr2w = matrix3f::rotation_from_euler_angles(gt_poses[i - 1].ea);
        vect3f gt_delta_pos = gt_mr2w.transpose() * (gt_poses[i].pos - gt_poses[i - 1].pos);

        matrix3f opt_mr2w_prev = prev.r2w_rotation().rotation_matrix();
        vect3f opt_delta_pos = opt_mr2w_prev.transpose()
            * (pose.r2w_translation() - prev.r2w_translation());

        float dpos_err = (opt_delta_pos - gt_delta_pos).norm();
        float gt_step = gt_delta_pos.norm();
        float dpos_rel = gt_step > 1e-6f ? 100.0f * dpos_err / gt_step : 0.0f;

        matrix3f gt_mr2w_cur = matrix3f::rotation_from_euler_angles(gt_poses[i].ea);
        vect3f gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles();

        matrix3f opt_mr2w_cur = pose.r2w_rotation().rotation_matrix();
        vect3f opt_delta_ea = (opt_mr2w_prev.transpose() * opt_mr2w_cur).get_euler_angles();

        float dea_err = (opt_delta_ea - gt_delta_ea).norm();
        float dea_err_deg = dea_err * 180.0f / float(M_PI);
        float gt_rot = gt_delta_ea.norm();
        float dea_rel = gt_rot > 1e-6f ? 100.0f * dea_err / gt_rot : 0.0f;

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
    std::vector<float> lm_errs, lm_rel_errs;
    std::vector<float> max_sigmas;
    size_t li_out = 0;
    for (auto lm : path.landmarks()) {
        vect3f gt_lm = gt_landmarks[kept_gt[li_out]].pos;

        // Closest GT pose.
        size_t closest_idx = 0;
        float best = std::numeric_limits<float>::max();
        for (size_t j = 0; j < gt_poses.size(); j++) {
            float d = (gt_lm - gt_poses[j].pos).norm();
            if (d < best) { best = d; closest_idx = j; }
        }
        matrix3f gt_mr2w = matrix3f::rotation_from_euler_angles(gt_poses[closest_idx].ea);
        vect3f gt_vec = gt_mr2w.transpose() * (gt_lm - gt_poses[closest_idx].pos);

        auto opt_pose = path.poses()[uint32_t(closest_idx)];
        matrix3f opt_mr2w = opt_pose.r2w_rotation().rotation_matrix();
        vect3f lm_world = lm.anchor() + lm.dir_unit() * (1.0f / lm.rho());
        vect3f opt_vec = opt_mr2w.transpose() * (lm_world - opt_pose.r2w_translation());
        float err = (opt_vec - gt_vec).norm();
        float gt_dist = gt_vec.norm();
        float rel_pct = 100.0f * err / gt_dist;

        // The landmark marginal is [dir chart (2); rho]; map it to
        // world position covariance with J = [unit_d / rho, -unit /
        // rho^2] (the anchor is constant data). The pose marginal is
        // [w (rotation); d (translation)], d in the reference rotation
        // frame: the world translation block is R * C_dd * R^T.
        // Near-infinity landmarks (rho ~ 0) and failed queries print
        // without a sigma.
        bool have_sigma = false;
        double sg[3] = {0, 0, 0};
        if (cov_r.is_ok() && std::abs(lm.rho()) >= 1e-4f) {
            auto& cov = cov_r.value();
            double rho = lm.rho();
            vect3f u = lm.dir_unit();
            vect3f ud0 = lm.dir_unit_d0(), ud1 = lm.dir_unit_d1();
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
            max_sigmas.push_back(float(sg[0]));
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

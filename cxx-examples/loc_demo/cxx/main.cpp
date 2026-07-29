// Localization demo over the generated arael C++ interface -- the C++
// twin of examples/loc_demo.rs. The model and solver are Rust
// (model/); composing the problem, the graduated ramp with the band
// solver, and the error reports are C++.
//
// Same as SLAM but with known (fixed) landmarks: they are plain data,
// so there is no gauge freedom and absolute pose errors are
// meaningful. No GPS. The Hessian is block-tridiagonal (fixed map, no
// loop closures), so the band solver fits: kd = 2*6 - 1 = 11 with
// 6-parameter poses, and CovMode::TriDiagonal recovers the last
// pose's covariance.
#include <path.hpp>
#include <arael/geometry.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace loc_demo;
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
    size_t num_poses = 20;
    size_t num_landmarks = 40;
    uint64_t seed = 42;
    float outlier_fraction = 0.5f; // fraction of invalid associations
    float outlier_scale = 30.0f;   // outlier pixel noise is 30x normal
    float s_amplitude = 1.5f;      // S-curve lateral amplitude (meters)
    float s_frequency = 0.8f;      // S-curve angular frequency
    float step_size = 0.25f;       // distance between poses (meters)
    float odo_pos_k = 0.10f;       // position noise fraction of distance
    float odo_pos_base = 0.03f;    // base position noise (meters)
    float odo_ea_k = 0.01f;
    float odo_ea_base = 0.001f;
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
        float dy = cfg.s_amplitude * cfg.s_frequency * std::cos(cfg.s_frequency * t);
        poses.push_back({{x, y, 0.0f}, {0.0f, 0.0f, std::atan2(dy, 1.0f)}});
        t += cfg.step_size;
    }
    return poses;
}

static std::vector<vect3f> generate_ground_truth_landmarks(
    const SceneConfig& cfg, Rng& rng, const std::vector<GtPose>& poses) {
    // Landmarks 5-30m from the closest pose.
    std::vector<vect3f> landmarks;
    for (size_t i = 0; i < cfg.num_landmarks; i++) {
        for (;;) {
            vect3f anchor = poses[rng.index(poses.size())].pos;
            float angle = rng.uniform() * 2.0f * float(M_PI);
            float dist = 5.0f + rng.uniform() * 25.0f;
            vect3f lm{anchor.x + dist * std::cos(angle),
                      anchor.y + dist * std::sin(angle),
                      rng.uniform() * 2.0f};
            float min_dist = std::numeric_limits<float>::max();
            for (const auto& p : poses)
                min_dist = std::min(min_dist, (lm - p.pos).norm());
            if (min_dist >= 5.0f && min_dist <= 30.0f) {
                landmarks.push_back(lm);
                break;
            }
        }
    }
    return landmarks;
}

static void build_path(const SceneConfig& cfg, Rng& rng, Path& path,
    const std::vector<GtPose>& gt_poses, const std::vector<vect3f>& gt_landmarks) {
    auto cameras = create_cameras();

    const float drift_pos_sigma = 1000.0f;   // meters
    const float drift_ea_sigma_deg = 1800.0f;
    const float tilt_sigma_deg = 0.25f;      // accelerometer accuracy
    const float tilt_sigma_rad = tilt_sigma_deg * float(M_PI) / 180.0f;

    path.set_gamma(2.0f * std::sqrt(25.0f) / float(M_PI));
    path.set_drift_pos_isigma(1.0f / drift_pos_sigma);
    path.set_drift_ea_isigma(1.0f / (drift_ea_sigma_deg * float(M_PI) / 180.0f));
    path.set_tilt_isigma(1.0f / tilt_sigma_rad);
    path.set_frine_isigma_scale(1.0f);

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
        vect3f delta_ea{0.0f, 0.0f, 0.0f};
        if (pi > 0) {
            matrix3f prev_mw2r =
                matrix3f::rotation_from_euler_angles(gt_poses[pi - 1].ea).transpose();
            delta_pos = prev_mw2r * (pos - gt_poses[pi - 1].pos);
            delta_ea = (prev_mw2r * mr2w).get_euler_angles();
        }

        // Odometry covariance proportional to motion.
        float dp_norm = std::max(delta_pos.norm(), 0.01f);
        float de_norm = std::max(delta_ea.norm(), 0.001f);
        float ps = cfg.odo_pos_k * dp_norm + cfg.odo_pos_base;
        vect3f pos_sigma{ps, ps * 0.5f, ps * 0.5f}; // lateral less noisy
        float es = cfg.odo_ea_k * de_norm + cfg.odo_ea_base;
        vect3f ea_sigma{es, es, es};

        auto pose = path.poses().push_back();
        auto info = pose.info();

        // Features: every landmark seen by every camera that faces it.
        for (size_t li = 0; li < gt_landmarks.size(); li++) {
            vect3f lm_pos = gt_landmarks[li];
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

        // Noisy initial pose estimate.
        const float init_noise_pos = 0.1f;  // meters
        const float init_noise_ea = 0.02f;  // radians
        pose.set_pos({pos.x + init_noise_pos * rng.normal(),
                      pos.y + init_noise_pos * rng.normal(),
                      pos.z + init_noise_pos * rng.normal()});
        pose.set_ea({ea.x + init_noise_ea * rng.normal(),
                     ea.y + init_noise_ea * rng.normal(),
                     ea.z + init_noise_ea * rng.normal()});

        info.set_delta_pos(delta_pos);
        info.set_delta_ea(delta_ea);
        matrix3f cov_r; vect3f cov_isigma;
        decompose_cov(diagonal_cov(pos_sigma), cov_r, cov_isigma);
        info.set_delta_pos_cov_r(cov_r);
        info.set_delta_pos_cov_isigma(cov_isigma);
        decompose_cov(diagonal_cov(ea_sigma), cov_r, cov_isigma);
        info.set_delta_ea_cov_r(cov_r);
        info.set_delta_ea_cov_isigma(cov_isigma);
        info.set_tilt_roll(ea.x + tilt_sigma_rad * rng.normal());
        info.set_tilt_pitch(ea.y + tilt_sigma_rad * rng.normal());
    }

    // Landmarks with frines (fixed at their GT positions).
    path.landmarks().reserve(gt_landmarks.size());
    for (size_t li = 0; li < gt_landmarks.size(); li++) {
        std::vector<const FrineData*> obs;
        for (const auto& fd : frine_data)
            if (fd.li == li) obs.push_back(&fd);
        if (obs.empty()) continue;
        auto lm = path.landmarks().get(path.landmarks().push());
        lm.set_pos(gt_landmarks[li]);
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
}

struct Stats { float mean, median, min, max; };
static Stats stats(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    float sum = 0;
    for (float x : v) sum += x;
    return {sum / float(v.size()), v[v.size() / 2], v.front(), v.back()};
}

int main() {
    // Line-buffer stdout so our lines interleave correctly with the
    // solver's verbose output (the Rust side flushes per line).
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    SceneConfig cfg;
    Rng rng(cfg.seed);
    auto gt_poses = generate_ground_truth_poses(cfg);
    auto gt_landmarks = generate_ground_truth_landmarks(cfg, rng, gt_poses);

    Path path;
    build_path(cfg, rng, path, gt_poses, gt_landmarks);

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
        vect3f p = pose.pos(), e = pose.ea();
        std::printf("Pose %2zu: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)\n",
            i, p.x, p.y, p.z, e.x, e.y, e.z);
        std::printf("      gt: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)\n",
            gt_poses[i].pos.x, gt_poses[i].pos.y, gt_poses[i].pos.z,
            gt_poses[i].ea.x, gt_poses[i].ea.y, gt_poses[i].ea.z);
    }
    std::printf("\n");

    // Graduated optimization: start with loose feature constraints,
    // tighten. The band solver fits the block-tridiagonal Hessian:
    // kd = 2*6 - 1 = 11 with 6-parameter poses.
    std::printf("--- Optimization ---\n");
    const float isigma_scales[3] = {0.01f, 0.1f, 1.0f};
    for (size_t pass = 0; pass < 3; pass++) {
        path.set_frine_isigma_scale(isigma_scales[pass]);
        std::printf("\nPass %zu (isigma scale=%g):\n", pass + 1, isigma_scales[pass]);
        LmConfig lm_cfg = LmConfig::well_conditioned();
        lm_cfg.verbose = true;
        LmResult r = path.solve_band(11, lm_cfg).value();
        std::printf("  %u iterations, cost %.4f -> %.4f\n",
            r.iterations, r.start_cost, r.end_cost);
    }

    std::printf("\nFinal cost: %.4f\n", path.cost());

    // Absolute pose errors vs GT (meaningful -- no gauge freedom).
    std::printf("\n--- Absolute pose errors ---\n");
    std::vector<float> pos_errs, ea_errs_deg;
    for (uint32_t i = 0; i < path.poses().size(); i++) {
        auto pose = path.poses()[i];
        vect3f p = pose.pos();
        float pos_err = (p - gt_poses[i].pos).norm();
        float ea_err_deg = (pose.ea() - gt_poses[i].ea).norm() * 180.0f / float(M_PI);
        std::printf("Pose %2u: |d|=%.4fm  ea=%.3fdeg  pos=(%.3f, %.3f, %.3f)\n",
            i, pos_err, ea_err_deg, p.x, p.y, p.z);
        pos_errs.push_back(pos_err);
        ea_errs_deg.push_back(ea_err_deg);
    }
    if (!pos_errs.empty()) {
        Stats s = stats(pos_errs);
        std::printf("Pos: mean=%.4fm  median=%.4fm  min=%.4fm  max=%.4fm\n",
            s.mean, s.median, s.min, s.max);
        s = stats(ea_errs_deg);
        std::printf("EA:  mean=%.3fdeg  median=%.3fdeg  min=%.3fdeg  max=%.3fdeg\n",
            s.mean, s.median, s.min, s.max);
    }

    // Relative pose errors: consecutive deltas in the local frame.
    std::printf("\n--- Relative pose errors ---\n");
    std::vector<float> dpos_errs, dpos_rel_errs, dea_errs_deg, dea_rel_errs;
    for (uint32_t i = 1; i < path.poses().size(); i++) {
        auto prev = path.poses()[i - 1];
        auto pose = path.poses()[i];

        matrix3f gt_mr2w = matrix3f::rotation_from_euler_angles(gt_poses[i - 1].ea);
        vect3f gt_delta_pos = gt_mr2w.transpose() * (gt_poses[i].pos - gt_poses[i - 1].pos);

        matrix3f opt_mr2w_prev = matrix3f::rotation_from_euler_angles(prev.ea());
        vect3f opt_delta_pos = opt_mr2w_prev.transpose() * (pose.pos() - prev.pos());

        float dpos_err = (opt_delta_pos - gt_delta_pos).norm();
        float gt_step = gt_delta_pos.norm();
        float dpos_rel = gt_step > 1e-6f ? 100.0f * dpos_err / gt_step : 0.0f;

        matrix3f gt_mr2w_cur = matrix3f::rotation_from_euler_angles(gt_poses[i].ea);
        vect3f gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles();

        matrix3f opt_mr2w_cur = matrix3f::rotation_from_euler_angles(pose.ea());
        vect3f opt_delta_ea = (opt_mr2w_prev.transpose() * opt_mr2w_cur).get_euler_angles();

        float dea_err = (opt_delta_ea - gt_delta_ea).norm();
        float dea_err_deg = dea_err * 180.0f / float(M_PI);
        float gt_rot = gt_delta_ea.norm();
        float dea_rel = gt_rot > 1e-6f ? 100.0f * dea_err / gt_rot : 0.0f;

        std::printf("Pair %2u-%2u: dpos=%.4fm (%.1f%%)  dea=%.3fdeg (%.1f%%)\n",
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

    // Current (last) pose estimate with 1-sigma uncertainty. H is
    // block-tridiagonal (fixed map, no loop closures), so
    // CovMode::TriDiagonal recovers the last pose's covariance with a
    // forward pass over the band.
    {
        uint32_t last = path.poses().size() - 1;
        auto cov = path.assemble_covariance(CovMode::TriDiagonal).value();
        auto pose = path.poses()[last];
        double sd[6];
        arael_assert_true(cov.std_dev(pose, sd, 6) == 6);
        vect3f p = pose.pos(), e = pose.ea();
        const double deg = 180.0 / M_PI;

        std::printf("\n--- Last pose (%u) estimate +- 1 sigma ---\n", last);
        std::printf("pos x:  %8.4f +- %.4f m\n", p.x, sd[0]);
        std::printf("pos y:  %8.4f +- %.4f m\n", p.y, sd[1]);
        std::printf("pos z:  %8.4f +- %.4f m\n", p.z, sd[2]);
        std::printf("roll :  %8.4f +- %.4f deg\n", e.x * deg, sd[3] * deg);
        std::printf("pitch:  %8.4f +- %.4f deg\n", e.y * deg, sd[4] * deg);
        std::printf("yaw  :  %8.4f +- %.4f deg\n", e.z * deg, sd[5] * deg);
    }
    return 0;
}

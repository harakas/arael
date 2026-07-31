#!/usr/bin/env python3
# Localization demo over the generated arael Python interface -- the
# Python twin of examples/loc_demo.rs and the C++ demo. The model and
# solver are Rust (the cxx-examples model crate, shared); composing
# the problem, the graduated ramp on the band solver, and the reports
# are Python. Own RNG -- same shape, same behavior, numbers differ.
#
# Known (fixed) landmarks: no gauge freedom, absolute pose errors are
# meaningful. The Hessian is block-tridiagonal, so the solves run on
# solve_band(11) and the last pose's 1-sigma comes from
# CovMode.TRI_DIAGONAL + std_dev.
#
# Needs the capi cdylib: `cargo build --release -p loc-demo-capi` in
# the demo root (or set ARAEL_CAPI).
import math
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "model", "python"))

from loc_demo import path as pathmod  # noqa: E402
from loc_demo.arael import cameraf, CovMode  # noqa: E402
from loc_demo.arael.math import matrix3f, vect2f, vect3f  # noqa: E402


class Cfg:
    num_poses = 20
    num_landmarks = 40
    seed = 42
    outlier_fraction = 0.5
    outlier_scale = 30.0
    s_amplitude = 1.5
    s_frequency = 0.8
    step_size = 0.25
    odo_pos_k = 0.10
    odo_pos_base = 0.03
    odo_ea_k = 0.01
    odo_ea_base = 0.001


def create_cameras():
    """5 cameras at 72-degree intervals, looking toward the horizon."""
    cameras = []
    w, h = 1024, 768
    fov_deg = 80.0
    fx = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    for i in range(5):
        yaw = math.radians(i * (360.0 / 5))
        sy, cy = math.sin(yaw), math.cos(yaw)
        mc2r = matrix3f.from_cols((-sy, cy, 0.0), (0.0, 0.0, -1.0),
                                  (cy, sy, 0.0))
        cameras.append(cameraf(fx, fx, w / 2.0, h / 2.0, w, h,
                              (cy * 0.1, sy * 0.1, 0.3), mc2r))
    return cameras


def decompose_cov(cov):
    """Covariance -> (R, 1/sqrt(d)), the form the constraints consume."""
    r, d = cov.symmetric_eigen()
    return r, vect3f(1.0 / math.sqrt(d.x), 1.0 / math.sqrt(d.y),
                     1.0 / math.sqrt(d.z))


def diagonal_cov(sigma):
    return matrix3f.from_elements(
        sigma[0] ** 2, 0.0, 0.0, 0.0, sigma[1] ** 2, 0.0,
        0.0, 0.0, sigma[2] ** 2)


def truth_poses(cfg):
    out = []
    t = 0.0
    for _ in range(cfg.num_poses):
        y = cfg.s_amplitude * math.sin(cfg.s_frequency * t)
        dy = cfg.s_amplitude * cfg.s_frequency * math.cos(cfg.s_frequency * t)
        out.append((vect3f(t, y, 0.0), vect3f(0.0, 0.0, math.atan2(dy, 1.0))))
        t += cfg.step_size
    return out


def truth_landmarks(cfg, rng, poses):
    """Landmarks 5-30m from the closest pose."""
    out = []
    while len(out) < cfg.num_landmarks:
        anchor = poses[rng.randrange(len(poses))][0]
        angle = rng.random() * 2.0 * math.pi
        dist = 5.0 + rng.random() * 25.0
        lm = vect3f(anchor.x + dist * math.cos(angle),
                    anchor.y + dist * math.sin(angle),
                    rng.random() * 2.0)
        min_dist = min((lm - p).norm() for p, _ in poses)
        if 5.0 <= min_dist <= 30.0:
            out.append(lm)
    return out


def build_path(cfg, rng, path, gt_poses, gt_landmarks):
    cameras = create_cameras()

    drift_pos_sigma = 1000.0
    drift_ea_sigma_deg = 1800.0
    tilt_sigma_rad = math.radians(0.25)

    path.gamma = 2.0 * math.sqrt(25.0) / math.pi
    path.drift_pos_isigma = 1.0 / drift_pos_sigma
    path.drift_ea_isigma = 1.0 / math.radians(drift_ea_sigma_deg)
    path.tilt_isigma = 1.0 / tilt_sigma_rad
    path.frine_isigma_scale = 1.0

    # (landmark index, observing-pose index, feature ref).
    frine_data = []

    # Reserve the known counts: growing a collection one push at a time
    # reallocates repeatedly.
    path.poses.reserve(len(gt_poses))
    for pi, (pos, ea) in enumerate(gt_poses):
        mr2w = matrix3f.rotation_from_euler_angles(ea)

        delta_pos = vect3f(0.0, 0.0, 0.0)
        delta_ea = vect3f(0.0, 0.0, 0.0)
        if pi > 0:
            prev_mw2r = matrix3f.rotation_from_euler_angles(
                gt_poses[pi - 1][1]).transpose()
            delta_pos = prev_mw2r * (pos - gt_poses[pi - 1][0])
            delta_ea = (prev_mw2r * mr2w).get_euler_angles()

        dp_norm = max(delta_pos.norm(), 0.01)
        de_norm = max(delta_ea.norm(), 0.001)
        ps = cfg.odo_pos_k * dp_norm + cfg.odo_pos_base
        pos_sigma = (ps, ps * 0.5, ps * 0.5)  # lateral less noisy
        es = cfg.odo_ea_k * de_norm + cfg.odo_ea_base
        ea_sigma = (es, es, es)

        pose = path.poses.push_back()
        info = pose.info

        # Features: every landmark seen by every camera that faces it.
        for li, lm_pos in enumerate(gt_landmarks):
            for cam in cameras:
                p_cam = cam.world_to_camera(lm_pos, pos, mr2w)
                if p_cam.z < 0.5:
                    continue
                pixel = cam.project(p_cam)
                if not cam.is_visible(pixel):
                    continue

                is_outlier = rng.random() < cfg.outlier_fraction
                noise_scale = cfg.outlier_scale if is_outlier else 1.0
                noisy_pixel = vect2f(
                    pixel.x + noise_scale * (rng.random() * 2.0 - 1.0),
                    pixel.y + noise_scale * (rng.random() * 2.0 - 1.0))

                d = cam.unproject_to_robot(noisy_pixel)
                cam_up = -(cam.mc2r.col(1))
                up_proj = cam_up - d * (cam_up * d)
                up_norm = up_proj.norm()
                if up_norm < 1e-6:
                    continue
                col2 = up_proj * (1.0 / up_norm)
                col1 = col2 % d

                sigma = cam.pixel_angular_size(noisy_pixel)

                feat = info.features.push()
                feat.pixel = noisy_pixel
                feat.mf2r = matrix3f.from_cols(d, col1, col2)
                feat.camera_pos = cam.camera_pos
                feat.isigma = (1.0 / sigma.x, 1.0 / sigma.y)
                frine_data.append((li, pi, info.features.last_ref()))

        # Noisy initial pose estimate.
        pose.pos = (pos.x + 0.1 * rng.gauss(0, 1),
                    pos.y + 0.1 * rng.gauss(0, 1),
                    pos.z + 0.1 * rng.gauss(0, 1))
        pose.ea = (ea.x + 0.02 * rng.gauss(0, 1),
                   ea.y + 0.02 * rng.gauss(0, 1),
                   ea.z + 0.02 * rng.gauss(0, 1))

        info.delta_pos = delta_pos
        info.delta_ea = delta_ea
        cov_r, cov_isigma = decompose_cov(diagonal_cov(pos_sigma))
        info.delta_pos_cov_r = cov_r
        info.delta_pos_cov_isigma = cov_isigma
        cov_r, cov_isigma = decompose_cov(diagonal_cov(ea_sigma))
        info.delta_ea_cov_r = cov_r
        info.delta_ea_cov_isigma = cov_isigma
        info.tilt_roll = ea.x + tilt_sigma_rad * rng.gauss(0, 1)
        info.tilt_pitch = ea.y + tilt_sigma_rad * rng.gauss(0, 1)

    # Landmarks with frines (fixed at their GT positions).
    path.landmarks.reserve(len(gt_landmarks))
    for li, lm_pos in enumerate(gt_landmarks):
        obs = [fd for fd in frine_data if fd[0] == li]
        if not obs:
            continue
        lm = path.landmarks[path.landmarks.push()]
        lm.pos = lm_pos
        lm.frines.reserve(len(obs))
        for _, pose_i, feat_ref in obs:
            fr = lm.frines.push()
            fr.pose = path.poses.ref_at(pose_i)
            fr.feature = feat_ref

    # Pose pairs for odometry.
    path.pose_pairs.reserve(len(path.poses) - 1)
    for i in range(1, len(path.poses)):
        pp = path.pose_pairs.push()
        pp.prev = path.poses.ref_at(i - 1)
        pp.cur = path.poses.ref_at(i)


def stats(v):
    v = sorted(v)
    return sum(v) / len(v), v[len(v) // 2], v[0], v[-1]


def main():
    cfg = Cfg()
    rng = random.Random(cfg.seed)
    gt_poses = truth_poses(cfg)
    gt_landmarks = truth_landmarks(cfg, rng, gt_poses)

    path = pathmod.Path()
    build_path(cfg, rng, path, gt_poses, gt_landmarks)

    n_frines = sum(len(lm.frines) for lm in path.landmarks)
    print("Path: %d poses, %d landmarks, %d frines, %d pose_pairs"
          % (len(path.poses), len(path.landmarks), n_frines,
             len(path.pose_pairs)))
    print("Parameters: %d (Pose=%d, Landmark=%d)\n"
          % (len(path.poses) * pathmod.Pose.param_count
             + len(path.landmarks) * pathmod.PointLandmark.param_count,
             pathmod.Pose.param_count, pathmod.PointLandmark.param_count))

    for i in (0, cfg.num_poses // 2, cfg.num_poses - 1):
        pose = path.poses[i]
        p, e = pose.pos, pose.ea
        print("Pose %2d: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)"
              % (i, p.x, p.y, p.z, e.x, e.y, e.z))
        gp, ge = gt_poses[i]
        print("      gt: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)"
              % (gp.x, gp.y, gp.z, ge.x, ge.y, ge.z))
    print()

    # Graduated optimization on the band solver: kd = 2*6 - 1 = 11
    # with 6-parameter poses.
    print("--- Optimization ---", flush=True)
    for passno, scale in enumerate((0.01, 0.1, 1.0), 1):
        path.frine_isigma_scale = scale
        print("\nPass %d (isigma scale=%g):" % (passno, scale), flush=True)
        lm_cfg = pathmod.LmConfig.well_conditioned()
        lm_cfg.verbose = True
        r = path.solve_band(11, lm_cfg)
        print("  %d iterations, cost %.4f -> %.4f"
              % (r.iterations, r.start_cost, r.end_cost), flush=True)

    print("\nFinal cost: %.4f" % path.cost())

    # Absolute pose errors vs GT (meaningful -- no gauge freedom).
    print("\n--- Absolute pose errors ---")
    pos_errs = []
    ea_errs_deg = []
    for i in range(len(path.poses)):
        pose = path.poses[i]
        p = pose.pos
        pos_err = (p - gt_poses[i][0]).norm()
        ea_err_deg = math.degrees((pose.ea - gt_poses[i][1]).norm())
        print("Pose %2d: |d|=%.4fm  ea=%.3fdeg  pos=(%.3f, %.3f, %.3f)"
              % (i, pos_err, ea_err_deg, p.x, p.y, p.z))
        pos_errs.append(pos_err)
        ea_errs_deg.append(ea_err_deg)
    m, md, mn, mx = stats(pos_errs)
    print("Pos: mean=%.4fm  median=%.4fm  min=%.4fm  max=%.4fm" % (m, md, mn, mx))
    m, md, mn, mx = stats(ea_errs_deg)
    print("EA:  mean=%.3fdeg  median=%.3fdeg  min=%.3fdeg  max=%.3fdeg"
          % (m, md, mn, mx))

    # Relative pose errors: consecutive deltas in the local frame.
    print("\n--- Relative pose errors ---")
    dpos_errs = []
    dpos_rel_errs = []
    dea_errs_deg = []
    dea_rel_errs = []
    for i in range(1, len(path.poses)):
        prev = path.poses[i - 1]
        pose = path.poses[i]

        gt_mr2w = matrix3f.rotation_from_euler_angles(gt_poses[i - 1][1])
        gt_delta_pos = gt_mr2w.transpose() * (gt_poses[i][0] - gt_poses[i - 1][0])

        opt_mr2w_prev = matrix3f.rotation_from_euler_angles(prev.ea)
        opt_delta_pos = opt_mr2w_prev.transpose() * (pose.pos - prev.pos)

        dpos_err = (opt_delta_pos - gt_delta_pos).norm()
        gt_step = gt_delta_pos.norm()
        dpos_rel = 100.0 * dpos_err / gt_step if gt_step > 1e-6 else 0.0

        gt_mr2w_cur = matrix3f.rotation_from_euler_angles(gt_poses[i][1])
        gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles()

        opt_mr2w_cur = matrix3f.rotation_from_euler_angles(pose.ea)
        opt_delta_ea = (opt_mr2w_prev.transpose() * opt_mr2w_cur).get_euler_angles()

        dea_err = (opt_delta_ea - gt_delta_ea).norm()
        dea_err_deg = math.degrees(dea_err)
        gt_rot = gt_delta_ea.norm()
        dea_rel = 100.0 * dea_err / gt_rot if gt_rot > 1e-6 else 0.0

        print("Pair %2d-%2d: dpos=%.4fm (%.1f%%)  dea=%.3fdeg (%.1f%%)"
              % (i - 1, i, dpos_err, dpos_rel, dea_err_deg, dea_rel))
        dpos_errs.append(dpos_err)
        dpos_rel_errs.append(dpos_rel)
        dea_errs_deg.append(dea_err_deg)
        dea_rel_errs.append(dea_rel)
    m, md, mn, mx = stats(dpos_errs)
    print("Delta pos: mean=%.4fm  median=%.4fm  min=%.4fm  max=%.4fm"
          % (m, md, mn, mx))
    m, md, mn, mx = stats(dpos_rel_errs)
    print("Delta pos: mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%"
          % (m, md, mn, mx))
    m, md, mn, mx = stats(dea_errs_deg)
    print("Delta ea:  mean=%.3fdeg  median=%.3fdeg  min=%.3fdeg  max=%.3fdeg"
          % (m, md, mn, mx))
    m, md, mn, mx = stats(dea_rel_errs)
    print("Delta ea:  mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%"
          % (m, md, mn, mx))

    # Last pose estimate with 1-sigma uncertainty: H is
    # block-tridiagonal, so TRI_DIAGONAL recovers it with a forward
    # pass over the band.
    last = len(path.poses) - 1
    cov = path.assemble_covariance(CovMode.TRI_DIAGONAL)
    pose = path.poses[last]
    sd = cov.std_dev(pose)
    p, e = pose.pos, pose.ea

    print("\n--- Last pose (%d) estimate +- 1 sigma ---" % last)
    print("pos x:  %8.4f +- %.4f m" % (p.x, sd[0]))
    print("pos y:  %8.4f +- %.4f m" % (p.y, sd[1]))
    print("pos z:  %8.4f +- %.4f m" % (p.z, sd[2]))
    print("roll :  %8.4f +- %.4f deg" % (math.degrees(e.x), math.degrees(sd[3])))
    print("pitch:  %8.4f +- %.4f deg" % (math.degrees(e.y), math.degrees(sd[4])))
    print("yaw  :  %8.4f +- %.4f deg" % (math.degrees(e.z), math.degrees(sd[5])))


if __name__ == "__main__":
    main()

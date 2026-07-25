#!/usr/bin/env python3
# Visual-inertial SLAM demo over the generated arael Python interface
# -- the Python twin of examples/slam_demo_gm.rs and the C++ demo. The
# model and solver are Rust (the cxx-examples model crate, shared);
# composing the synthetic world, the graduated ramp with landmark
# re-anchoring, and the error/uncertainty reports are Python. Own RNG
# -- same shape, same behavior, numbers differ.
#
#   python3 main.py [--solver dense|sparse] [--loss gm|cauchy]
#                   [--poses N] [--landmarks N] [--seed N]
#   SINGLE_PASS=1 skips the ramp (and fails here -- the ramp is what
#   carries the landmarks into their inlier basins).
#
# Needs the capi cdylib: `cargo build --release -p slam-demo-gm-capi`
# in the demo root (or set ARAEL_CAPI).
import math
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "model", "python"))

from slam_demo_gm import path as pathmod  # noqa: E402
from slam_demo_gm.arael import Camera  # noqa: E402
from slam_demo_gm.arael.math import (matrix3d, matrix3f, quaternf,  # noqa: E402
                                     vect2f, vect3f)


class Cfg:
    num_poses = 60
    num_landmarks = 240
    seed = 42
    outlier_fraction = 0.5   # fraction of invalid associations
    outlier_scale = 30.0     # outlier pixel noise is 30x normal
    s_amplitude = 1.5
    s_frequency = 0.8
    step_size = 0.25
    gps_sigma = 0.3
    gps_sigma_inflate = 2.0  # loose GPS covariance: whole-path duty only
    odo_pos_k = 0.10
    odo_pos_base = 0.03
    odo_ea_k = 0.01
    odo_ea_base = 0.001
    lm_visibility_range = 15
    lm_visibility_prob = 0.75


def create_cameras():
    """5 cameras at 72-degree intervals, looking toward the horizon."""
    cameras = []
    w, h = 1024, 768
    fx = (w / 2.0) / math.tan(math.radians(80.0 / 2.0))
    for i in range(5):
        yaw = math.radians(i * (360.0 / 5))
        sy, cy = math.sin(yaw), math.cos(yaw)
        mc2r = matrix3f.from_cols((-sy, cy, 0.0), (0.0, 0.0, -1.0),
                                  (cy, sy, 0.0))
        cameras.append(Camera(fx, fx, w / 2.0, h / 2.0, w, h,
                              (cy * 0.1, sy * 0.1, 0.3), mc2r))
    return cameras


def decompose_cov(cov):
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
    """(landmark position, anchor pose index) pairs, 5-30m out."""
    out = []
    while len(out) < cfg.num_landmarks:
        anchor_idx = rng.randrange(len(poses))
        anchor = poses[anchor_idx][0]
        angle = rng.random() * 2.0 * math.pi
        dist = 5.0 + rng.random() * 25.0
        lm = vect3f(anchor.x + dist * math.cos(angle),
                    anchor.y + dist * math.sin(angle),
                    rng.random() * 2.0)
        min_dist = min((lm - p).norm() for p, _ in poses)
        if 5.0 <= min_dist <= 30.0:
            out.append((lm, anchor_idx))
    return out


def build_path(cfg, rng, path, gt_poses, gt_landmarks):
    """Composes the problem; returns the kept gt index per landmark."""
    cameras = create_cameras()

    tilt_sigma_rad = math.radians(0.25)
    path.drift_rho_isigma = 1.0
    path.tilt_isigma = 1.0 / tilt_sigma_rad
    path.frine_isigma_scale = 1.0
    path.frine_c2 = 2.99
    path.frine_cauchy = -1.0
    path.gps_c2 = 7.815

    frine_data = []  # (landmark index, pose index, feature ref)

    for pi, (pos, ea) in enumerate(gt_poses):
        mr2w = matrix3f.rotation_from_euler_angles(ea)

        delta_pos = vect3f(0.0, 0.0, 0.0)
        delta_rot = matrix3f.identity()
        if pi > 0:
            prev_mw2r = matrix3f.rotation_from_euler_angles(
                gt_poses[pi - 1][1]).transpose()
            delta_pos = prev_mw2r * (pos - gt_poses[pi - 1][0])
            delta_rot = prev_mw2r * mr2w

        dp_norm = max(delta_pos.norm(), 0.01)
        tr = delta_rot[0].x + delta_rot[1].y + delta_rot[2].z
        de_norm = max(math.acos(max(-1.0, min(1.0, (tr - 1.0) * 0.5))), 0.001)
        ps = cfg.odo_pos_k * dp_norm + cfg.odo_pos_base
        pos_sigma = (ps, ps * 0.5, ps * 0.5)
        rs = cfg.odo_ea_k * de_norm + cfg.odo_ea_base
        rot_sigma = (rs, rs, rs)

        pose = path.poses.push_back()
        info = pose.info

        # Features for landmarks visible from this pose.
        for li, (lm_pos, anchor_idx) in enumerate(gt_landmarks):
            if abs(pi - anchor_idx) > cfg.lm_visibility_range:
                continue
            if rng.random() > cfg.lm_visibility_prob:
                continue
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

        # GPS: iid per-fix noise; the constraint covariance is inflated.
        gps_pos = (pos.x + cfg.gps_sigma * rng.gauss(0, 1),
                   pos.y + cfg.gps_sigma * rng.gauss(0, 1),
                   pos.z + cfg.gps_sigma * rng.gauss(0, 1))
        ms = cfg.gps_sigma * cfg.gps_sigma_inflate

        pose.r2w_translation = (pos.x + 0.1 * rng.gauss(0, 1),
                                pos.y + 0.1 * rng.gauss(0, 1),
                                pos.z + 0.1 * rng.gauss(0, 1))
        pose.r2w_rotation = quaternf.from_euler_angles(
            (ea.x + 0.02 * rng.gauss(0, 1),
             ea.y + 0.02 * rng.gauss(0, 1),
             ea.z + 0.02 * rng.gauss(0, 1)))

        info.delta_pos = delta_pos
        info.delta_rot = delta_rot
        cov_r, cov_isigma = decompose_cov(diagonal_cov(pos_sigma))
        info.delta_pos_cov_r = cov_r
        info.delta_pos_cov_isigma = cov_isigma
        cov_r, cov_isigma = decompose_cov(diagonal_cov(rot_sigma))
        info.delta_rot_cov_r = cov_r
        info.delta_rot_cov_isigma = cov_isigma

        gps = info.make_gps()
        gps.pos = gps_pos
        cov_r, cov_isigma = decompose_cov(diagonal_cov((ms, ms, ms)))
        gps.cov_r = cov_r
        gps.cov_isigma = cov_isigma

        # Tilt: angle-space noise, stored as the implied up direction.
        r = ea.x + tilt_sigma_rad * rng.gauss(0, 1)
        p = ea.y + tilt_sigma_rad * rng.gauss(0, 1)
        info.tilt_g = (-math.sin(p), math.cos(p) * math.sin(r),
                       math.cos(p) * math.cos(r))

    # Landmarks: anchored inverse-depth at the middlest observing pose.
    kept_gt = []
    for li, (lm_pos, _) in enumerate(gt_landmarks):
        noisy_lm = vect3f(lm_pos.x + 0.5 * rng.gauss(0, 1),
                          lm_pos.y + 0.5 * rng.gauss(0, 1),
                          lm_pos.z + 0.3 * rng.gauss(0, 1))
        obs = [fd for fd in frine_data if fd[0] == li]
        if not obs:
            continue
        kept_gt.append(li)

        anchor_pose = path.poses.ref_at(obs[len(obs) // 2][1])
        anchor = path.poses.get(anchor_pose).r2w_translation
        d = noisy_lm - anchor

        lm = path.landmarks[path.landmarks.push()]
        lm.anchor = anchor
        lm.anchor_pose = anchor_pose
        lm.dir_unit = d * (1.0 / d.norm())
        lm.rho = 1.0 / d.norm()
        for _, pose_i, feat_ref in obs:
            fr = lm.frines.push()
            fr.pose = path.poses.ref_at(pose_i)
            fr.feature = feat_ref

    for i in range(1, len(path.poses)):
        pp = path.pose_pairs.push()
        pp.prev = path.poses.ref_at(i - 1)
        pp.cur = path.poses.ref_at(i)
    return kept_gt


def stats(v):
    v = sorted(v)
    return sum(v) / len(v), v[len(v) // 2], v[0], v[-1]


def main():
    solver_name = "sparse"
    loss_name = "gm"
    cfg = Cfg()
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]

        def nxt():
            nonlocal i
            i += 1
            return args[i] if i < len(args) else ""

        if a == "--solver":
            solver_name = nxt()
        elif a == "--loss":
            loss_name = nxt()
        elif a == "--poses":
            cfg.num_poses = int(nxt())
        elif a == "--landmarks":
            cfg.num_landmarks = int(nxt())
        elif a == "--seed":
            cfg.seed = int(nxt())
        else:
            print("Unknown argument: %s" % a, file=sys.stderr)
            return 1
        i += 1
    if solver_name == "faer":
        solver_name = "sparse"  # faer is the sparse backend
    if solver_name not in ("dense", "sparse"):
        print("Unknown solver: %s" % solver_name, file=sys.stderr)
        return 1

    print("Solver: %s  Loss: %s  Poses: %d  Landmarks: %d  Seed: %d"
          % (solver_name, loss_name, cfg.num_poses, cfg.num_landmarks,
             cfg.seed))

    rng = random.Random(cfg.seed)
    gt_poses = truth_poses(cfg)
    gt_landmarks = truth_landmarks(cfg, rng, gt_poses)

    path = pathmod.Path()
    kept_gt = build_path(cfg, rng, path, gt_poses, gt_landmarks)

    # Each family's measured-best threshold.
    if loss_name == "gm":
        path.frine_cauchy = -1.0
        path.frine_c2 = 2.99
    elif loss_name == "cauchy":
        path.frine_cauchy = 1.0
        path.frine_c2 = 1.5
    else:
        print("Unknown loss: %s" % loss_name, file=sys.stderr)
        return 1

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
        t = pose.r2w_translation
        e = pose.r2w_rotation.get_euler_angles()
        print("Pose %2d: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)"
              % (i, t.x, t.y, t.z, e.x, e.y, e.z))
        gp, ge = gt_poses[i]
        print("      gt: pos=(%7.3f, %7.3f, %7.3f) ea=(%7.4f, %7.4f, %7.4f)"
              % (gp.x, gp.y, gp.z, ge.x, ge.y, ge.z))
    print()

    # Graduated optimization: loose feature constraints first, tighten;
    # landmark anchors re-snapshot between passes.
    print("--- Optimization ---", flush=True)
    scales = [1.0] if os.environ.get("SINGLE_PASS") else [0.01, 0.1, 1.0]
    for passno, scale in enumerate(scales, 1):
        path.frine_isigma_scale = scale
        print("\nPass %d (isigma scale=%g):" % (passno, scale), flush=True)
        lm_cfg = pathmod.LmConfig.well_conditioned()
        lm_cfg.rel_precision = 1e-6
        lm_cfg.verbose = True
        r = (path.solve_dense(lm_cfg) if solver_name == "dense"
             else path.solve_sparse(lm_cfg))
        print("  %d iterations, cost %.4f -> %.4f"
              % (r.iterations, r.start_cost, r.end_cost), flush=True)

        # Re-anchor: move each landmark's anchor to its anchor pose's
        # CURRENT position; values only. Near-infinity keeps its ray.
        if passno < len(scales):
            for lm in path.landmarks:
                if abs(lm.rho) < 1e-4:
                    continue
                world = lm.anchor + lm.dir_unit * (1.0 / lm.rho)
                c_new = path.poses.get(lm.anchor_pose).r2w_translation
                d = world - c_new
                n = d.norm()
                if n < 1e-3:
                    continue
                lm.anchor = c_new
                lm.dir_unit = d * (1.0 / n)
                lm.rho = 1.0 / n

    # Mean absolute pose error vs GT.
    pos_err_sum = 0.0
    ea_err_sum = 0.0
    for i in range(len(path.poses)):
        pose = path.poses[i]
        pos_err_sum += (pose.r2w_translation - gt_poses[i][0]).norm()
        ea_err_sum += (pose.r2w_rotation.get_euler_angles()
                       - gt_poses[i][1]).norm()
    n = len(path.poses)
    print("\nFinal cost: %.4f" % path.cost())
    print("Mean pose error vs GT: pos=%.4fm  ea=%.3fdeg"
          % (pos_err_sum / n, math.degrees(ea_err_sum / n)))

    # Relative pose errors.
    print("\n--- Relative pose errors ---")
    dpos_errs, dpos_rel_errs, dea_errs_deg, dea_rel_errs = [], [], [], []
    for i in range(1, len(path.poses)):
        prev = path.poses[i - 1]
        pose = path.poses[i]

        gt_mr2w = matrix3f.rotation_from_euler_angles(gt_poses[i - 1][1])
        gt_delta_pos = gt_mr2w.transpose() * (gt_poses[i][0] - gt_poses[i - 1][0])

        opt_mr2w_prev = prev.r2w_rotation.rotation_matrix()
        opt_delta_pos = opt_mr2w_prev.transpose() * (
            pose.r2w_translation - prev.r2w_translation)

        dpos_err = (opt_delta_pos - gt_delta_pos).norm()
        gt_step = gt_delta_pos.norm()
        dpos_rel = 100.0 * dpos_err / gt_step if gt_step > 1e-6 else 0.0

        gt_mr2w_cur = matrix3f.rotation_from_euler_angles(gt_poses[i][1])
        gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles()

        opt_mr2w_cur = pose.r2w_rotation.rotation_matrix()
        opt_delta_ea = (opt_mr2w_prev.transpose()
                        * opt_mr2w_cur).get_euler_angles()

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

    # Landmark uncertainty: relative covariance C_ll + C_pp - C_lp -
    # C_pl over landmark and pose position blocks cancels the shared
    # gauge uncertainty; ellipsoid semi-axes = sqrt of eigenvalues.
    cov = path.assemble_covariance()

    print("\n--- Landmark errors (relative to closest pose) ---")
    lm_errs, lm_rel_errs, max_sigmas = [], [], []
    for li_out, (li, lm) in enumerate(zip(kept_gt, path.landmarks)):
        gt_lm = gt_landmarks[li][0]

        closest_idx = min(range(len(gt_poses)),
                          key=lambda j: (gt_lm - gt_poses[j][0]).norm())
        gt_mr2w = matrix3f.rotation_from_euler_angles(gt_poses[closest_idx][1])
        gt_vec = gt_mr2w.transpose() * (gt_lm - gt_poses[closest_idx][0])

        opt_pose = path.poses[closest_idx]
        opt_mr2w = opt_pose.r2w_rotation.rotation_matrix()
        lm_world = lm.anchor + lm.dir_unit * (1.0 / lm.rho)
        opt_vec = opt_mr2w.transpose() * (lm_world - opt_pose.r2w_translation)
        err = (opt_vec - gt_vec).norm()
        gt_dist = gt_vec.norm()
        rel_pct = 100.0 * err / gt_dist

        # Landmark marginal is [dir chart (2); rho]; J maps it to world
        # position covariance. Pose marginal is [w (3); d (3)].
        sg = None
        if abs(lm.rho) >= 1e-4:
            try:
                rho = float(lm.rho)
                u = lm.dir_unit
                ud0, ud1 = lm.dir_unit_d0, lm.dir_unit_d1
                j = matrix3d.from_elements(
                    ud0.x / rho, ud1.x / rho, -u.x / (rho * rho),
                    ud0.y / rho, ud1.y / rho, -u.y / (rho * rho),
                    ud0.z / rho, ud1.z / rho, -u.z / (rho * rho))
                marg = cov.marginal(lm)
                pose_m = cov.marginal(opt_pose)   # 6x6 row-major tuples
                x = cov.cross(lm, opt_pose)       # 3x6 row-major tuples
                c_ll = j * marg * j.transpose()
                r_m = opt_mr2w.cast()
                c_dd = matrix3d.from_elements(
                    *[pose_m[r][c] for r in range(3, 6) for c in range(3, 6)])
                c_pp = r_m * c_dd * r_m.transpose()
                xd = matrix3d.from_elements(
                    *[x[r][c] for r in range(3) for c in range(3, 6)])
                c_lp = j * xd * r_m.transpose()
                cov_rel = c_ll + c_pp - c_lp - c_lp.transpose()
                _, eval_ = cov_rel.symmetric_eigen()
                sg = sorted((math.sqrt(max(eval_.x, 0.0)),
                             math.sqrt(max(eval_.y, 0.0)),
                             math.sqrt(max(eval_.z, 0.0))), reverse=True)
            except Exception:
                sg = None
        if sg is not None:
            print("LM %3d: |d|=%.3fm  rel=%.2f%%  dist=%.1fm  "
                  "sigma=(%.3f,%.3f,%.3f)m  frines=%d"
                  % (li_out, err, rel_pct, gt_dist, sg[0], sg[1], sg[2],
                     len(lm.frines)))
            max_sigmas.append(sg[0])
        else:
            print("LM %3d: |d|=%.3fm  rel=%.2f%%  dist=%.1fm  frines=%d"
                  % (li_out, err, rel_pct, gt_dist, len(lm.frines)))
        lm_errs.append(err)
        lm_rel_errs.append(rel_pct)

    m, md, mn, mx = stats(lm_errs)
    print("LM pos:  mean=%.3fm  median=%.3fm  min=%.3fm  max=%.3fm"
          % (m, md, mn, mx))
    m, md, mn, mx = stats(lm_rel_errs)
    print("LM rel:  mean=%.2f%%  median=%.2f%%  min=%.2f%%  max=%.2f%%"
          % (m, md, mn, mx))
    if max_sigmas:
        m, md, mn, mx = stats(max_sigmas)
        print("Max principal sigma: mean=%.3fm  median=%.3fm  min=%.3fm  "
              "max=%.3fm" % (m, md, mn, mx))
    return 0


if __name__ == "__main__":
    sys.exit(main())

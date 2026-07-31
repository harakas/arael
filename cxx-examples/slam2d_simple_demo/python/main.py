#!/usr/bin/env python3
# 2D SLAM from Python over the generated arael interface -- the Python
# twin of examples/slam2d_simple_demo.rs and the C++ demo. The model
# and solver are Rust (the cxx-examples model crate, shared); this
# file synthesizes the world, composes the problem, solves, reports
# errors against ground truth, and plots the map to an EPS file. Own
# RNG, so the numbers differ from the other twins -- same shape, same
# behavior.
#
# Needs the capi cdylib: `cargo build --release -p slam2d-simple-capi`
# in the demo root (or set ARAEL_CAPI).
import math
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "model", "python"))

from slam2d_simple import path as pathmod  # noqa: E402
from slam2d_simple.arael.math import matrix2f, vect2f  # noqa: E402


class Cfg:
    n_poses = 20
    n_landmarks = 30
    seed = 42
    step = 1.5
    turn = 0.10
    fov_half = math.radians(60.0)
    range_min = 4.0
    range_max = 50.0
    odo_pos_sigma = 0.05
    odo_gamma_sigma = math.radians(0.3)
    bearing_sigma = math.radians(1.0)
    init_range = 20.0


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def truth_poses(cfg):
    """Gentle left-turning arc starting at the origin facing east."""
    out = [(vect2f(0, 0), 0.0)]
    pos, gamma = out[0]
    for _ in range(1, cfg.n_poses):
        pos = pos + vect2f(cfg.step * math.cos(gamma),
                           cfg.step * math.sin(gamma))
        gamma += cfg.turn
        out.append((pos, gamma))
    return out


def truth_landmarks(cfg, rng, poses):
    """Corners scattered around the trajectory, at observable distance."""
    out = []
    while len(out) < cfg.n_landmarks:
        anchor = poses[rng.randrange(len(poses))][0]
        theta = rng.random() * 2.0 * math.pi
        r = cfg.range_min + rng.random() * (cfg.range_max - cfg.range_min)
        lm = anchor + vect2f(r * math.cos(theta), r * math.sin(theta))
        if any(cfg.range_min <= (lm - p).norm() <= cfg.range_max
               for p, _ in poses):
            out.append(lm)
    return out


def observe(cfg, pos, gamma, lm):
    """Bearing from (pos, gamma) to lm, FOV/range gated; None = unseen."""
    d = lm - pos
    dist = d.norm()
    if dist < cfg.range_min or dist > cfg.range_max:
        return None
    local = matrix2f.rotation(gamma).transpose() * d
    bearing = math.atan2(local.y, local.x)
    if abs(bearing) > cfg.fov_half:
        return None
    return bearing


def ellipse_from_cov(center, c):
    """(center, semi_major, semi_minor, angle) of the 95% ellipse."""
    r, d = c.symmetric_eigen()  # ascending eigenvalues
    chi2_95 = 5.991
    major = r.col(1)
    return (center,
            math.sqrt(max(d.y, 0.0) * chi2_95),
            math.sqrt(max(d.x, 0.0) * chi2_95),
            math.atan2(major.y, major.x))


def hsv_to_rgb(h, s, v):
    h6 = (h - math.floor(h)) * 6.0
    c = v * s
    x = c * (1.0 - abs(math.fmod(h6, 2.0) - 1.0))
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x),
               (0, x, c), (x, 0, c), (c, 0, x)][min(int(h6), 5)]
    m = v - c
    return r + m, g + m, b + m


def landmark_color(i, n, ray):
    h = 0.0 if n == 0 else i / n
    return hsv_to_rgb(h, 0.40 if ray else 0.85, 0.97 if ray else 0.78)


def write_eps(file, gt_poses, gt_lms, est_poses, est_lms, lm_sightings,
              lm_to_gt, ellipses):
    pts = ([p for p, _ in est_poses] + est_lms + [p for p, _ in gt_poses]
           + [gt_lms[gi] for gi in lm_to_gt])
    xmin = min(p.x for p in pts) - 3
    xmax = max(p.x for p in pts) + 3
    ymin = min(p.y for p in pts) - 3
    ymax = max(p.y for p in pts) + 3

    page_w, page_h, pad = 540.0, 420.0, 18.0
    s = min((page_w - 2 * pad) / (xmax - xmin),
            (page_h - 2 * pad) / (ymax - ymin))
    dx = (page_w - s * (xmax - xmin)) * 0.5
    dy = (page_h - s * (ymax - ymin)) * 0.5

    def X(p):
        return dx + (p.x - xmin) * s

    def Y(p):
        return dy + (p.y - ymin) * s

    f = open(file, "w")
    f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
    f.write("%%%%BoundingBox: 0 0 %d %d\n" % (int(page_w), int(page_h)))
    f.write("%%Creator: slam2d_simple_demo (Python)\n%%EndComments\n")
    f.write("/tri { gsave 4 2 roll translate exch rotate "
            "dup 0 moveto "
            "dup -0.55 mul 1 index 0.45 mul lineto "
            "dup -0.55 mul exch -0.45 mul lineto "
            "closepath fill grestore } def\n")
    f.write("/dot { newpath 0 360 arc fill } def\n")

    def polyline(pts):
        f.write("newpath ")
        for i, (p, _) in enumerate(pts):
            f.write("%.2f %.2f %s " % (X(p), Y(p), "lineto" if i else "moveto"))
        f.write("stroke\n")

    # Ground-truth pose shadow (behind everything).
    f.write("0.62 0.62 0.62 setrgbcolor 0.8 setlinewidth [3 2] 0 setdash\n")
    polyline(gt_poses)
    f.write("[] 0 setdash\n")
    for p, g in gt_poses:
        f.write("%.2f %.2f %.2f 8 tri\n" % (X(p), Y(p), math.degrees(g)))

    # Bearing rays from each optimized pose.
    f.write("0.25 setlinewidth\n")
    n_lm = len(est_lms)
    for li in range(n_lm):
        r, g, b = landmark_color(li, n_lm, True)
        f.write("%.3f %.3f %.3f setrgbcolor\n" % (r, g, b))
        for pose_i, bearing in lm_sightings[li]:
            pp = est_poses[pose_i][0]
            world_dir = est_poses[pose_i][1] + bearing
            dist = (est_lms[li] - pp).norm() * 1.10
            tip = pp + vect2f(dist * math.cos(world_dir),
                              dist * math.sin(world_dir))
            f.write("newpath %.2f %.2f moveto %.2f %.2f lineto stroke\n"
                    % (X(pp), Y(pp), X(tip), Y(tip)))

    # Optimized pose chain (dashed) + filled triangles along gamma.
    f.write("0.08 0.15 0.30 setrgbcolor 1.0 setlinewidth [4 2] 0 setdash\n")
    polyline(est_poses)
    f.write("[] 0 setdash 0.10 0.18 0.40 setrgbcolor\n")
    for p, g in est_poses:
        f.write("%.2f %.2f %.2f 6.5 tri\n" % (X(p), Y(p), math.degrees(g)))

    # 95% confidence ellipses, each in its landmark's hue.
    f.write("0.6 setlinewidth\n")
    for i, (center, semi_major, semi_minor, angle) in enumerate(ellipses):
        if semi_major <= 0 or semi_minor <= 0:
            continue
        r, g, b = landmark_color(i, n_lm, False)
        f.write("%.3f %.3f %.3f setrgbcolor\nnewpath " % (r, g, b))
        segs = 48
        ct, st = math.cos(angle), math.sin(angle)
        for j in range(segs + 1):
            phi = 2.0 * math.pi * j / segs
            lx = semi_major * math.cos(phi)
            ly = semi_minor * math.sin(phi)
            w = vect2f(center.x + ct * lx - st * ly,
                       center.y + st * lx + ct * ly)
            f.write("%.2f %.2f %s " % (X(w), Y(w), "lineto" if j else "moveto"))
        f.write("closepath stroke\n")

    # Landmark error lines + GT landmark dots.
    f.write("0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth\n")
    for i in range(n_lm):
        gt = gt_lms[lm_to_gt[i]]
        f.write("newpath %.2f %.2f moveto %.2f %.2f lineto stroke\n"
                % (X(est_lms[i]), Y(est_lms[i]), X(gt), Y(gt)))
    for gi in lm_to_gt:
        f.write("%.2f %.2f 2.2 dot\n" % (X(gt_lms[gi]), Y(gt_lms[gi])))

    # Optimized landmarks, one hue per landmark.
    for i in range(n_lm):
        r, g, b = landmark_color(i, n_lm, False)
        f.write("%.3f %.3f %.3f setrgbcolor %.2f %.2f 2.8 dot\n"
                % (r, g, b, X(est_lms[i]), Y(est_lms[i])))

    f.write("%%EOF\n")
    f.close()


def main():
    cfg = Cfg()
    rng = random.Random(cfg.seed)

    gt_poses = truth_poses(cfg)
    gt_lms = truth_landmarks(cfg, rng, gt_poses)

    path = pathmod.Path()

    # Dead-reckoned initial estimates from noisy odometry.
    est_pos = vect2f(0, 0)
    est_gamma = 0.0

    # Reserve the known counts: growing a collection one push at a time
    # reallocates repeatedly.
    path.poses.reserve(len(gt_poses))
    path.pose_pairs.reserve(len(gt_poses))
    for pi in range(len(gt_poses)):
        pose = path.poses.push_back()
        if pi == 0:
            # Hold the first pose fixed: every measurement is relative.
            pose.pos_optimize = False
            pose.gamma_optimize = False
            continue
        gt_p, gt_g = gt_poses[pi]
        prev_p, prev_g = gt_poses[pi - 1]
        true_delta = matrix2f.rotation(prev_g).transpose() * (gt_p - prev_p)
        true_dg = gt_g - prev_g
        noisy_delta = true_delta + vect2f(cfg.odo_pos_sigma * rng.gauss(0, 1),
                                          cfg.odo_pos_sigma * rng.gauss(0, 1))
        noisy_dg = true_dg + cfg.odo_gamma_sigma * rng.gauss(0, 1)

        est_pos = est_pos + matrix2f.rotation(est_gamma) * noisy_delta
        est_gamma += noisy_dg

        pose.pos = est_pos
        pose.gamma = est_gamma
        pose.delta_pos = noisy_delta
        pose.delta_gamma = noisy_dg
        pose.delta_pos_isigma = 1.0 / cfg.odo_pos_sigma
        pose.delta_gamma_isigma = 1.0 / cfg.odo_gamma_sigma

        pair = path.pose_pairs.push()
        pair.prev = path.poses.ref_at(pi - 1)
        pair.cur = path.poses.ref_at(pi)

    # Landmarks with at least two sightings; init on the first ray.
    lm_refs = []
    lm_to_gt = []
    lm_sightings = []
    n_frines = 0
    path.landmarks.reserve(len(gt_lms))
    for li in range(len(gt_lms)):
        sightings = []
        for pi in range(len(gt_poses)):
            b = observe(cfg, gt_poses[pi][0], gt_poses[pi][1], gt_lms[li])
            if b is None:
                continue
            sightings.append((pi, b + cfg.bearing_sigma * rng.gauss(0, 1)))
        if len(sightings) < 2:  # two rays to triangulate
            continue

        r = path.landmarks.push()
        lm = path.landmarks[r]
        first_pi, first_b = sightings[0]
        p0 = path.poses[first_pi]
        world_b = p0.gamma + first_b
        lm.pos = p0.pos + vect2f(cfg.init_range * math.cos(world_b),
                                 cfg.init_range * math.sin(world_b))
        lm.frines.reserve(len(sightings))
        for pose_i, bearing in sightings:
            fr = lm.frines.push()
            fr.pose = path.poses.ref_at(pose_i)
            fr.bearing = bearing
            fr.isigma = 1.0 / cfg.bearing_sigma
            n_frines += 1
        lm_refs.append(r)
        lm_to_gt.append(li)
        lm_sightings.append(sightings)

    # Iteration works on every container view (arena walks live slots).
    frines_in_model = sum(len(lm.frines) for lm in path.landmarks)
    print("Path: %d poses, %d pose_pairs, %d landmarks, %d frines (%d wired)"
          % (len(path.poses), len(path.pose_pairs), len(path.landmarks),
             n_frines, frines_in_model), flush=True)

    # gather_timing fills the result's timing block, which the report
    # below breaks down per phase.
    cfg_lm = pathmod.LmConfig.well_conditioned()
    cfg_lm.verbose = True
    cfg_lm.gather_timing = True
    r = path.solve_sparse(cfg_lm)  # raises on failure
    # The result prints itself: status, cost, where the time went --
    # rendered by the Rust side from the full solve result.
    print("\n%s\n" % r.pretty_report(), flush=True)

    print("-- Pose errors vs GT --")
    est_poses = []
    pos_sum = 0.0
    g_sum = 0.0
    for i, (gt_p, gt_g) in enumerate(gt_poses):
        p = path.poses[i].pos
        g = path.poses[i].gamma
        est_poses.append((p, g))
        pe = (p - gt_p).norm()
        ge = abs(wrap_angle(g - gt_g))
        print("  pose %2d: |dp|=%.3fm  |dgamma|=%.3fdeg"
              % (i, pe, math.degrees(ge)))
        pos_sum += pe
        g_sum += ge
    print("  mean: pos=%.4fm  gamma=%.3fdeg"
          % (pos_sum / len(gt_poses), math.degrees(g_sum / len(gt_poses))))

    print("\n-- Landmark errors vs GT --")
    est_lms = []
    lm_sum = 0.0
    for i, r in enumerate(lm_refs):
        lm = path.landmarks[r]
        p = lm.pos
        est_lms.append(p)
        e = (p - gt_lms[lm_to_gt[i]]).norm()
        print("  lm %2d: |d|=%.3fm  frines=%d" % (i, e, len(lm.frines)))
        lm_sum += e
    if lm_refs:
        print("  mean: |d|=%.4fm" % (lm_sum / len(lm_refs)))

    # Per-landmark uncertainty from the parameter covariance.
    ellipses = []
    cov = path.assemble_covariance()
    for r in lm_refs:
        lm = path.landmarks[r]
        ellipses.append(ellipse_from_cov(lm.pos, cov.marginal(lm)))
    print("\n%d landmark uncertainty ellipses (95%%)" % len(ellipses))

    out = "slam2d_simple_py.eps"
    write_eps(out, gt_poses, gt_lms, est_poses, est_lms, lm_sightings,
              lm_to_gt, ellipses)
    print("\nMap plotted to %s" % out)


if __name__ == "__main__":
    main()

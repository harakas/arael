#!/usr/bin/env python3
"""Report and remove the degenerate observations in a BAL problem file.

A BAL camera looks down -z, so an observation is only meaningful when the
point sits in front of the lens: X_cam.z < 0, by some margin. Ladybug-1723
carries observations that fail this -- points behind the camera, and points
whose depth cancels to within rounding of the optical centre. Double
precision absorbs them; single precision does not, and the perspective
divide turns into 0/0.

Run with no --write to see the depth distribution and decide a threshold:

    ./clean_bal.py datasets/problem-1723-156502-pre.txt

Then write a cleaned file, which every runner (arael, Ceres, g2o) must be
pointed at alike -- each parses the dataset itself, so cleaning inside one
loader would silently hand that system a different problem:

    ./clean_bal.py datasets/problem-1723-156502-pre.txt \\
        --write datasets/problem-1723-156502-clean.txt

Dropping observations can leave a point with too few views to triangulate,
so points below --min-views go too, and cameras left with no observations
after that. Indices are renumbered contiguously; the result is a valid BAL
file. stdlib only.
"""

import argparse
import math
import sys


def read_bal(path):
    """Parse a BAL file into (cameras, points, observations).

    Values are whitespace-separated and free-form across lines, so the whole
    file is read as one token stream.
    """
    with open(path) as f:
        tok = f.read().split()
    at = 0

    def take(n):
        nonlocal at
        vals = tok[at:at + n]
        at += n
        return vals

    n_cams, n_points, n_obs = (int(v) for v in take(3))
    obs = []
    for _ in range(n_obs):
        c, p, x, y = take(4)
        obs.append((int(c), int(p), float(x), float(y)))
    cams = [[float(v) for v in take(9)] for _ in range(n_cams)]
    points = [[float(v) for v in take(3)] for _ in range(n_points)]
    if at != len(tok):
        sys.exit(f"{path}: {len(tok) - at} trailing values, file is malformed")
    return cams, points, obs


def cam_depth(cam, point):
    """z of the point in camera coordinates: X_cam = R(w) X + t, third row.

    R is the Rodrigues axis-angle rotation, matching the runners' residual.
    """
    wx, wy, wz = cam[0], cam[1], cam[2]
    px, py, pz = point
    theta2 = wx * wx + wy * wy + wz * wz
    if theta2 > 1e-24:
        theta = math.sqrt(theta2)
        c = math.cos(theta)
        s = math.sin(theta)
        kx, ky, kz = wx / theta, wy / theta, wz / theta
        # (k x p).z and (k.p), the two terms the z row needs
        cross_z = kx * py - ky * px
        dot = kx * px + ky * py + kz * pz
        z = pz * c + cross_z * s + kz * dot * (1.0 - c)
    else:
        z = pz + wx * py - wy * px
    return z + cam[5]


def report(depths):
    """Depth distribution, oriented so 'in front' is a positive margin."""
    front = [-d for d in depths]  # in front of the lens => positive
    behind = sum(1 for m in front if m <= 0.0)
    print(f"observations            : {len(depths)}")
    print(f"behind the camera (z>=0): {behind}")
    print(f"exactly at z == 0       : {sum(1 for d in depths if d == 0.0)}")
    print()
    print("margin in front of the lens, cumulative counts below a threshold:")
    for t in (0.0, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0):
        print(f"  < {t:<8g} : {sum(1 for m in front if m < t)}")
    ok = sorted(m for m in front if m > 0.0)
    if ok:
        print()
        print("smallest positive margins:", ", ".join(f"{m:.3e}" for m in ok[:8]))


def clean(cams, points, obs, depths, min_depth, min_views):
    """Drop shallow observations, then points too sparsely seen, then
    cameras left with nothing. Returns renumbered (cams, points, obs)."""
    kept = [o for o, d in zip(obs, depths) if -d >= min_depth]
    dropped_shallow = len(obs) - len(kept)

    views = {}
    for c, p, _, _ in kept:
        views[p] = views.get(p, 0) + 1
    good_points = {p for p, n in views.items() if n >= min_views}
    kept = [o for o in kept if o[1] in good_points]

    used_cams = sorted({o[0] for o in kept})
    used_points = sorted(good_points)
    cam_map = {c: i for i, c in enumerate(used_cams)}
    point_map = {p: i for i, p in enumerate(used_points)}

    print(f"dropped {dropped_shallow} observations shallower than {min_depth:g}")
    print(f"dropped {len(points) - len(used_points)} points seen fewer than "
          f"{min_views} times")
    print(f"dropped {len(cams) - len(used_cams)} cameras left with no observations")
    print(f"kept {len(used_cams)} cameras, {len(used_points)} points, "
          f"{len(kept)} observations")

    out_obs = [(cam_map[c], point_map[p], x, y) for c, p, x, y in kept]
    return ([cams[c] for c in used_cams], [points[p] for p in used_points], out_obs)


def write_bal(path, cams, points, obs):
    with open(path, "w") as f:
        f.write(f"{len(cams)} {len(points)} {len(obs)}\n")
        for c, p, x, y in obs:
            f.write(f"{c} {p} {x!r} {y!r}\n")
        for cam in cams:
            for v in cam:
                f.write(f"{v!r}\n")
        for pt in points:
            for v in pt:
                f.write(f"{v!r}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--write", metavar="OUT",
                    help="write the cleaned problem here (default: report only)")
    ap.add_argument("--min-depth", type=float, default=1e-6,
                    help="required margin in front of the lens (default 1e-6)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="drop points seen fewer than this many times (default 2)")
    args = ap.parse_args()

    cams, points, obs = read_bal(args.input)
    depths = [cam_depth(cams[c], points[p]) for c, p, _, _ in obs]
    print(f"{args.input}: {len(cams)} cameras, {len(points)} points")
    report(depths)
    if not args.write:
        return
    print()
    cams, points, obs = clean(cams, points, obs, depths, args.min_depth, args.min_views)
    write_bal(args.write, cams, points, obs)
    print(f"wrote {args.write}")


if __name__ == "__main__":
    main()

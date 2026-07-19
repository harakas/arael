# SymForce code generation for the plane SLAM benchmark.
#
# Defines the SAME residuals every other system optimizes, symbolically,
# and generates flattened C++ linearization functions (residual + Jacobian
# + Gauss-Newton Hessian block + rhs) with CSE via SymForce's own codegen.
# cpp/symforce_plane.cpp consumes the generated headers (committed under
# cpp/symforce_gen/).
#
# Poses are sf.Pose3 (quaternion rotation + translation, SymForce's native
# decoupled retraction -- the same separate-blocks class as arael, Ceres
# and g2o). A plane is a V3 (dy, dz, c): a fixed tangent chart around the
# plane's INITIAL normal (the anchor rotation baked into each observation
# residual as a constant) plus the absolute distance coefficient -- the
# same formulation as the factrs runner. The gauge is a strong prior on
# pose 0, zero at the start point. Epsilon is set to zero so atan2 behaves
# as the plain C function the other systems call.
#
# usage: SYMFORCE_SYMBOLIC_API=sympy \
#        PYTHONPATH=~/.local/opt/symforce:~/.local/opt/symforce/gen/python \
#        ~/.local/opt/symforce/.venv/bin/python symforce_gen.py <output_dir>

import sys

import symforce

symforce.set_epsilon_to_number(0)

import symforce.symbolic as sf
from symforce.codegen import Codegen, CppConfig


def vee(m: sf.M33) -> sf.V3:
    """vee((M - M^T)/2)."""
    return sf.V3(
        0.5 * (m[2, 1] - m[1, 2]),
        0.5 * (m[0, 2] - m[2, 0]),
        0.5 * (m[1, 0] - m[0, 1]),
    )


def chart_normal(x: sf.V3, anchor: sf.M33) -> sf.V3:
    """World normal from the chart delta (x[0], x[1]) around the anchor:
    the rotated first column of the small-rotation matrix of
    (1, (0, dy, dz)/2) normalized -- exact on the sphere for every delta."""
    dy, dz = x[0], x[1]
    s2 = 1 + (dy * dy + dz * dz) / 4
    local = sf.V3(1 - (dy * dy + dz * dz) / (2 * s2), dz / s2, -dy / s2)
    return anchor * local


def odo_residual(
    pose_a: sf.Pose3,
    pose_b: sf.Pose3,
    tm: sf.V3,
    rm_t: sf.M33,
    wt: sf.Scalar,
    wr: sf.Scalar,
) -> sf.V6:
    """The benchmark's shared odometry between-residual:
    err_t = R_a^T (t_b - t_a) - t_m; err_r = vee((R_m^T R_a^T R_b - .^T)/2)."""
    ra = pose_a.rotation().to_rotation_matrix()
    rb = pose_b.rotation().to_rotation_matrix()
    dt = ra.T * (pose_b.position() - pose_a.position()) - tm
    dr = rm_t * ra.T * rb
    er = vee(dr)
    return sf.V6(dt[0] * wt, dt[1] * wt, dt[2] * wt,
                 er[0] * wr, er[1] * wr, er[2] * wr)


def obs_residual(
    pose: sf.Pose3,
    plane: sf.V3,
    anchor: sf.M33,
    nm: sf.V3,
    cm: sf.Scalar,
    waz: sf.Scalar,
    wel: sf.Scalar,
    wd: sf.Scalar,
) -> sf.V3:
    """The benchmark's shared plane observation (g2o Plane3D::ominus,
    algebraically): azimuth/elevation of the measured normal in the frame
    aligning the predicted local normal with e1, plus the distance
    difference."""
    nw = chart_normal(plane, anchor)
    rp = pose.rotation().to_rotation_matrix()
    nl = rp.T * nw
    cl = plane[2] + pose.position().dot(nw)
    h = sf.sqrt(nl[0] * nl[0] + nl[1] * nl[1])
    mx = nl.dot(nm)
    my = (nm[1] * nl[0] - nm[0] * nl[1]) / h
    mz = (nm[2] * (nl[0] * nl[0] + nl[1] * nl[1])
          - nl[2] * (nl[0] * nm[0] + nl[1] * nm[1])) / h
    return sf.V3(
        sf.atan2(my, mx) * waz,
        sf.atan2(mz, sf.sqrt(mx * mx + my * my)) * wel,
        (cm - cl) * wd,
    )


def prior_residual(
    pose: sf.Pose3,
    prior: sf.Pose3,
    w: sf.Scalar,
) -> sf.V6:
    """Strong 6-DOF gauge prior on pose 0, zero at the start point."""
    dt = pose.position() - prior.position()
    dr = (prior.rotation().inverse() * pose.rotation()).to_rotation_matrix()
    er = vee(dr)
    return sf.V6(dt[0] * w, dt[1] * w, dt[2] * w,
                 er[0] * w, er[1] * w, er[2] * w)


def main() -> None:
    output_dir = sys.argv[1]
    for fn, which_args in [
        (odo_residual, ["pose_a", "pose_b"]),
        (obs_residual, ["pose", "plane"]),
        (prior_residual, ["pose"]),
    ]:
        codegen = Codegen.function(fn, config=CppConfig())
        with_lin = codegen.with_linearization(which_args=which_args)
        with_lin.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    print(f"generated into {output_dir}")


if __name__ == "__main__":
    main()

# SymForce code generation for the heterogeneous SLAM benchmark.
#
# Defines the SAME six residuals every other system optimizes,
# symbolically, and generates flattened C++ linearization functions
# (residual + Jacobian + Gauss-Newton Hessian block + rhs) with CSE via
# SymForce's own codegen -- the production path Skydio benchmarks with.
# cpp/symforce_slam.cpp consumes the generated headers.
#
# Poses are plain 6-vectors [x, y, z, roll, pitch, yaw]; the rotation is
# built from the euler angles inside each residual, matching arael's
# SimpleEulerAngleParam (no manifold -- SymForce retracts a V6 additively,
# same as arael's parameter update). Landmarks are 3-vectors. The euler
# convention and the odometry euler extraction reproduce arael's matrix3
# twins so every residual equals scene::reference_cost. Epsilon is set to
# zero so atan2/asin behave as the plain C functions the other systems
# call.
#
# usage: symforce-venv/bin/python3 symforce_gen.py <output_dir>

import sys

import symforce

symforce.set_epsilon_to_number(0)

import symforce.symbolic as sf
from symforce.codegen import Codegen, CppConfig


# -- arael's rotation convention (x=roll, y=pitch, z=yaw; Rz Ry Rx) --
def euler_to_rot(ea: sf.V3) -> sf.M33:
    sx, cx = sf.sin(ea[0]), sf.cos(ea[0])
    sy, cy = sf.sin(ea[1]), sf.cos(ea[1])
    sz, cz = sf.sin(ea[2]), sf.cos(ea[2])
    return sf.M33([
        [cy * cz, -cx * sz + cz * sx * sy, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ])


# arael's get_euler_angles main branch (error rotation is near identity).
def rot_to_euler(m: sf.M33) -> sf.V3:
    return sf.V3(sf.atan2(m[2, 1], m[2, 2]), -sf.asin(m[2, 0]), sf.atan2(m[1, 0], m[0, 0]))


def gps_residual(pose: sf.V6, gps_pos: sf.V3, cov_r: sf.M33, isigma: sf.V3) -> sf.V3:
    """Position vs a biased GPS reading, whitened by cov_r^T (plain Gaussian)."""
    raw = sf.V3(pose[0], pose[1], pose[2]) - gps_pos
    rt = cov_r.T * raw
    return sf.V3(rt[0] * isigma[0], rt[1] * isigma[1], rt[2] * isigma[2])


def drift_residual(pose: sf.V6, prior: sf.V6, pi: sf.Scalar, ei: sf.Scalar) -> sf.V6:
    """Soft prior to the initialization (position weight pi, euler weight ei)."""
    return sf.V6((pose[0] - prior[0]) * pi, (pose[1] - prior[1]) * pi,
                 (pose[2] - prior[2]) * pi, (pose[3] - prior[3]) * ei,
                 (pose[4] - prior[4]) * ei, (pose[5] - prior[5]) * ei)


def tilt_residual(pose: sf.V6, tilt: sf.V2, isigma: sf.Scalar) -> sf.V2:
    """Roll/pitch from an accelerometer."""
    return sf.V2((pose[3] - tilt[0]) * isigma, (pose[4] - tilt[1]) * isigma)


def lm_drift_residual(lm: sf.V3, prior: sf.V3, isigma: sf.Scalar) -> sf.V3:
    """Soft prior to the landmark initialization."""
    return sf.V3((lm[0] - prior[0]) * isigma, (lm[1] - prior[1]) * isigma,
                 (lm[2] - prior[2]) * isigma)


def bearing_residual(lm: sf.V3, pose: sf.V6, mf2r: sf.M33, camera_pos: sf.V3,
                     isigma: sf.V2, scale: sf.Scalar) -> sf.V2:
    """atan2 bearing residual in the feature frame (plain Gaussian)."""
    mr2w = euler_to_rot(sf.V3(pose[3], pose[4], pose[5]))
    d = lm - sf.V3(pose[0], pose[1], pose[2])
    lm_r = mr2w.T * d
    r_r = lm_r - camera_pos
    r_f = mf2r.T * r_r
    return sf.V2(sf.atan2(r_f[1], r_f[0]) * (isigma[0] * scale),
                 sf.atan2(r_f[2], r_f[0]) * (isigma[1] * scale))


def odo_residual(prev: sf.V6, cur: sf.V6, dpos: sf.V3, dea: sf.V3, pcr: sf.M33,
                 pci: sf.V3, ecr: sf.M33, eci: sf.V3) -> sf.V6:
    """Full 6-DOF relative motion (rotation composition + euler extraction)."""
    mr2w_prev = euler_to_rot(sf.V3(prev[3], prev[4], prev[5]))
    mr2w_cur = euler_to_rot(sf.V3(cur[3], cur[4], cur[5]))
    d = sf.V3(cur[0], cur[1], cur[2]) - sf.V3(prev[0], prev[1], prev[2])
    pos_diff = mr2w_prev.T * d
    pos_err = pos_diff - dpos
    pos_w = pcr.T * pos_err
    expected = mr2w_prev * euler_to_rot(dea)
    error_rot = expected.T * mr2w_cur
    ea_err = rot_to_euler(error_rot)
    ea_w = ecr.T * ea_err
    return sf.V6(pos_w[0] * pci[0], pos_w[1] * pci[1], pos_w[2] * pci[2],
                 ea_w[0] * eci[0], ea_w[1] * eci[1], ea_w[2] * eci[2])


def main() -> None:
    output_dir = sys.argv[1]
    for fn, which_args in [
        (gps_residual, ["pose"]),
        (drift_residual, ["pose"]),
        (tilt_residual, ["pose"]),
        (lm_drift_residual, ["lm"]),
        (bearing_residual, ["lm", "pose"]),
        (odo_residual, ["prev", "cur"]),
    ]:
        codegen = Codegen.function(fn, config=CppConfig())
        with_lin = codegen.with_linearization(which_args=which_args)
        with_lin.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    print(f"generated into {output_dir}")


if __name__ == "__main__":
    main()

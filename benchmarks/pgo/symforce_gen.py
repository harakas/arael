# SymForce code generation for the pose-graph benchmark.
#
# Defines the SAME rotation-frame between residual every other system
# optimizes, symbolically, and generates flattened C++ linearization
# functions (residual + Jacobian + Gauss-Newton Hessian block + rhs)
# with CSE via SymForce's own codegen -- the production path Skydio
# benchmarks with. cpp/symforce_bench.cpp consumes the generated
# headers.
#
# usage: symforce-venv/bin/python3 symforce_gen.py <output_dir>

import sys

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.codegen import Codegen, CppConfig


def between_residual(
    pose_a: sf.Pose2,
    pose_b: sf.Pose2,
    dx: sf.Scalar,
    dy: sf.Scalar,
    dtheta: sf.Scalar,
    wt: sf.Scalar,
    wr: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V3:
    """Rotation-frame SE2 between residual with sqrt-info row weights:
    r = [ wt * (R_b^T (p_a + R_a t_d - p_b)) ; wr * wrap(th_a + dth - th_b) ]
    """
    g = pose_a.position() + pose_a.rotation() * sf.V2(dx, dy) - pose_b.position()
    local = pose_b.rotation().inverse() * g
    rel = pose_b.rotation().inverse() * pose_a.rotation() * sf.Rot2.from_tangent([dtheta])
    dth = rel.to_tangent(epsilon=epsilon)[0]
    return sf.V3(local[0] * wt, local[1] * wt, dth * wr)


def prior_residual(
    pose: sf.Pose2,
    px: sf.Scalar,
    py: sf.Scalar,
    ptheta: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V3:
    """Unit-weight gauge prior on the anchor pose."""
    rel = sf.Rot2.from_tangent([ptheta]).inverse() * pose.rotation()
    dth = rel.to_tangent(epsilon=epsilon)[0]
    return sf.V3(pose.position()[0] - px, pose.position()[1] - py, dth)


def rot3_vec_residual(rel: sf.Rot3) -> sf.V3:
    """2 * vec(q) on the qw >= 0 branch: the canonical quaternion-vector
    rotation residual (2 sin(theta/2) * axis), smooth at zero error."""
    s = 2 * sf.sign_no_zero(rel.q.w)
    return sf.V3(rel.q.x * s, rel.q.y * s, rel.q.z * s)


def between_residual3(
    pose_a: sf.Pose3,
    pose_b: sf.Pose3,
    meas: sf.Pose3,
    sqrt_info: sf.M66,
) -> sf.V6:
    """Canonical SE3 between residual with the full 6x6 upper sqrt
    information factor (info = U^T U):
    r = U * [ R_a^T (t_b - t_a) - t_meas ; 2 vec(q_meas^-1 q_a^-1 q_b) ]
    """
    ra_inv = pose_a.rotation().inverse()
    terr = ra_inv * (pose_b.position() - pose_a.position()) - meas.position()
    rrot = rot3_vec_residual(meas.rotation().inverse() * ra_inv * pose_b.rotation())
    r = sf.V6(terr[0], terr[1], terr[2], rrot[0], rrot[1], rrot[2])
    return sqrt_info * r


def prior_residual3(
    pose: sf.Pose3,
    prior: sf.Pose3,
) -> sf.V6:
    """Unit-weight gauge prior on the anchor pose, same convention."""
    terr = pose.position() - prior.position()
    rrot = rot3_vec_residual(prior.rotation().inverse() * pose.rotation())
    return sf.V6(terr[0], terr[1], terr[2], rrot[0], rrot[1], rrot[2])


def main() -> None:
    output_dir = sys.argv[1]
    for fn, which_args in [
        (between_residual, ["pose_a", "pose_b"]),
        (prior_residual, ["pose"]),
        (between_residual3, ["pose_a", "pose_b"]),
        (prior_residual3, ["pose"]),
    ]:
        codegen = Codegen.function(fn, config=CppConfig())
        with_lin = codegen.with_linearization(which_args=which_args)
        with_lin.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    print(f"generated into {output_dir}")


if __name__ == "__main__":
    main()

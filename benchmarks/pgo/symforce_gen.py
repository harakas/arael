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


def main() -> None:
    output_dir = sys.argv[1]
    for fn, which_args in [
        (between_residual, ["pose_a", "pose_b"]),
        (prior_residual, ["pose"]),
    ]:
        codegen = Codegen.function(fn, config=CppConfig())
        with_lin = codegen.with_linearization(which_args=which_args)
        with_lin.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    print(f"generated into {output_dir}")


if __name__ == "__main__":
    main()

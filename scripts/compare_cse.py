#!/usr/bin/env python3
"""Compare our CSE output with SymPy's CSE on the same frine constraint."""

from sympy import symbols, sin, cos, atan2, atan, cse, Matrix, Symbol
from sympy.printing import ccode
import itertools

# Pose params
pose_pos_x, pose_pos_y, pose_pos_z = symbols('pose.pos.work().x pose.pos.work().y pose.pos.work().z')
pose_ea_x, pose_ea_y, pose_ea_z = symbols('pose.ea.work().x pose.ea.work().y pose.ea.work().z')

# Landmark params
lm_pos_x, lm_pos_y, lm_pos_z = symbols('lm.pos.work().x lm.pos.work().y lm.pos.work().z')

# Feature constants (not differentiated)
mf2r = [[Symbol(f'feature.mf2r[{i}].{c}') for c in 'xyz'] for i in range(3)]
cam_pos_x, cam_pos_y, cam_pos_z = symbols('feature.camera_pos.x feature.camera_pos.y feature.camera_pos.z')
isigma_x, isigma_y = symbols('feature.isigma.x feature.isigma.y')
gamma = Symbol('path.gamma')

# Build rotation matrix from euler angles (same formula as arael)
sx, cx = sin(pose_ea_x), cos(pose_ea_x)
sy, cy = sin(pose_ea_y), cos(pose_ea_y)
sz, cz = sin(pose_ea_z), cos(pose_ea_z)

mr2w = Matrix([
    [cy*cz, -cx*sz + cz*sx*sy, cx*cz*sy + sx*sz],
    [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
    [-sy, cy*sx, cx*cy]
])

# lm_r = mr2w.T * (lm.pos - pose.pos)  -- landmark in robot frame
cam_pos = Matrix([cam_pos_x, cam_pos_y, cam_pos_z])
pose_pos = Matrix([pose_pos_x, pose_pos_y, pose_pos_z])
lm_pos = Matrix([lm_pos_x, lm_pos_y, lm_pos_z])
lm_r = mr2w.T * (lm_pos - pose_pos)

# r_r = lm_r - camera_pos  -- vector from camera to landmark in robot frame
r_r = lm_r - cam_pos

# r_f = mf2r.T * r_r
mf2r_mat = Matrix(mf2r)
r_f = mf2r_mat.T * r_r

# Residuals with robust suppression
plain1 = atan2(r_f[1], r_f[0]) * isigma_x
plain2 = atan2(r_f[2], r_f[0]) * isigma_y
err1 = gamma * atan(plain1 / gamma)
err2 = gamma * atan(plain2 / gamma)

# Param variables to differentiate w.r.t.
params = [lm_pos_x, lm_pos_y, lm_pos_z,
          pose_pos_x, pose_pos_y, pose_pos_z,
          pose_ea_x, pose_ea_y, pose_ea_z]

print(f"Computing derivatives of {2} residuals w.r.t. {len(params)} params...")

# Compute all derivatives
all_exprs = [err1, err2]
for r in [err1, err2]:
    for p in params:
        all_exprs.append(r.diff(p))

print(f"Total expressions: {len(all_exprs)}")
print(f"Before CSE: total ops ~ {sum(e.count_ops() for e in all_exprs)}")

# Apply SymPy CSE
intermediates, simplified = cse(all_exprs, optimizations='basic')

print(f"\nSymPy CSE: {len(intermediates)} intermediates")
total_ops = sum(e.count_ops() for _, e in intermediates) + sum(e.count_ops() for e in simplified)
print(f"After CSE: total ops ~ {total_ops}")

print(f"\n--- SymPy intermediates ---")
for name, expr in intermediates:
    print(f"let {name} = {expr};")

print(f"\n--- SymPy simplified ---")
names = ['r_0', 'r_1']
for p in params:
    names.append(f'dr_0_{str(p).replace(".", "_")}')
for p in params:
    names.append(f'dr_1_{str(p).replace(".", "_")}')

# Actually the order is: r_0, dr_0_0..dr_0_8, r_1, dr_1_0..dr_1_8
real_names = ['r_0']
for i in range(len(params)):
    real_names.append(f'dr_0_{i}')
real_names.append('r_1')
for i in range(len(params)):
    real_names.append(f'dr_1_{i}')

for name, expr in zip(real_names, simplified):
    s = str(expr)
    if len(s) > 120:
        s = s[:120] + '...'
    print(f"let {name} = {s};")

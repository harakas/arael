# Parity program for the arael Python math library: computes the same
# operations as tests/cxx_math/golden.rs from the same inputs and
# prints "name value" lines. The Rust test runs and compares.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "cargo-arael", "python"))

from arael import g2o                                            # noqa: E402
from arael.geometry import camerad, cameraf                                # noqa: E402
from arael.math import (matrix2d, matrix3d, matrix3f, quaternd,  # noqa: E402
                        quaternf, vect2d, vect2f, vect3d, vect3f)


def p(n, x):
    print("%s %.17e" % (n, x))


def pv2(n, v):
    p(n + ".x", v.x)
    p(n + ".y", v.y)


def pv3(n, v):
    p(n + ".x", v.x)
    p(n + ".y", v.y)
    p(n + ".z", v.z)


def pq(n, q):
    p(n + ".t", q.t)
    p(n + ".vx", q.v.x)
    p(n + ".vy", q.v.y)
    p(n + ".vz", q.v.z)


def pm3(n, m):
    for r in range(3):
        for c in range(3):
            p("%s.%d%d" % (n, r, c), m[r][c])


def pm2(n, m):
    for r in range(2):
        for c in range(2):
            p("%s.%d%d" % (n, r, c), m[r][c])


a = vect3d(1.2, -0.5, 2.0)
b = vect3d(0.3, 0.9, -1.1)
pv3("v3_add", a + b)
pv3("v3_sub", a - b)
pv3("v3_neg", -a)
p("v3_dot", a * b)
pv3("v3_cross", a % b)
pv3("v3_scale", a * 2.5)
pv3("v3_lscale", 2.5 * a)
p("v3_norm", a.norm())
p("v3_square", a.square())
pv3("v3_unit", a.unit())
pv3("v3_across_a", a.across())
pv3("v3_across_x", vect3d(1.0, 0.0, 0.0).across())
pv3("v3_deg2rad", a.rad2deg().deg2rad())

u = vect2d(0.8, -1.3)
w = vect2d(-0.2, 0.7)
pv2("v2_add", u + w)
p("v2_dot", u * w)
p("v2_cross", u.cross(w))
pv2("v2_across", u.across())
pv2("v2_unit", u.unit())

ea = vect3d(0.3, -0.7, 1.9)
r = matrix3d.rotation_from_euler_angles(ea)
pm3("m3_rot", r)
pv3("m3_rot_ea", r.get_euler_angles())
pm3("m3_rot_t", r.transpose())
p("m3_rot_det", r.det())
pv3("v3_ea_rotmat", ea.rotation_matrix().col(0))

# Near gimbal lock.
ea_lock = vect3d(0.4, 1.5707963258, -1.2)
rl = matrix3d.rotation_from_euler_angles(ea_lock)
pv3("m3_lock_ea", rl.get_euler_angles())

am = matrix3d.from_elements(1.1, -0.2, 0.4, 0.0, 2.2, -1.0, 0.7, 0.3, 1.9)
pm3("m3_mul", r * am)
pv3("m3_mul_v", am * a)
pv3("v3_mul_m", a * am)
pm3("m3_add", r + am)
pm3("m3_scale", am * -1.5)
p("m3_det", am.det())
pv3("m3_col1", am.col(1))

axis = vect3d(1.0, 2.0, -2.0).unit()
ra = matrix3d.rotation_from_axis_angle(axis, 2.1)
pm3("m3_axis", ra)

small = vect3d(0.01, -0.02, 0.03)
pm3("m3_rvs", matrix3d.from_rotation_vector_small(small))
pv3("m3_rvs_get",
    matrix3d.from_rotation_vector_small(small).get_rotation_vector_small())

q = quaternd.from_euler_angles(ea)
pq("q_ea", q)
pv3("q_ea_back", q.get_euler_angles())
pm3("q_rotmat", q.rotation_matrix())
pq("q_from_m", quaternd.from_rotation_matrix(r))
pv3("q_lock_ea", quaternd.from_euler_angles(ea_lock).get_euler_angles())

qa = quaternd.from_axis_angle(axis, 2.1)
pq("q_axis", qa)
qax, qaa = qa.get_axis_angle()
pv3("q_axis_get", qax)
p("q_angle_get", qaa)

pq("q_mul", q * qa)
pq("q_conj", q.conj())
p("q_dot", q.dot(qa))
pv3("q_rotate", q.rotate(a))
pq("q_pow", qa.pow(0.37))
pq("q_slerp", quaternd.slerp(q, qa, 0.3))
pq("q_log", qa.log())
pq("q_exp", qa.log().exp())
pq("q_rv", quaternd.from_rotation_vector(vect3d(0.4, -0.2, 0.9)))
pq("q_rv_tiny", quaternd.from_rotation_vector(vect3d(1e-5, 0.0, 0.0)))
pq("q_rvs", quaternd.from_rotation_vector_small(small))
pq("q_two", quaternd.from_two_vectors(vect3d(1.0, 0.0, 0.0),
                                      vect3d(0.2, 0.9, -0.1).unit()))
pq("q_two_anti", quaternd.from_two_vectors(vect3d(1.0, 0.0, 0.0),
                                           vect3d(-1.0, 0.0, 0.0)))

r2 = matrix2d.rotation(0.7)
pm2("m2_rot", r2)
p("m2_angle", r2.get_rotation_angle())
pm2("m2_mul", r2 * matrix2d.from_elements(1.2, -0.3, 0.5, 0.8))
pv2("m2_mul_v", r2 * u)
p("m2_det", r2.det())

# Symmetric eigen: eigenvalues directly; eigenvectors through the
# reconstruction R diag(d) R^T (sign/algorithm independent).
s3 = matrix3d.from_elements(4.0, 0.5, -0.3, 0.5, 2.5, 0.7, -0.3, 0.7, 1.2)
r_e, d = s3.symmetric_eigen()
pv3("eig3_d", d)
pm3("eig3_recon", r_e * matrix3d.from_elements(
    d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z) * r_e.transpose())
s2 = matrix2d.from_elements(2.0, 0.6, 0.6, 1.1)
r2e, d2 = s2.symmetric_eigen()
pv2("eig2_d", d2)
pm2("eig2_recon", r2e * matrix2d.from_elements(d2.x, 0.0, 0.0, d2.y)
    * r2e.transpose())

# Pinhole camera (cameraf: f32 storage, compared at f32 accuracy).
cam = cameraf(800.0, 820.0, 512.0, 384.0, 1024, 768,
              vect3f(0.1, -0.05, 0.3),
              matrix3f.rotation_from_euler_angles(vect3f(0.1, -0.2, 0.5)))
px = vect2f(600.0, 300.0)
pv2("f32_cam_proj", cam.project(vect3f(0.4, -0.3, 2.0)).cast())
pv3("f32_cam_unproj", cam.unproject(px).cast())
pv3("f32_cam_w2c", cam.world_to_camera(
    vect3f(3.0, 1.0, 0.5), vect3f(1.0, 0.2, 0.0),
    matrix3f.rotation_from_euler_angles(vect3f(0.02, 0.05, 1.1))).cast())
pv3("f32_cam_unproj_robot", cam.unproject_to_robot(px).cast())
pv2("f32_cam_pixang", cam.pixel_angular_size(px).cast())
p("f32_cam_vis_in", 1.0 if cam.is_visible(px) else 0.0)
p("f32_cam_vis_out", 1.0 if cam.is_visible(vect2f(-1.0, 300.0)) else 0.0)

# Pinhole camera at f64 (camerad), compared exactly.
cam_d = camerad(800.0, 820.0, 512.0, 384.0, 1024, 768,
                vect3d(0.1, -0.05, 0.3),
                matrix3d.rotation_from_euler_angles(vect3d(0.1, -0.2, 0.5)))
px_d = vect2d(600.0, 300.0)
pv2("cam_d_proj", cam_d.project(vect3d(0.4, -0.3, 2.0)))
pv3("cam_d_w2c", cam_d.world_to_camera(
    vect3d(3.0, 1.0, 0.5), vect3d(1.0, 0.2, 0.0),
    matrix3d.rotation_from_euler_angles(vect3d(0.02, 0.05, 1.1))))
pv2("cam_d_pixang", cam_d.pixel_angular_size(px_d))

# g2o SE2 parsing (exact doubles through the text round trip).
ds = g2o.Dataset2.parse(
    "VERTEX_SE2 0 0.25 -1.5 0.125\n"
    "VERTEX_SE2 1 1.75 0.5 -0.25\n"
    "FIX 0\n"
    "EDGE_SE2 0 1 1.5 2.0 -0.375 100 0 0 100 0 400\n")
p("g2o_n_poses", float(len(ds.poses)))
p("g2o_n_deltas", float(len(ds.deltas)))
pv2("g2o_p1_t", ds.poses[1].t)
p("g2o_p1_th", ds.poses[1].th)
pv2("g2o_d0_dt", ds.deltas[0].dt)
p("g2o_d0_dth", ds.deltas[0].dth)
iso = ds.deltas[0].iso_sqrt_info()
p("g2o_d0_iso", 1.0 if iso is not None else 0.0)
p("g2o_d0_wt", iso[0])
p("g2o_d0_wr", iso[1])

# g2o SE3 parsing: quaternion normalization, the symmetric information
# matrix, and its Cholesky blocks.
ds3 = g2o.Dataset3.parse(
    "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
    "VERTEX_SE3:QUAT 1 1.0 2.0 -0.5 0.2 0.4 -0.4 1.0\n"
    "EDGE_SE3:QUAT 0 1 1.5 -0.25 0.75 0.1 -0.2 0.3 0.9 "
    "100 1 2 0 0 0 100 3 0 0 0 100 0 0 0 400 4 0 400 5 400\n")
p("g2o3_n_poses", float(len(ds3.poses)))
p("g2o3_n_deltas", float(len(ds3.deltas)))
pv3("g2o3_p1_t", ds3.poses[1].t)
pq("g2o3_p1_q", ds3.poses[1].q)
pm3("g2o3_p1_rot", ds3.poses[1].rot())
d3 = ds3.deltas[0]
pv3("g2o3_d0_dt", d3.dt)
pq("g2o3_d0_dq", d3.dq)
p("g2o3_d0_i03", d3.info[0][3])
p("g2o3_d0_i34", d3.info[3][4])
u_tt, u_tr, u_rr = d3.u_blocks()
pm3("g2o3_u_tt", u_tt)
pm3("g2o3_u_tr", u_tr)
pm3("g2o3_u_rr", u_rr)

# similar / is_finite / null_space / quatern cast.
a2 = vect3d(1.2, -0.5, 2.0)
b2 = vect3d(0.3, 0.9, -1.1)
r2 = matrix3d.rotation_from_euler_angles(vect3d(0.3, -0.7, 1.9))
q2 = quaternd.from_euler_angles(vect3d(0.3, -0.7, 1.9))
p("sim_v3_same", 1.0 if a2.similar(a2) else 0.0)
p("sim_v3_diff", 1.0 if a2.similar(b2) else 0.0)
p("sim_m3_same", 1.0 if r2.similar(r2) else 0.0)
p("sim_q_same", 1.0 if q2.similar(q2) else 0.0)
p("fin_v3", 1.0 if a2.is_finite() else 0.0)
p("fin_v3_nan", 1.0 if vect3d(float("nan"), 0.0, 0.0).is_finite() else 0.0)
ax = vect3d(1.0, 2.0, -2.0).unit()
pm3("m3_nullspace", matrix3d.null_space(ax))
pq("f32_q_cast", quaternf.from_euler_angles(vect3f(0.3, -0.7, 1.9)).cast())

# f32 smoke: euler round trip, printed at double precision.
eaf_back = quaternf.from_euler_angles(vect3f(0.3, -0.7, 1.9)).get_euler_angles()
pv3("f32_ea_back", eaf_back.cast())

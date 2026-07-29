# Builds the fixture problem through the GENERATED Python interface
# (ctypes over the capi cdylib named by $ARAEL_CAPI), solves, and
# prints "name value" lines for the Rust side to compare -- the same
# protocol and names as parity_main.cpp.
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "model", "python"))

from cxx_fit import fit  # noqa: E402
from cxx_fit.arael import AraelError  # noqa: E402
from cxx_fit.arael.math import matrix3d, quaternd, vect3d  # noqa: E402


def p(n, v):
    print("%s %.17e" % (n, v))


def pi(n, v):
    print("%s %d" % (n, v))


def fill(f):
    # The counts are known, so reserve them.
    f.obs.reserve(6)
    f.items.reserve(3)
    for i in range(6):
        o = f.obs.push()
        o.x = float(i)
        o.y = 2.0 * i + 1.0 + (0.05 if i % 2 == 0 else -0.05)
    t = [1.5, -0.3, 0.7]
    w = [1.0, 2.0, 0.5]
    for i in range(3):
        n = f.items.push()
        n.t = t[i]
        n.w = w[i]


f = fit.Fit()
fill(f)
pi("clean", 1 if len(f.validate()) == 0 else 0)
pi("n_obs", len(f.obs))
pi("n_items", len(f.items))
p("obs3_y", f.obs[3].y)
p("item1_t", f.items[1].t)

# The config holds the preset's actual Rust values.
defs = fit.LmConfig()
p("cfg_abs", defs.abs_precision)
p("cfg_rel", defs.rel_precision)
pi("cfg_max_iters", defs.max_iters)
pi("cfg_min_iters", defs.min_iters)
pi("cfg_patience", defs.patience)
pi("cfg_threads", defs.num_threads)
pi("cfg_verbose", 1 if defs.verbose else 0)
p("cfg_lambda", defs.initial_lambda)
p("cfg_cost_threshold", defs.cost_threshold)
p("cfg_lambda_floor", defs.lambda_floor)
pi("cfg_grad_has", 1 if defs.gradient_tolerance is not None else 0)
pi("cfg_time_has", 1 if defs.time_limit_seconds is not None else 0)
p("cfg_wc_lambda", fit.LmConfig.well_conditioned().initial_lambda)

cfg = fit.LmConfig()
cfg.max_iters = 50
r = f.solve_dense(cfg)
pi("dense_status", int(r.status))
p("dense_start", r.start_cost)
p("dense_end", r.end_cost)
pi("dense_iters", r.iterations)
p("dense_m", f.m)
p("dense_c", f.c)
for i in range(3):
    p("dense_v%d" % i, f.items[i].v)

# Covariance at the solution: the 1x1 marginal of the first item.
cov = f.assemble_covariance()
pi("cov_ok", 1)
m0 = cov.marginal(f.items[0])
pi("cov_item0_ok", 1 if isinstance(m0, float) else 0)
p("cov_item0", m0)

fit2 = fit.Fit()
fill(fit2)
r2 = fit2.solve_sparse(cfg)
pi("sparse_status", int(r2.status))
p("sparse_end", r2.end_cost)
p("sparse_m", fit2.m)
p("sparse_c", fit2.c)

# Observer callback + timing + report + conditional covariance.
f7 = fit.Fit()
pi("report_empty_before", 1 if len(f7.last_report()) == 0 else 0)
fill(f7)
cfg7 = fit.LmConfig()
cfg7.max_iters = 50
cfg7.gather_timing = True
obs_state = {"calls": 0, "plen": 0}


def observer(it):
    obs_state["calls"] += 1
    obs_state["plen"] = it.params_len
    return True


cfg7.observer = observer
r7 = f7.solve_dense(cfg7)
pi("obs_calls_eq_iters", 1 if obs_state["calls"] == r7.iterations else 0)
pi("obs_params_len", obs_state["plen"])
p("obs_end", r7.end_cost)
pi("tm_has", 1 if r7.timing is not None else 0)
pi("tm_total_pos", 1 if r7.timing.total > 0.0 else 0)
pi("tm_assembly_count", r7.timing.assembly_count)
pi("tm_solve_count", r7.timing.linear_solve_count)
pi("tm_cost_count", r7.timing.cost_eval_count)
pi("report_nonempty", 1 if len(f7.last_report()) > 0 else 0)
pi("report_pretty_nonempty", 1 if len(f7.last_pretty_report()) > 0 else 0)
cov7 = f7.assemble_covariance()
cc = cov7.conditional(f7.items[0])
pi("cond_n", 1 if isinstance(cc, float) else 0)
p("cond_item0", cc)

# Compound params: universal euler, quaternion param, user component.
f10 = fit.Fit()
fill(f10)
ea_a = vect3d(0.2, -0.3, 0.7)
ea_b = vect3d(-0.4, 0.1, -1.2)
rot_a = matrix3d.rotation_from_euler_angles(ea_a)
rot_b = matrix3d.rotation_from_euler_angles(ea_b)
f10.rigs.reserve(2)
rig0 = f10.rigs.push()
rig0.target_u0 = rot_a.row(0)
rig0.target_u2 = rot_a.row(2)
rig0.target_q0 = rot_b.row(0)
rig0.target_q2 = rot_b.row(2)
rig0.target_g = 1.75
rig0.ea_u = (0.15, -0.25, 0.6)
rig0.q = quaternd.from_euler_angles((-0.35, 0.05, -1.1))
rig0.gain.g = 0.25
# Second rig with the euler param frozen: it must stay put.
rig1 = f10.rigs.push()
rig1.target_u0 = rot_b.row(0)
rig1.target_u2 = rot_b.row(2)
rig1.target_q0 = rot_a.row(0)
rig1.target_q2 = rot_a.row(2)
rig1.target_g = -0.5
rig1.ea_u = ea_a
rig1.ea_u_optimize = False
rig1.q = quaternd.from_euler_angles(ea_a)
rig1.gain.g = -0.75
r10 = f10.solve_dense(cfg)
pi("rig_status", int(r10.status))
p("rig_end", r10.end_cost)
r0e = rig0.ea_u
p("rig0_ea_x", r0e.x)
p("rig0_ea_y", r0e.y)
p("rig0_ea_z", r0e.z)
r0q = rig0.q
p("rig0_q_t", r0q.t)
p("rig0_q_x", r0q.v.x)
p("rig0_q_y", r0q.v.y)
p("rig0_q_z", r0q.v.z)
p("rig0_g", rig0.gain.g)
r1e = rig1.ea_u
p("rig1_ea_x", r1e.x)
p("rig1_ea_y", r1e.y)
p("rig1_ea_z", r1e.z)
p("rig1_g", rig1.gain.g)

# Observer termination: False stops the solve.
f8 = fit.Fit()
fill(f8)
cfg8 = fit.LmConfig()
cfg8.max_iters = 50
cfg8.observer = lambda it: False
r8 = f8.solve_dense(cfg8)
pi("obs_stop_status", int(r8.status))
pi("obs_stop_iters", r8.iterations)

# Band solve: kd spans the whole parameter vector.
fitb = fit.Fit()
fill(fitb)
rb2 = fitb.solve_band(4, cfg)
pi("band_status", int(rb2.status))
p("band_end", rb2.end_cost)
p("band_m", fitb.m)
p("band_c", fitb.c)
covb = fitb.assemble_covariance()
pi("band_cov_ok", 1)
sd = covb.std_dev(fitb.items[0])
pi("band_sd_n", len(sd))
p("band_sd_item0", sd[0])

# Stage 3 surface: math types, deque chain with ties through refs,
# arena with a removal, Option entity, fixed euler param.
f3 = fit.Fit()
fill(f3)
f3.cal = (0.25, -0.5)

targets = [(0, 0, 0), (1, 0.5, 0), (2, 1, 0)]
f3.poses.reserve(3)
f3.ties.reserve(2)
p1 = f3.poses.push_back()
p2 = f3.poses.push_back()
p0 = f3.poses.push_front()
ps = [p0, p1, p2]
for i in range(3):
    ps[i].target = targets[i]
    ps[i].pos = (0.1 * i, -0.1 * i, 0.05)
    ps[i].ea = (0.1, 0.2, 0.3 * i)
    ps[i].ea_optimize = False
    th = 0.2 + 0.3 * i
    ps[i].target_dir = (math.cos(th), math.sin(th))
gps = ps[0].info.make_gps()
gps.pos = (7.0, 8.0, 9.0)
gps.isigma = 2.5

t01 = f3.ties.push()
t01.a = f3.poses.ref_at(0)
t01.b = f3.poses.ref_at(1)
t01.d = (1.0, 0.4, 0.0)
t01.w = 3.0
t12 = f3.ties.push()
t12.a = f3.poses.ref_at(1)
t12.b = f3.poses.ref_at(2)
t12.d = (1.0, 0.6, 0.0)
t12.w = 3.0

m0 = f3.marks.push()
m1 = f3.marks.push()
m2 = f3.marks.push()
f3.marks[m0].t = 0.4
f3.marks[m0].w = 1.0
f3.marks[m1].t = 9.0
f3.marks[m1].w = 1.0
f3.marks[m2].t = -0.6
f3.marks[m2].w = 2.0
f3.marks.remove(m1)

pi("s3_clean", 1 if len(f3.validate()) == 0 else 0)
r3 = f3.solve_dense(cfg)
pi("s3_status", int(r3.status))
p("s3_end", r3.end_cost)
p("s3_cal_x", f3.cal.x)
p("s3_cal_y", f3.cal.y)
for i in range(3):
    q = f3.poses[i].pos
    p("s3_p%d_x" % i, q.x)
    p("s3_p%d_y" % i, q.y)
    p("s3_p%d_z" % i, q.z)
    p("s3_p%d_h" % i, f3.poses[i].heading_angle)
p("s3_ea0_z", f3.poses[0].ea.z)
pi("s3_has_gps0", 1 if ps[0].info.gps is not None else 0)
pi("s3_has_gps1", 1 if ps[1].info.gps is not None else 0)
p("s3_gps0_y", ps[0].info.gps.pos.y)
p("s3_gps0_isigma", ps[0].info.gps.isigma)
pi("s3_marks_len", len(f3.marks))
# Iteration over every container kind (the arena skips the removed
# slot).
p("it_obs_sum", sum(o.y for o in f3.obs))
p("it_pose_sum", sum(q.pos.x for q in f3.poses))
it_marks = [mk.t for mk in f3.marks]
p("it_marks_sum", sum(it_marks))
pi("it_marks_n", len(it_marks))
p("it_arrow_sum", sum(o.x for o in f3.obs) + sum(mk.w for mk in f3.marks))
# Backward walks: position-weighted sums pin the ORDER.
p("back_obs", sum((k + 1) * o.y
                  for k, o in enumerate(reversed(list(f3.obs)))))
p("back_marks", sum((k + 1) * t
                    for k, t in enumerate(reversed(it_marks))))
p("r_obs", sum((k + 1) * o.y
               for k, o in enumerate(reversed(list(f3.obs)))))
p("r_marks", sum((k + 1) * t
                 for k, t in enumerate(reversed(it_marks))))
p("s3_mark0_v", f3.marks[m0].v)
p("s3_mark2_v", f3.marks[m2].v)

# Container removal ops on a scratch model.
f4 = fit.Fit()
fill(f4)
f4.obs.pop()
pi("ops_obs_after_pop", len(f4.obs))
f4.obs.truncate(2)
pi("ops_obs_after_trunc", len(f4.obs))
f4.obs.clear()
pi("ops_obs_after_clear", len(f4.obs))
f4.poses.push_back()
f4.poses.push_back()
f4.poses.push_front()
f4.poses.pop_front()
f4.poses.pop_back()
pi("ops_poses_left", len(f4.poses))
pi("ops_pop_empty", 1 if f4.obs.pop() else 0)
f4.marks.push()
f4.marks.push()
f4.marks.clear()
pi("ops_marks_after_clear", len(f4.marks))

# reserve/len/contains/try_get/[0]/[-1] on a scratch model.
f5 = fit.Fit()
f5.obs.reserve(64)
f5.items.reserve(64)
f5.poses.reserve(64)
f5.marks.reserve(64)
pi("cap_obs_empty", 1 if len(f5.obs) == 0 else 0)
i5 = f5.items.push()
i5.t = 0.25
d5 = f5.poses.push_back()
d5.pos = (1.5, 0, 0)
f5.poses.push_back().pos = (2.5, 0, 0)
a5 = f5.marks.push()
f5.marks[a5].t = 0.75
a5b = f5.marks.push()
pi("cap_obs_still_empty", 1 if len(f5.obs) == 0 else 0)
pi("cap_items_nonempty", 1 if len(f5.items) == 0 else 0)
i5r = f5.items.ref_at(0)
pi("cap_items_contains", 1 if i5r in f5.items else 0)
pi("cap_items_contains_default", 1 if fit.NRef() in f5.items else 0)
p("cap_items_try_get", f5.items.try_get(i5r).t)
pi("cap_poses_contains", 1 if f5.poses.ref_at(1) in f5.poses else 0)
p("cap_poses_front_x", f5.poses[0].pos.x)
p("cap_poses_back_x", f5.poses[-1].pos.x)
pi("cap_marks_contains", 1 if a5 in f5.marks else 0)
p("cap_marks_try_get", f5.marks.try_get(a5).t)
f5.marks.remove(a5b)
pi("cap_marks_stale_contains", 1 if a5b in f5.marks else 0)
pi("cap_marks_stale_try_get", 1 if f5.marks.try_get(a5b) is not None else 0)
# End refs and the null sentinel on empty containers.
pi("cap_items_first_valid", 1 if f5.items.first_ref().valid else 0)
p("cap_items_last_get", f5.items.get(f5.items.last_ref()).t)
p("cap_poses_front_ref_x", f5.poses.get(f5.poses.front_ref()).pos.x)
p("cap_poses_back_ref_x", f5.poses.get(f5.poses.back_ref()).pos.x)
f6 = fit.Fit()
pi("cap_empty_first_valid", 1 if f6.items.first_ref().valid else 0)
pi("cap_empty_front_valid", 1 if f6.poses.front_ref().valid else 0)

# Degenerate model: the failure comes back as AraelError.
bad = fit.Fit()
n = bad.items.push()
n.t = 1.0
n.w = 1.0
try:
    bad.solve_dense(cfg)
    pi("bad_status", 0)
    pi("bad_has_error", 0)
except AraelError as e:
    pi("bad_status", e.status)
    pi("bad_has_error", 1 if len(e.message) > 0 else 0)

# A wrapper is keyed, not a cached pointer: growing the collection past its
# capacity must not leave an earlier wrapper reading the old buffer. Checked
# here rather than against the Rust mirror -- Rust indexes its collections
# directly and has nothing to mirror.
f11 = fit.Fit()
held = f11.rigs.push()
held.target_g = 1.75
for _ in range(200):
    f11.rigs.push()
f11.rigs[0].target_g = 9.99          # write through a fresh lookup
assert held.target_g == 9.99, (
    "wrapper went stale after the collection grew: %r" % held.target_g)

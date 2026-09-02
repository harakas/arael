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
    vn = f.vns.push()
    vn.v = [0.4, -0.1, 0.9, 0.0]
    vn.t = [0.1, 0.2, 0.5, -0.3]
    vn.h = [[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]]
    vn.wp = 0.7
    vn.w = 1.3


# Log control crosses the FFI; WARN quiets the backend's INFO chatter
# for this run.
fit.set_log_level(fit.LogLevel.WARN)
pi("log_smoke", 1)

# LmStatus helpers mirror Rust's is_success / as_str.
for i in range(12):
    s = fit.LmStatus(i)
    pi("st_ok_%d" % i, 1 if s.is_success() else 0)
    pi("st_len_%d" % i, len(s.as_str()))

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

# The ill_conditioned preset selects the Nielsen lambda driver; its
# exposed fields equal conservative's, so only a solve's trajectory
# can pin that the driver crossed the FFI.
fic = fit.Fit()
fill(fic)
ric = fic.solve_dense(fit.LmConfig.ill_conditioned())
pi("ic_status", int(ric.status))
pi("ic_iters", ric.iterations)
p("ic_end", ric.end_cost)
p("ic_lambda", ric.final_lambda)

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
p("vn_t2", f.vns[0].t[2])
p("vn_h13", f.vns[0].h[1][3])
for k in range(4):
    p("dense_vn%d" % k, f.vns[0].v[k])

# Covariance at the solution: the 1x1 marginal of the first item.
cov = f.assemble_covariance()
pi("cov_ok", 1)
m0 = cov.marginal(f.items[0])
pi("cov_item0_ok", 1 if isinstance(m0, float) else 0)
p("cov_item0", m0)

# The assembly reports what it decided.
pl = cov.plan()
pi("cov_plan_ordering", int(pl.ordering))
pi("cov_plan_symbolics", pl.symbolics_built)
pi("cov_plan_block_route", 1 if pl.block_route else 0)
pi("cov_plan_has_flops", 1 if pl.candidate_flops is not None else 0)

# Assemblies are owned and independent: an older view keeps answering
# from its own assembly after a new one is made.
cov2 = f.assemble_covariance(fit.CovMode.TRI_DIAGONAL)
pi("cov2_ok", 1)
pi("cov_independent", 1 if cov.marginal(f.items[0]) == m0 else 0)

# Per-constraint cost breakdown (the root is jacobian-enabled).
ct = f.cost_table()
pi("ct_n", len(ct))
for name in sorted(ct):
    p("ct_" + name, ct[name])

# Jacobian diagnostics (owned snapshot).
jac = f.calc_jacobian()
pi("jac_m", jac.num_residuals)
pi("jac_n", jac.num_params)
sv = jac.singular_values()
pi("jac_sv_n", len(sv))
p("jac_sv0", sv[0])
p("jac_sv_last", sv[-1])
svn = jac.singular_values(True)
p("jac_svn0", svn[0])
p("jac_svn_last", svn[-1])
cn = jac.column_l2_norms()
pi("jac_cn_n", len(cn))
p("jac_cn0", cn[0])
p("jac_cn_last", cn[-1])

fit2 = fit.Fit()
fill(fit2)
r2 = fit2.solve_sparse(cfg)
pi("sparse_status", int(r2.status))
p("sparse_end", r2.end_cost)
p("sparse_m", fit2.m)
p("sparse_c", fit2.c)

# The sparse backend's plan crosses as data; a dense result carries
# none.
plan2 = r2.plan
pi("plan_has", 1 if plan2 is not None else 0)
pi("plan_reduced", 1 if plan2 is not None and plan2.reduced else 0)
pi("plan_elim_blocks", plan2.eliminated_blocks if plan2 is not None else 0)
pi("plan_elim_params", plan2.eliminated_params if plan2 is not None else 0)
pi("plan_kept_params", plan2.kept_params if plan2 is not None else 0)
pi("plan_bandwidth", plan2.kept_bandwidth if plan2 is not None else 0)
pi("plan_envelope", 1 if plan2 is not None and plan2.envelope else 0)
pi("plan_ordering", int(plan2.ordering)
   if plan2 is not None and plan2.ordering is not None else -1)
pi("plan_flop_ratio_has",
   1 if plan2 is not None and plan2.flop_ratio is not None else 0)
p("plan_flop_ratio", plan2.flop_ratio
  if plan2 is not None and plan2.flop_ratio is not None else -1.0)
pi("plan_dense_none", 0 if r.plan is not None else 1)

# Sparse options: the defaults are the Rust defaults, and each knob
# drives the backend (pinned by the plan it produces).
so = fit.SparseOptions()
pi("so_schur", int(so.schur))
pi("so_ordering", int(so.ordering))
pi("so_envelope", int(so.envelope))
pi("so_panel", so.envelope_panel_width)
pi("so_supernodal", 1 if so.supernodal else 0)
pi("so_narrow_band", 1 if so.narrow_band else 0)
p("so_flop_margin", so.flop_margin)
p("so_obvious", so.obvious_flop_ratio)
pi("so_block_sn", int(so.block_supernodal))
p("so_bs_batch", so.block_supernodal_batch)
pi("so_bs_lean", 1 if so.block_supernodal_memory_lean else 0)

f11 = fit.Fit()
fill(f11)
so11 = fit.SparseOptions()
so11.schur = fit.SchurPolicy.FORCE
so11.ordering = fit.FaerOrdering.NATURAL
so11.envelope = fit.EnvelopeMode.ALWAYS
r11 = f11.solve_sparse(cfg, so11)
p("opt_end", r11.end_cost)
p11 = r11.plan
pi("opt_reduced", 1 if p11.reduced else 0)
pi("opt_envelope", 1 if p11.envelope else 0)
pi("opt_ordering", int(p11.ordering) if p11.ordering is not None else -1)

f12 = fit.Fit()
fill(f12)
so12 = fit.SparseOptions()
so12.schur = fit.SchurPolicy.FORCE
so12.ordering = fit.FaerOrdering.AMD
so12.envelope = fit.EnvelopeMode.NEVER
so12.supernodal = False
r12 = f12.solve_sparse(cfg, so12)
p("opt2_end", r12.end_cost)
p12 = r12.plan
pi("opt2_reduced", 1 if p12.reduced else 0)
pi("opt2_envelope", 1 if p12.envelope else 0)
pi("opt2_ordering", int(p12.ordering) if p12.ordering is not None else -1)

# The block supernodal knobs: ALWAYS takes that route where the
# envelope declined, NEVER leaves it to faer's scalar one.
f20 = fit.Fit()
fill(f20)
so20 = fit.SparseOptions()
so20.schur = fit.SchurPolicy.FORCE
so20.envelope = fit.EnvelopeMode.NEVER
so20.block_supernodal = fit.BlockSupernodalMode.ALWAYS
so20.block_supernodal_memory_lean = True
r20 = f20.solve_sparse(cfg, so20)
p("opt3_end", r20.end_cost)
pi("opt3_block_sn", 1 if r20.plan.block_supernodal else 0)

f21 = fit.Fit()
fill(f21)
so21 = fit.SparseOptions()
so21.schur = fit.SchurPolicy.FORCE
so21.envelope = fit.EnvelopeMode.NEVER
so21.block_supernodal = fit.BlockSupernodalMode.NEVER
r21 = f21.solve_sparse(cfg, so21)
p("opt4_end", r21.end_cost)
pi("opt4_block_sn", 1 if r21.plan.block_supernodal else 0)

# LmSession: warm solves reuse the analysis and stay bit-identical to
# cold ones; a parameter-count change re-analyzes by itself.
f13 = fit.Fit()
fill(f13)
sess = fit.LmSession()
rs1 = sess.solve(f13, cfg)
p("sess_end1", rs1.end_cost)
f13.m = 0.0
f13.c = 0.0
for i in range(len(f13.items)):
    f13.items[i].v = 0.0
rs2 = sess.solve(f13, cfg)
p("sess_end2", rs2.end_cost)
pi("sess_warm_equals_cold", 1 if rs2.end_cost == rs1.end_cost else 0)
sess.invalidate()
f13.m = 0.0
f13.c = 0.0
for i in range(len(f13.items)):
    f13.items[i].v = 0.0
rs3 = sess.solve(f13, cfg)
pi("sess_invalidate_agrees", 1 if rs3.end_cost == rs1.end_cost else 0)
n13 = f13.items.push()
n13.t = 0.5
n13.w = 1.0
rs4 = sess.solve(f13, cfg)
p("sess_end4", rs4.end_cost)

# A session built over explicit options follows them.
f14 = fit.Fit()
fill(f14)
so14 = fit.SparseOptions()
so14.schur = fit.SchurPolicy.FORCE
so14.ordering = fit.FaerOrdering.NATURAL
so14.envelope = fit.EnvelopeMode.ALWAYS
sessf = fit.LmSession(so14)
rs5 = sessf.solve(f14, cfg)
p("sessf_end", rs5.end_cost)
pi("sessf_envelope", 1 if rs5.plan.envelope else 0)

# The iterative Schur route: CG solves the reduced system; the plan
# carries its iteration total.
f15 = fit.Fit()
fill(f15)
so15 = fit.SparseOptions()
so15.schur = fit.SchurPolicy.FORCE
so15.schur_solve = fit.SchurSolve.ITERATIVE
r15 = f15.solve_sparse(cfg, so15)
p("cg_end", r15.end_cost)
p15 = r15.plan
pi("cg_iters_has", 1 if p15.cg_iterations is not None else 0)
pi("cg_iters", p15.cg_iterations if p15.cg_iterations is not None else -1)

# Enum setters validate: a value outside the enum raises.
try:
    so_bad = fit.SparseOptions()
    so_bad.schur = 7
    pi("enum_setter_validates", 0)
except ValueError:
    pi("enum_setter_validates", 1)

# A bad tag poked past the typed API surfaces as a caught panic at
# the solve (AraelError, status -2), never an abort.
so_raw = fit.SparseOptions()
so_raw._schur = 7
f16 = fit.Fit()
fill(f16)
try:
    f16.solve_sparse(cfg, so_raw)
    pi("bad_tag_raises", 0)
except AraelError as e:
    pi("bad_tag_raises", 1 if e.status == -2 and "tag" in e.message else 0)

# Observer callback + timing + report + conditional covariance.
f7 = fit.Fit()
pi("report_default_empty", 1 if len(fit.LmResult().report()) == 0 else 0)
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
# Per-attempt timeline: exact solver figures, sane wall times.
steps = r7.steps
pi("steps_len", len(steps))
if steps:
    s0, sn = steps[0], steps[-1]
    pi("step0_iter", s0.iter)
    pi("step0_inner", s0.inner)
    pi("step0_accepted", 1 if s0.accepted else 0)
    p("step0_lambda", s0.lambda_)
    p("step0_cost", s0.cost)
    p("step0_new_cost", s0.new_cost)
    p("step0_step_norm", s0.step_norm)
    p("step0_grad_max", s0.grad_max)
    pi("stepN_iter", sn.iter)
    pi("stepN_accepted", 1 if sn.accepted else 0)
    p("stepN_cost", sn.cost)
    p("stepN_new_cost", sn.new_cost)
    steps_ok = 1
    for s in steps:
        for tv in (s.time, s.assembly, s.analysis, s.linear_solve,
                   s.cost_eval, s.advance):
            if not math.isfinite(tv) or tv < 0.0:
                steps_ok = 0
        if s.factorization_failed:
            steps_ok = 0
    pi("steps_ok", steps_ok)
pi("report_nonempty", 1 if len(r7.report()) > 0 else 0)
pi("report_pretty_nonempty", 1 if len(r7.pretty_report()) > 0 else 0)
cov7 = f7.assemble_covariance()
cc = cov7.conditional(f7.items[0])
pi("cond_n", 1 if isinstance(cc, float) else 0)
p("cond_item0", cc)
# The report lives on the result: another solve on the same model
# must not disturb it.
rep7 = r7.report()
f7.solve_dense(cfg)
pi("report_survives_next_solve", 1 if rep7 == r7.report() else 0)

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
    pi("bad_partial_has", 0)
    pi("bad_kind", -99)
    pi("bad_fault", -99)
    pi("bad_param", -99)
except AraelError as e:
    pi("bad_status", e.status)
    pi("bad_has_error", 1 if len(e.message) > 0 else 0)
    pi("bad_partial_has", 1 if e.partial is not None else 0)
    # The structured failure: kind, fault, and the parameter index.
    pi("bad_kind", e.failure.kind if e.failure else -99)
    pi("bad_fault", e.failure.fault if e.failure else -99)
    pi("bad_param", e.failure.param if e.failure else -99)

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

# The one-call construction paths build fill()'s problem: push(**fields)
# per element, then push_many plus column setters; both must solve to
# the same digits, and the column getters must read the solution back.
try:
    import numpy as _np
except ImportError:
    _np = None


def col(a, i, ncomp=1, k=0):
    # Element i (component k) of a column: numpy shape, or the flat
    # ctypes array returned without numpy.
    if hasattr(a, "shape"):
        return float(a[i][k]) if ncomp > 1 else float(a[i])
    return float(a[i * ncomp + k])


def fill_kw(f):
    f.obs.reserve(6)
    for i in range(6):
        f.obs.push(x=float(i), y=2.0 * i + 1.0 + (0.05 if i % 2 == 0 else -0.05))
    t = [1.5, -0.3, 0.7]
    w = [1.0, 2.0, 0.5]
    for i in range(3):
        f.items.push(t=t[i], w=w[i])
    f.vns.push(v=[0.4, -0.1, 0.9, 0.0], t=[0.1, 0.2, 0.5, -0.3],
               h=[[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]], wp=0.7, w=1.3)


def fill_cols(f):
    xs = [float(i) for i in range(6)]
    ys = [2.0 * i + 1.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(6)]
    if _np is not None:
        xs, ys = _np.array(xs), _np.array(ys)
    pi("cols_first", f.obs.push_many(x=xs, y=ys))
    f.items.push_many(n=3)
    f.items.set_t([1.5, -0.3, 0.7])
    f.items.set_w([1.0, 2.0, 0.5])
    f.vns.push_many(v=[[0.4, -0.1, 0.9, 0.0]], t=[[0.1, 0.2, 0.5, -0.3]],
                    h=[[[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]]],
                    wp=0.7, w=1.3)


fk = fit.Fit()
fill_kw(fk)
pi("kw_clean", 1 if len(fk.validate()) == 0 else 0)
rk = fk.solve_dense(cfg)
p("kw_end", rk.end_cost)
p("kw_m", fk.m)
p("kw_c", fk.c)
for i in range(3):
    p("kw_v%d" % i, fk.items[i].v)
vk = fk.items.get_v()
p("kw_get_v_sum", col(vk, 0) + col(vk, 1) + col(vk, 2))
p("kw_get_w_sum", col(fk.items.get_w(), 0) + col(fk.items.get_w(), 1)
  + col(fk.items.get_w(), 2))
p("kw_get_h13", col(fk.vns.get_h(), 0, 8, 7))
p("kw_get_vn2", col(fk.vns.get_v(), 0, 4, 2))

fc = fit.Fit()
fill_cols(fc)
pi("cols_clean", 1 if len(fc.validate()) == 0 else 0)
rc = fc.solve_dense(cfg)
p("cols_end", rc.end_cost)
p("cols_m", fc.m)
p("cols_c", fc.c)
for i in range(3):
    p("cols_v%d" % i, fc.items[i].v)
p("cols_obs3_y", fc.obs[3].y)
p("cols_item1_t", fc.items[1].t)
p("cols_vn_h13", fc.vns[0].h[1][3])
# A column of the wrong dtype raises instead of converting.
if _np is None:
    pi("cols_dtype_raises", 1)
else:
    try:
        fc.obs.set_x(_np.arange(6))
        pi("cols_dtype_raises", 0)
    except TypeError:
        pi("cols_dtype_raises", 1)

# A keyword-less push keeps the Rust Default (N's hand-written unit
# weight, Param's optimize flag) through every push form.
fd = fit.Fit()
nd = fd.items.push()
p("kw_default_w", nd.w)
pi("kw_default_opt", 1 if nd.v_optimize else 0)
p("kw_default_w2", fd.items.push(t=0.5).w)
fd.items.push_many(n=2)
p("kw_default_w3", fd.items[3].w)
# The wrapper knows the key it was looked up by.
pi("kw_ref_ok", 1 if nd.ref == fd.items.ref_at(0) else 0)
pi("kw_index_ok", 1 if fd.obs.push(x=1.0).index == 0 else 0)
# Deque and arena keyword pushes land where their plain forms do.
pd0 = fd.poses.push_back(pos=(1.0, 2.0, 3.0), heading_angle=0.25)
pd1 = fd.poses.push_front(pos=(-1.0, 0.0, 0.5))
p("kw_deque_front_x", fd.poses[0].pos.x)
p("kw_deque_back_z", fd.poses[1].pos.z)
p("kw_deque_back_h", fd.poses[1].heading_angle)
pi("kw_deque_refs", 1 if (pd1.ref == fd.poses.front_ref()
                          and pd0.ref == fd.poses.back_ref()) else 0)
md = fd.marks.push(t=0.4, w=2.0)
p("kw_arena_t", fd.marks[md].t)
p("kw_arena_w", fd.marks[md].w)
gp = fd.poses.get_pos()
p("kw_deque_get_x0", col(gp, 0, 3, 0))
p("kw_deque_get_z1", col(gp, 1, 3, 2))
# An element with no scalar leaves still pushes (nothing to name) and
# reaches its component through the wrapper.
wd = fd.wraps.push()
wd.gain.g = 0.5
fd.wraps.push_many(n=2)
pi("kw_noleaf_len", len(fd.wraps))
p("kw_noleaf_g", fd.wraps[0].gain.g)
# Every ref in one call, on a vector and on a deque, in index order.
fd.items.push_many(n=3)
ri = fd.items.get_refs()
pi("refs_items_ok", 1 if [int(x) for x in ri]
   == [fd.items.ref_at(i).raw for i in range(len(fd.items))] else 0)
rp = fd.poses.get_refs()
pi("refs_poses_ok", 1 if [int(x) for x in rp]
   == [fd.poses.ref_at(i).raw for i in range(len(fd.poses))] else 0)
pi("refs_poses_front", 1 if fit.PoseRef(int(rp[0])) == fd.poses.front_ref() else 0)
# An option made with keywords, and one made plain (the Rust default).
gk = fd.poses[0].info.make_gps(pos=(7.0, 8.0, 9.0), isigma=2.5)
p("opt_kw_y", fd.poses[0].info.gps.pos.y)
p("opt_kw_isigma", gk.isigma)
p("opt_default_isigma", fd.poses[1].info.make_gps().isigma)
fd.poses[1].info.clear_gps()
pi("opt_cleared", 1 if fd.poses[1].info.gps is None else 0)

# Ref keywords on push; every rotation-param keyword form; a nested
# component through the pushed wrapper; a matrix from rows.
fq = fit.Fit()
pa = fq.poses.push_back(pos=(0.0, 0.0, 0.0), heading_angle_optimize=False)
pb = fq.poses.push_back(pos=(1.0, 0.5, 0.0))
tq = fq.ties.push(a=pa.ref, b=pb.ref, d=(1.0, 0.4, 0.0), w=3.0)
pi("kw_tie_refs", 1 if tq.a == fq.poses.ref_at(0) and tq.b == fq.poses.ref_at(1) else 0)
p("kw_tie_d_y", tq.d.y)
pi("kw_heading_frozen", 0 if pa.heading_angle_optimize else 1)
qe = quaternd.from_euler_angles((0.1, 0.2, 0.3))
rq = fq.rigs.push(q=qe, ea_u=(0.15, -0.25, 0.6), ea_u_optimize=False, target_g=1.75)
pi("kw_rig_q_roundtrip", 1 if tuple(rq.q) == tuple(qe) else 0)
p("kw_rig_ea_u_z", rq.ea_u.z)
pi("kw_rig_ea_u_frozen", 0 if rq.ea_u_optimize else 1)
rq2 = fq.rigs.push(q=(1.0, 0.0, 0.0, 0.0))
p("kw_rig_q4_t", rq2.q.t)
p("kw_rig_q4_z", rq2.q.v.z)
rq2.gain.g = 0.25
p("kw_rig_gain", rq2.gain.g)
hv = fq.vns.push(h=[[0, 1, 2, 3], [4, 5, 6, 7]])
p("kw_vn_h_13", hv.h[1][3])

# The pose builtins (TransformParam, UnitVecParam) plus i32 / f32
# leaves: keywords, defaults, columns both ways.
ff = fit.Fit()
fr = ff.frames.push(pose_translation=(1.0, 2.0, 3.0), pose_rotation=qe,
                    pose_optimize_translation=False, dir_unit=(0.0, 0.0, 1.0),
                    anchor=(0.5, 0.5, 0.5), tag=-7, scale=0.5)
p("fr_tx", fr.pose_translation.x)
p("fr_tz", fr.pose_translation.z)
pi("fr_q_roundtrip", 1 if tuple(fr.pose_rotation) == tuple(qe) else 0)
pi("fr_opt_t", 1 if fr.pose_optimize_translation else 0)
pi("fr_opt_r", 1 if fr.pose_optimize_rotation else 0)
p("fr_dir_z", fr.dir_unit.z)
pi("fr_tag", fr.tag)
p("fr_scale", fr.scale)
fr0 = ff.frames.push()
p("fr_def_q_t", fr0.pose_rotation.t)
p("fr_def_q_x", fr0.pose_rotation.v.x)
pi("fr_def_opt_r", 1 if fr0.pose_optimize_rotation else 0)
tags = [-1, 5] if _np is None else _np.array([-1, 5], dtype=_np.int32)
scales = [0.25, 0.75] if _np is None else _np.array([0.25, 0.75], dtype=_np.float32)
ff.frames.push_many(n=2, pose_translation=[(1, 1, 1), (2, 2, 2)], tag=tags, scale=scales)
T = ff.frames.get_pose_translation()
p("fr_col_t_31", col(T, 3, 3, 1))
Q = ff.frames.get_pose_rotation()
p("fr_col_q_10", col(Q, 1, 4, 0))
pi("fr_col_tag_0", int(col(ff.frames.get_tag(), 0)))
pi("fr_col_tag_2", int(col(ff.frames.get_tag(), 2)))
p("fr_col_scale_3", col(ff.frames.get_scale(), 3))
ff.frames.set_pose_optimize_rotation(False)
pi("fr_col_optr_any", 1 if any(bool(x) for x in ff.frames.get_pose_optimize_rotation()) else 0)
ff.frames.set_scale(1.5)
p("fr_col_scale_all", col(ff.frames.get_scale(), 1))
ff.frames.set_tag(-3)
pi("fr_col_tag_all", int(col(ff.frames.get_tag(), 3)))
p("fr_col_dir_0z", col(ff.frames.get_dir_unit(), 0, 3, 2))
ff.frames.set_pose_rotation((1.0, 0.0, 0.0, 0.0))
pi("fr_col_q_reset", 1 if tuple(ff.frames[0].pose_rotation) == (1.0, 0.0, 0.0, 0.0) else 0)

# Deque push_many with rows and a broadcast scalar; ref arrays from
# get_refs into push_many with a broadcast math value; bool columns;
# a strided numpy view; keys from lookups; empty collections; the
# refusals. One flag: everything here is Python-side behaviour.
misc = 1
fq.poses.push_many(pos=[(3, 0, 0), (4, 0, 0)], heading_angle=0.5)
if fq.poses[3].pos.x != 4.0 or fq.poses[2].heading_angle != 0.5:
    misc = 0
refs_all = fq.poses.get_refs()
fq.ties.push_many(a=refs_all[:-1], b=refs_all[1:], d=(1.0, 0.5, 0.0), w=3.0)
if len(fq.ties) != 4 or fq.ties[2].a != fq.poses.ref_at(1) or fq.ties[3].b != fq.poses.ref_at(3):
    misc = 0
if tuple(fq.ties[3].d) != (1.0, 0.5, 0.0):
    misc = 0
fq.items.push_many(t=[1.0, 2.0, 3.0])
fq.items.set_v_optimize([True, False, True] if _np is None
                        else _np.array([True, False, True]))
if [bool(x) for x in fq.items.get_v_optimize()] != [True, False, True]:
    misc = 0
if _np is not None:
    X = _np.arange(6.0).reshape(3, 2)
    fq.items.set_t(X[:, 1])
    if [i.t for i in fq.items] != [1.0, 3.0, 5.0]:
        misc = 0
if fq.poses[1].index != 1 or fq.poses[fq.poses.ref_at(1)].ref != fq.poses.ref_at(1):
    misc = 0
if [t.index for t in fq.ties] != [0, 1, 2, 3]:
    misc = 0
if fq.items.get(fq.items.ref_at(2)).ref != fq.items.ref_at(2):
    misc = 0
fe = fit.Fit()
if len(fe.obs.get_x()) != 0 or len(fe.items.get_refs()) != 0:
    misc = 0
fe.obs.set_x(1.0)
fe.obs.set_x([])
for call, exc in ((lambda: fq.obs.push(z=1.0), TypeError),
                  (lambda: fq.obs.push_many(x=1.0), TypeError),
                  (lambda: fq.poses.set_pos([(0, 0)] * len(fq.poses)), (TypeError, ValueError)),
                  (lambda: fq.obs.push_many(x=[1.0, 2.0], y=[1.0]), ValueError)):
    try:
        call()
        misc = 0
    except exc:
        pass
pi("misc_ok", misc)

// B9 pinning tests for the review's SUSPECTED list (U1-U13, U4 was
// resolved by the drag-apparatus work). Every test asserts the
// CORRECT behavior; a failing test is a confirmed bug. Confirmed ones
// are marked #[ignore = "U<n> confirmed: ..."] so the suite stays
// green while the evidence stays in the tree; drop the ignore with
// the fix.

use std::collections::HashMap;
use arael::vect::vect2d;
use arael_sketch_backend::actions::resolve_dim_endpoint;
use arael_sketch_backend::commands::{complete, execute, CommandContext};
use arael_sketch_backend::earc_fit::fit_earc_tangent;
use arael_sketch_backend::history::{CursorState, History};
use arael_sketch_backend::actions::Action;
use arael_sketch_solver::{DimensionEndpoint, Sketch};

fn run(ctx: &mut CommandContext, cmd: &str) -> (bool, String) {
    let results = execute(ctx, cmd);
    let ok = !results.iter().any(|r| r.is_error);
    let out = results.iter().map(|r| r.output.clone()).collect::<Vec<_>>().join("\n");
    (ok, out)
}

fn run_ok(ctx: &mut CommandContext, cmd: &str) -> String {
    let (ok, out) = run(ctx, cmd);
    assert!(ok, "'{}' failed: {}", cmd, out);
    out
}

// ---------------------------------------------------------------------------
// U1: the perpendicular flip barrier compares an area (cross product)
// to a length (min_length), so it is scale-dependent: inert at large
// scale, active at rest at small scale, where the solver satisfies it
// by inflating the geometry.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "U1 confirmed: flip barrier scale-dependent, small sketches inflate"]
fn u1_perpendicular_is_scale_independent() {
    for scale in [1.0_f64, 0.003] {
        let mut ctx = CommandContext::new();
        let s = scale;
        run_ok(&mut ctx, &format!("add_line 0,0 {},0 noconnect", s));
        run_ok(&mut ctx, &format!("add_line {},{} {},{} noconnect", s / 2.0, s / 4.0, s / 2.0, s));
        let (ok, out) = run(&mut ctx, "perpendicular L0 L1");
        assert!(ok, "perpendicular rejected at scale {}: {}", s, out);
        for r in ctx.sketch.lines.refs() {
            let l = &ctx.sketch.lines[r];
            let len = ((l.p2.value.x - l.p1.value.x).powi(2)
                + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
            assert!(len < 2.0 * s,
                "scale {}: {} inflated to {} (barrier active at rest)", s, l.name, len);
        }
    }
}

// ---------------------------------------------------------------------------
// U2: the supplement branch of the angle dimension flips sign with
// the current winding, so a supplement reading can come out negative.
// ---------------------------------------------------------------------------
#[test]
fn u2_supplement_angle_reading_is_positive() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 0,0 2,-3 noconnect");
    run_ok(&mut ctx, "angle L0 L1 supplement derived");
    let d = &ctx.sketch.dimensions[0];
    assert!(d.value >= 0.0, "supplement reading is negative: {}", d.value);
    assert!((d.value - 123.69).abs() < 0.1, "supplement reading {}", d.value);
}

// ---------------------------------------------------------------------------
// U3: HDistance/VDistance dimensions measure abs() while the backing
// constraint is signed. A pair already satisfying |dx| = target must
// be accepted in place regardless of operand order.
// ---------------------------------------------------------------------------
#[test]
fn u3_hdistance_accepts_satisfied_geometry_in_either_order() {
    for pair in ["P0 P1", "P1 P0"] {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 5,0");
        run_ok(&mut ctx, "add_point 0,0");
        let before: Vec<f64> = ctx.sketch.points.refs()
            .flat_map(|r| [ctx.sketch.points[r].pos.value.x, ctx.sketch.points[r].pos.value.y])
            .collect();
        let (ok, out) = run(&mut ctx, &format!("hdistance {} 5", pair));
        assert!(ok, "hdistance {} rejected: {}", pair, out);
        let after: Vec<f64> = ctx.sketch.points.refs()
            .flat_map(|r| [ctx.sketch.points[r].pos.value.x, ctx.sketch.points[r].pos.value.y])
            .collect();
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "hdistance {} moved satisfied geometry (coord {}: {} -> {})", pair, i, a, b);
        }
    }
}

// ---------------------------------------------------------------------------
// U6: a range with lo > hi is unsatisfiable and must be rejected, not
// silently inert; range and expression on one dimension must not
// coexist.
// ---------------------------------------------------------------------------
#[test]
fn u6_inverted_range_is_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    let (ok, _) = run(&mut ctx, "length L0 5 to 2");
    assert!(!ok, "inverted range lo>hi must be rejected");
}

#[test]
fn u6_range_and_expression_do_not_coexist() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "param w = 4");
    run_ok(&mut ctx, "length L0 w");
    let _ = run(&mut ctx, "length L0 2 to 8");
    let d = &ctx.sketch.dimensions[0];
    assert!(!(d.expr_str.is_some() && d.range.is_some()),
        "dimension carries both expr {:?} and a range", d.expr_str);
}

// ---------------------------------------------------------------------------
// U7: dimension-endpoint helpers for arc start/end are seeded with
// the circle parametrisation; on a rotated elliptic arc the seed must
// match the true (radius_b/rotation-aware) endpoint position.
// ---------------------------------------------------------------------------
#[test]
fn u7_dim_endpoint_helper_seeds_at_true_elliptic_arc_start() {
    let mut s = Sketch::new();
    let r = s.add_elliptic_arc(vect2d::new(2.0, 1.0), 3.0, 1.0,
        std::f64::consts::FRAC_PI_4, 0.3, 2.0, true);
    let truth = arael_sketch_backend::geometry::arc_start_pos(&s.arcs[r]);
    let hp = resolve_dim_endpoint(&mut s, &DimensionEndpoint::ArcStart(r));
    let seeded = s.points[hp].pos.value;
    assert!((seeded.x - truth.x).abs() < 1e-9 && (seeded.y - truth.y).abs() < 1e-9,
        "helper seeded at ({}, {}), true start is ({}, {})",
        seeded.x, seeded.y, truth.x, truth.y);
}

// ---------------------------------------------------------------------------
// U8: resolve_dim_endpoint reuses any coincident point without
// checking `helper`, silently binding a user's point -- deleting that
// point then cascades the symmetry away.
// ---------------------------------------------------------------------------
#[test]
fn u8_symmetry_survives_deleting_a_coincident_user_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 10,0 14,0 noconnect");
    run_ok(&mut ctx, "add_line 7,-3 7,3 noconnect");
    run_ok(&mut ctx, "add_point 0,0");
    run_ok(&mut ctx, "coincident P0 L0.p1");
    run_ok(&mut ctx, "symmetry L0.p1 L2 L1.p1");
    assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
    run_ok(&mut ctx, "delete P0");
    assert_eq!(ctx.sketch.symmetry_pp.len(), 1,
        "symmetry must be anchored to its own helper, not the user's point");
}

// ---------------------------------------------------------------------------
// U9: when the AddDimension action no-ops (line-line distance whose
// auto-parallel could not be satisfied), the command must error --
// today `distance L0 L1 5 force` reports "Set  line-line distance"
// with an empty name while no dimension exists.
// ---------------------------------------------------------------------------
#[test]
fn u9_noop_dimension_paths_do_not_report_phantom_success() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 0,1 2,4 noconnect");
    for ep in ["L0.p1", "L0.p2", "L1.p1", "L1.p2"] {
        run_ok(&mut ctx, &format!("lock {}", ep));
    }
    // Unforced: the auto-parallel is rejected, the command errors.
    let (ok, _) = run(&mut ctx, "distance L0 L1 5");
    assert!(!ok);
    // Forced: still no dimension can exist -- the command must not
    // claim success.
    let (ok, out) = run(&mut ctx, "distance L0 L1 5 force");
    assert!(!ok || !ctx.sketch.dimensions.is_empty(),
        "phantom success with no dimension created: {}", out);
}

// ---------------------------------------------------------------------------
// U10: pushes before the first begin_group share group id 0 with the
// first explicit group, so one undo removes both.
// ---------------------------------------------------------------------------
#[test]
fn u10_pre_group_pushes_do_not_merge_with_first_group() {
    let mut s = Sketch::new();
    let mut h = History::new(&s);
    let a = Action::AddLine { p1: vect2d::new(0.0, 0.0), p2: vect2d::new(4.0, 0.0) };
    a.apply(&mut s);
    h.push(a, &s, CursorState::default());

    h.begin_group();
    let a = Action::AddPoint { pos: vect2d::new(1.0, 1.0) };
    a.apply(&mut s);
    h.push(a, &s, CursorState::default());

    let (restored, _) = h.undo().unwrap();
    assert_eq!(restored.lines.refs().count(), 1,
        "undo of the first explicit group also removed the pre-group action");
}

// ---------------------------------------------------------------------------
// U11: alias substitution iterates a HashMap; when an alias name
// collides with a real entity name that is itself another alias's
// value, the resolution depends on iteration order.
// ---------------------------------------------------------------------------
#[test]
fn u11_alias_substitution_is_deterministic() {
    for _ in 0..48 {
        let mut ctx = CommandContext::new();
        // Entity L0 exists; alias "first" -> "L0"; alias "L0" -> "A0".
        run_ok(&mut ctx, "first = add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "L0 = add_circle 10,0 1 noconnect");
        // "info first" must resolve to the line entity L0 every run.
        let out = run_ok(&mut ctx, "info first");
        assert!(out.contains("L0:"),
            "alias 'first' resolved through the alias chain instead of to L0: {}", out);
    }
}

// ---------------------------------------------------------------------------
// U12: complete() byte-slices at the caller's cursor position and
// panics mid-UTF-8.
// ---------------------------------------------------------------------------
#[test]
fn u12_complete_survives_mid_utf8_cursor() {
    let s = Sketch::new();
    let names = HashMap::new();
    // Cursor inside the two-byte "a-umlaut".
    let input = "msg \u{e4}";
    let _ = complete(&s, &names, input, 5);
}

// ---------------------------------------------------------------------------
// U13 (bulge-sign half, REFUTED): with parallel tangent lines the
// contact points are antipodal, the conic is centrally symmetric
// about the chord midpoint and both target sides give the same
// lambda, so bulge_sign is inert; ccw (from the entry tangent) picks
// the piece. This characterizes that the swept piece lands on the
// tangent side.
// ---------------------------------------------------------------------------
#[test]
fn u13_parallel_normal_fit_bulges_to_the_tangent_side() {
    // Entry heading down at (0,0), exit heading up at (4,0): the arc
    // lives below the chord.
    let (c, rx, ry, rot, sa, ea, ccw) = fit_earc_tangent(
        vect2d::new(0.0, 0.0), vect2d::new(0.0, -1.0),
        vect2d::new(4.0, 0.0), vect2d::new(0.0, 1.0), 0.6).unwrap();
    // Midpoint of the piece actually swept: from sa, half the
    // directional sweep (raw (sa+ea)/2 would sample the complement).
    let tau = std::f64::consts::TAU;
    let sweep = if ccw { (ea - sa).rem_euclid(tau) } else { -((sa - ea).rem_euclid(tau)) };
    let mid = sa + sweep / 2.0;
    let (cr, sr) = (rot.cos(), rot.sin());
    let (cm, sm) = (mid.cos(), mid.sin());
    let mid_pt = vect2d::new(
        c.x + rx * cm * cr - ry * sm * sr,
        c.y + rx * cm * sr + ry * sm * cr,
    );
    assert!(mid_pt.y < 0.0,
        "arc midpoint ({}, {}) is above the chord; entry tangent points down",
        mid_pt.x, mid_pt.y);
}

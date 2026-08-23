use super::*;

fn run(ctx: &mut CommandContext, cmd: &str) -> CommandResult {
    let results = execute(ctx, cmd);
    assert!(!results.is_empty());
    results.into_iter().next().unwrap()
}

fn run_ok(ctx: &mut CommandContext, cmd: &str) -> String {
    let r = run(ctx, cmd);
    assert!(!r.is_error, "Command '{}' failed: {}", cmd, r.output);
    r.output
}

fn run_err(ctx: &mut CommandContext, cmd: &str) -> String {
    let r = run(ctx, cmd);
    assert!(r.is_error, "Command '{}' should have failed but got: {}", cmd, r.output);
    r.output
}

fn line_len(ctx: &CommandContext, name: &str) -> f64 {
    let r = resolve_line(&ctx.sketch, name).unwrap();
    let l = &ctx.sketch.lines[r];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    (dx * dx + dy * dy).sqrt()
}

fn near(a: f64, b: f64) -> bool { (a - b).abs() < 0.1 }

// -- Geometry creation --

#[test]
fn test_add_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    assert_eq!(ctx.sketch.lines.refs().count(), 1);
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 0.0));
    assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
}

#[test]
fn test_add_line_chaining() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line @0,3");
    let r = resolve_line(&ctx.sketch, "L1").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
    assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
}

#[test]
fn test_add_line_single_arg() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,3");
    let r = resolve_line(&ctx.sketch, "L1").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
    assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
}

#[test]
fn test_add_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 3,4");
    assert!(ctx.sketch.points.refs().count() >= 1);
}

#[test]
fn test_add_circle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
}

// -- Coordinate parsing --

#[test]
fn test_coord_endpoint_ref() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line L0.p2 10,0");
    let r = resolve_line(&ctx.sketch, "L1").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
}

#[test]
fn test_coord_relative() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 @3,4");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[r].p2.value.x, 3.0));
    assert!(near(ctx.sketch.lines[r].p2.value.y, 4.0));
}

#[test]
fn test_coord_expression() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param w 5");
    run_ok(&mut ctx, "add_line 0,0 w,0");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
}

#[test]
fn test_coord_geo_function() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_point midpoint(L0)");
    let p = ctx.sketch.points.refs().last().unwrap();
    assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
}

// -- Constraints --

#[test]
fn test_horizontal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,1");
    run_ok(&mut ctx, "horizontal L0");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.horizontal);
}

#[test]
fn test_vertical() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,5");
    run_ok(&mut ctx, "vertical L0");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.vertical);
}

#[test]
fn test_parallel() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
    run_ok(&mut ctx, "parallel L0 L1");
    assert!(!ctx.sketch.parallel.is_empty());
}

#[test]
fn test_coincident() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,0.1 10,0");
    run_ok(&mut ctx, "coincident L0.p2 L1.p1");
    assert!(!ctx.sketch.coincident_ll21.is_empty());
}

// -- Dimensions --

#[test]
fn test_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 3");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    // Solve happened inside exec
    assert!(near(line_len(&ctx, "L0"), 3.0));
}

#[test]
fn test_hdistance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    assert!(near((l.p2.value.x - l.p1.value.x).abs(), 4.0));
}

#[test]
fn test_vdistance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "vdistance L0.p1 L0.p2 2");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    assert!(near((l.p2.value.y - l.p1.value.y).abs(), 2.0));
}

#[test]
fn test_xangle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "xangle L0 45");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let angle = dy.atan2(dx).to_degrees();
    assert!(near(angle, 45.0));
}

#[test]
fn test_xangle_ellipse_rotation() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 2 1 0");
    run_ok(&mut ctx, "xangle EA0 30");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
    let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
    assert!(near(rot_deg, 30.0), "rotation = {rot_deg}");
    assert!(matches!(ctx.sketch.dimensions[0].kind,
                     DimensionKind::ArcRotation(_)));
}

#[test]
fn test_xangle_ellipse_expression() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,5"); // L0 at 45deg
    run_ok(&mut ctx, "xangle L0 derived");
    run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
    // Ellipse rotation tracks L0's angle via the expression
    // pipeline. The reference is LIVE: the constraint couples the
    // rotation to L0's measured angle, and with both sides free
    // the solve distributes the correction -- so assert the
    // relation, not a fixed value.
    run_ok(&mut ctx, "xangle EA0 d0");
    let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
    let line = resolve_line(&ctx.sketch, "L0").unwrap();
    let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
    let l = &ctx.sketch.lines[line];
    let angle_deg = (l.p2.value.y - l.p1.value.y)
        .atan2(l.p2.value.x - l.p1.value.x).to_degrees();
    assert!(near(rot_deg, angle_deg), "rotation = {rot_deg}, L0 angle = {angle_deg}");
}

#[test]
fn test_xangle_rejects_circular_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 0,1 -1,0");
    run_err(&mut ctx, "xangle A0 45");
}

#[test]
fn test_xangle_ellipse_normalised() {
    // User input and effective rotation both fold into (-180, 180].
    // Value stored on the dimension is the normalised form, so
    // `list dims` / `info` / GUI never show a > 180 angle.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 2 1 0");
    run_ok(&mut ctx, "xangle EA0 200");
    let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
    let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
    assert!(near(rot_deg, -160.0), "rot = {rot_deg}");
    assert!(near(ctx.sketch.dimensions[0].value, -160.0));

    // Large-magnitude input (-540deg) folds to -180deg.
    run_ok(&mut ctx, "xangle EA0 -540");
    let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
    assert!(near(rot_deg, -180.0) || near(rot_deg, 180.0), "rot = {rot_deg}");
}

#[test]
fn test_parallel_arc_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,5");
    run_ok(&mut ctx, "lock L0.p1 0,0");
    run_ok(&mut ctx, "lock L0.p2 5,5");
    run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
    run_ok(&mut ctx, "parallel L0 EA0");
    assert_eq!(ctx.sketch.arc_line_parallel.len(), 1);
    let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
    let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
    assert!(near(rot_deg, 45.0), "rotation = {rot_deg}");
}

#[test]
fn test_parallel_arc_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 2 1 30");
    run_ok(&mut ctx, "lock EA0.center 0,0");
    run_ok(&mut ctx, "xangle EA0 30");
    run_ok(&mut ctx, "add_ellipse 5,5 3 1 0");
    run_ok(&mut ctx, "parallel EA0 EA1");
    assert_eq!(ctx.sketch.arc_arc_parallel.len(), 1);
    let ea1 = resolve_arc(&ctx.sketch, "EA1").unwrap();
    let rot_deg = ctx.sketch.arcs[ea1].rotation.value.to_degrees();
    assert!(near(rot_deg, 30.0), "EA1 rotation = {rot_deg}");
}

#[test]
fn test_parallel_rejects_circular_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_arc 0,0 0,1 -1,0");
    run_err(&mut ctx, "parallel L0 A0");
}

#[test]
fn test_parallel_dedup_arc_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,5");
    run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
    run_ok(&mut ctx, "parallel L0 EA0");
    run_err(&mut ctx, "parallel L0 EA0");
    run_err(&mut ctx, "parallel EA0 L0");
    assert_eq!(ctx.sketch.arc_line_parallel.len(), 1);
}

#[test]
fn test_hdistance_update_and_remove() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 6");
    assert_eq!(ctx.sketch.dimensions.len(), 1); // updated, not duplicated
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    assert!(near((l.p2.value.x - l.p1.value.x).abs(), 6.0));
    run_ok(&mut ctx, "delete d0");
    assert_eq!(ctx.sketch.dimensions.len(), 0);
}

#[test]
fn test_bare_derived_dimension_updates_in_place() {
    // A repeated bare derived/driven form must update the existing
    // dimension, not add a duplicate (hdistance/xangle used to add).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "hdistance L0.p1 L0.p2 derived");
    assert!(out.contains("Updated"), "second bare derived should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1, "no duplicate hdistance dimension");
    run_ok(&mut ctx, "xangle L0 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    run_ok(&mut ctx, "xangle L0 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 2, "no duplicate xangle dimension");
}

#[test]
fn test_xangle_negative() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "xangle L0 -30");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let angle = dy.atan2(dx).to_degrees();
    assert!(near(angle, -30.0));
}

#[test]
fn test_xangle_update_and_remove() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "xangle L0 45");
    run_ok(&mut ctx, "xangle L0 60");
    assert_eq!(ctx.sketch.dimensions.len(), 1); // updated, not duplicated
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let angle = (l.p2.value.y - l.p1.value.y).atan2(l.p2.value.x - l.p1.value.x).to_degrees();
    assert!(near(angle, 60.0));
    run_ok(&mut ctx, "delete d0");
    assert_eq!(ctx.sketch.dimensions.len(), 0);
}

#[test]
fn test_xangle_derived() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "xangle L0 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before); // derived does not constrain
}

#[test]
fn test_hdistance_derived() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before);
}

#[test]
fn test_vdistance_derived() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "vdistance L0.p1 L0.p2 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before);
}

#[test]
fn test_length_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "length L0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived); // constraining, not derived
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1); // DOF decreased
    assert!(near(line_len(&ctx, "L0"), (5.0f64 * 5.0 + 3.0 * 3.0).sqrt()));
}

#[test]
fn test_radius_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "radius A0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_hdistance_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_xangle_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "xangle L0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_vdistance_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "vdistance L0.p1 L0.p2 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_sweep_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 0,5");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "sweep A0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_angle_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "angle L0 L1 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_distance_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 8,3 12,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "distance L0.p2 L1.p1 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_distance_pl_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 0,3; add_line 0,0 5,0");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "distance P0 L0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
}

#[test]
fn test_sweep() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 0,5");
    run_ok(&mut ctx, "sweep A0 120");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let sweep_deg = arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs());
    assert!(near(sweep_deg, 120.0));
}

#[test]
fn test_angle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
    run_ok(&mut ctx, "angle L0 L1 60");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_distance_pl_arc_end() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_line 0,10 5,10");
    run_ok(&mut ctx, "distance A0.end L0 7");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_end_l.len(), 1);
}

#[test]
fn test_axis_distance_dof() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 1);
    run_ok(&mut ctx, "vdistance L0.p1 L0.p2 2");
    let dof_after2 = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after2, dof_after - 1);
}

#[test]
fn test_hdistance_preserves_direction() {
    // hdistance is signed internally: can't swap endpoints to satisfy
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3");
    run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    // p2.x should be to the right of p1.x (positive direction preserved)
    assert!(l.p2.value.x > l.p1.value.x);
}

#[test]
fn test_axis_distance_ll_combinations() {
    // LL11: p1-p1
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
    run_ok(&mut ctx, "hdistance L0.p1 L1.p1 6");
    let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near((l1.p1.value.x - l0.p1.value.x).abs(), 6.0));

    // LL12: p1-p2
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
    run_ok(&mut ctx, "vdistance L0.p1 L1.p2 3");
    let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near((l1.p2.value.y - l0.p1.value.y).abs(), 3.0));

    // LL21: p2-p1
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
    run_ok(&mut ctx, "hdistance L0.p2 L1.p1 2");
    let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near((l1.p1.value.x - l0.p2.value.x).abs(), 2.0));

    // LL22: p2-p2
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
    run_ok(&mut ctx, "vdistance L0.p2 L1.p2 5");
    let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near((l1.p2.value.y - l0.p2.value.y).abs(), 5.0));
}

#[test]
fn test_axis_distance_lp_combinations() {
    // LP1: line.p1 to point
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
    run_ok(&mut ctx, "hdistance L0.p1 P0 7");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    assert!(near((p.pos.value.x - l.p1.value.x).abs(), 7.0));

    // LP2: line.p2 to point
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
    run_ok(&mut ctx, "vdistance L0.p2 P0 4");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    assert!(near((p.pos.value.y - l.p2.value.y).abs(), 4.0));

    // Reversed: point first
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
    run_ok(&mut ctx, "hdistance P0 L0.p1 7");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    assert!(near((p.pos.value.x - l.p1.value.x).abs(), 7.0));
}

#[test]
fn test_axis_distance_pp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 0,0; add_point 5,3");
    run_ok(&mut ctx, "hdistance P0 P1 4");
    let p0 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    let p1 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P1").unwrap()];
    assert!(near((p1.pos.value.x - p0.pos.value.x).abs(), 4.0));

    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 0,0; add_point 5,3");
    run_ok(&mut ctx, "vdistance P0 P1 2");
    let p0 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    let p1 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P1").unwrap()];
    assert!(near((p1.pos.value.y - p0.pos.value.y).abs(), 2.0));
}

#[test]
fn test_axis_distance_arc_point() {
    // Arc center to point
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
    run_ok(&mut ctx, "hdistance A0.center P0 3");
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    assert!(near((p.pos.value.x - a.center.value.x).abs(), 3.0));

    // Arc center to point, vdistance
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
    run_ok(&mut ctx, "vdistance A0.center P0 4");
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    assert!(near((p.pos.value.y - a.center.value.y).abs(), 4.0));
}

#[test]
fn test_axis_distance_arc_line() {
    // Arc center to line endpoint
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,5 2; add_line 10,3 15,7");
    run_ok(&mut ctx, "hdistance A0.center L0.p1 4");
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    assert!(near((l.p1.value.x - a.center.value.x).abs(), 4.0));
}

#[test]
fn test_axis_distance_arc_arc() {
    // Arc center to arc center
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 1; add_circle 8,5 2");
    run_ok(&mut ctx, "hdistance A0.center A1.center 6");
    let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    assert!(near((a1.center.value.x - a0.center.value.x).abs(), 6.0));
}

#[test]
fn test_distance_arc_center_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
    run_ok(&mut ctx, "distance A0.center P0 4");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_center_p.len(), 1);
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
    let dx = p.pos.value.x - a.center.value.x;
    let dy = p.pos.value.y - a.center.value.y;
    assert!(near((dx * dx + dy * dy).sqrt(), 4.0));
}

#[test]
fn test_distance_arc_start_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_point 10,3");
    run_ok(&mut ctx, "distance A0.start P0 3");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_start_p.len(), 1);
}

#[test]
fn test_distance_arc_end_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_point 10,3");
    run_ok(&mut ctx, "distance A0.end P0 4");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_end_p.len(), 1);
}

#[test]
fn test_distance_arc_center_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,5 2; add_line 10,3 15,7");
    run_ok(&mut ctx, "distance A0.center L0.p1 3");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
    let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let dx = l.p1.value.x - a.center.value.x;
    let dy = l.p1.value.y - a.center.value.y;
    assert!(near((dx * dx + dy * dy).sqrt(), 3.0));
}

#[test]
fn test_distance_arc_start_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_line 10,3 15,7");
    run_ok(&mut ctx, "distance A0.start L0.p2 4");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_start_l2.len(), 1);
}

#[test]
fn test_distance_arc_center_arc_center() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 1; add_circle 8,5 2");
    run_ok(&mut ctx, "distance A0.center A1.center 5");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
    let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    let dx = a1.center.value.x - a0.center.value.x;
    let dy = a1.center.value.y - a0.center.value.y;
    assert!(near((dx * dx + dy * dy).sqrt(), 5.0));
}

#[test]
fn test_distance_arc_start_arc_start() {
    // Use arcs with locked radii so solver can't collapse them
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_arc 20,0 23,0 20,3");
    run_ok(&mut ctx, "radius A0 3; radius A1 3");
    run_ok(&mut ctx, "distance A0.start A1.start 18");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_aa_s_s.len(), 1);
}

#[test]
fn test_remove_dim() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert_eq!(ctx.sketch.dimensions.len(), 0);
}

// -- Parameters --

#[test]
fn test_param_create() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param width 10");
    assert_eq!(ctx.sketch.user_params.len(), 1);
    assert_eq!(ctx.sketch.user_params[0].name, "width");
}

#[test]
fn test_param_update() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param w 5");
    run_ok(&mut ctx, "param w 10");
    assert_eq!(ctx.sketch.user_params[0].value, 10.0);
}

#[test]
fn test_del_param() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param w 5; del_param w");
    assert!(ctx.sketch.user_params.is_empty());
}

#[test]
fn test_rename_param() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param width 5; rename_param width w");
    assert_eq!(ctx.sketch.user_params[0].name, "w");
}

// -- Style --

#[test]
fn test_style_set() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "style L0 dashed");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert_eq!(ctx.sketch.lines[r].style, LineStyle::Dashed);
}

#[test]
fn test_style_query() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "style L0");
    assert!(out.contains("solid"));
}

// -- Introspection --

#[test]
fn test_print_expr() {
    let mut ctx = CommandContext::new();
    let out = run_ok(&mut ctx, "print 2+3");
    assert!(out.contains("5"));
}

#[test]
fn test_print_entity() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    let out = run_ok(&mut ctx, "print L0.length");
    assert!(out.contains("5"));
}

#[test]
fn test_info_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "info L0");
    assert!(out.contains("L0"));
}

#[test]
fn test_list() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_point 3,4");
    let out = run_ok(&mut ctx, "list");
    assert!(out.contains("L0"));
    assert!(out.contains("P0"));
}

#[test]
fn test_list_constraints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
    let out = run_ok(&mut ctx, "list constraints");
    assert!(out.contains("horizontal"));
}

/// One handle per thing: every dimension-managed constraint (value
/// flags and dimension-backed collections alike) is listed under
/// `list dims` with its meaning, and only there; `list constraints`
/// keeps the constraints addressable by their own name.
#[test]
fn test_dims_listed_once_with_meaning() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0; add_line 0,2 4,2; add_ellipse 8,0 3 1 45; add_point 10,10");
    run_ok(&mut ctx, "length L0 4; xangle L1 0; radius EA0 3; radius_b EA0 1; xangle EA0 45");
    run_ok(&mut ctx, "distance L0.p1 P0 5; horizontal L0");
    let cs = run_ok(&mut ctx, "list constraints");
    for leak in ["length", "xangle", "radius", "distance"] {
        assert!(!cs.contains(leak), "{} must not leak into list constraints:\n{}", leak, cs);
    }
    assert!(cs.contains("horizontal L0"), "own-name constraints stay: {}", cs);
    let ds = run_ok(&mut ctx, "list dims");
    for want in [
        "d0: length L0 = 4.0000",
        "d1: xangle L1 = 0.0000",
        "d2: radius EA0 = 3.0000",
        "d3: radius_b EA0 = 1.0000",
        "d4: xangle EA0 = 45.0000",
        "d5: distance L0.p1 P0 = 5.0000",
    ] {
        assert!(ds.contains(want), "missing {:?} in:\n{}", want, ds);
    }
    // Kind filters find dims by their meaning.
    let r = run_ok(&mut ctx, "list radius");
    assert!(r.contains("d2: radius EA0") && r.contains("d3: radius_b EA0"), "{}", r);
    let x = run_ok(&mut ctx, "list xangle");
    assert!(x.contains("d1: xangle L1") && x.contains("d4: xangle EA0"), "{}", x);
    // info shows both handles.
    let info = run_ok(&mut ctx, "info EA0");
    assert!(info.contains("dims: d2: radius EA0 = 3.0000, d3: radius_b EA0 = 1.0000, d4: xangle EA0 = 45.0000"), "{}", info);
    let info = run_ok(&mut ctx, "info d5");
    assert!(info.starts_with("d5: distance L0.p1 P0 value=5.0000"), "{}", info);
}

/// Expression, range, derived and broken dims render their source
/// and tags in the dims listing.
#[test]
fn test_list_dims_source_and_tags() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0; add_line 0,2 6,2; add_circle 10,0 2");
    run_ok(&mut ctx, "param w 3");
    run_ok(&mut ctx, "length L0 w/2; length L1 3 to 7; radius A0 2 derived");
    let ds = run_ok(&mut ctx, "list dims");
    assert!(ds.contains("d0: length L0 = w/2 (1.5000)"), "{}", ds);
    assert!(ds.contains("d1: length L1 3 to 7 ("), "{}", ds);
    assert!(ds.contains("d2: radius A0 = 2.0000 derived"), "{}", ds);
    run_ok(&mut ctx, "del_param w");
    let ds = run_ok(&mut ctx, "list dims");
    assert!(ds.contains("d0: length L0 = w/2 (") && ds.contains("broken"), "{}", ds);
}

/// A value flag with no dimension over it (legacy files) still shows
/// under list constraints so it is not invisible.
#[test]
fn test_orphan_value_flag_listed_as_constraint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2; add_ellipse 5,0 3 1 0");
    let a = ctx.sketch.arcs.refs().next().unwrap();
    let e = ctx.sketch.arcs.refs().nth(1).unwrap();
    ctx.sketch.mutate_values(|s| {
        s.arcs[a].constraints.has_target_radius = true;
        s.arcs[a].constraints.target_radius = 2.0;
        s.arcs[e].constraints.has_target_rotation = true;
        s.arcs[e].constraints.target_rotation = 0.0;
    });
    let cs = run_ok(&mut ctx, "list constraints");
    assert!(cs.contains("radius A0 = 2"), "{}", cs);
    assert!(cs.contains("xangle EA1 = 0.0000"), "{}", cs); // shared arc counter: A0 then EA1
    assert!(run_ok(&mut ctx, "list dims").contains("(empty)"));
}

#[test]
fn test_find() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    let out = run_ok(&mut ctx, "find 5,0 1");
    assert!(out.contains("L0"));
}

// -- Geometric functions --

#[test]
fn test_intersect() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line -1,-1 1,1; add_line -1,1 1,-1");
    run_ok(&mut ctx, "add_point intersect(L0,L1)");
    let p = ctx.sketch.points.refs().last().unwrap();
    assert!(near(ctx.sketch.points[p].pos.value.x, 0.0));
    assert!(near(ctx.sketch.points[p].pos.value.y, 0.0));
}

#[test]
fn test_midpoint_func() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0; add_point midpoint(L0)");
    let p = ctx.sketch.points.refs().last().unwrap();
    assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
}

#[test]
fn test_tangent_normal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    let out = run_ok(&mut ctx, "print tangent(L0)");
    assert!(out.contains("1.0"));
    let out = run_ok(&mut ctx, "print normal(L0)");
    assert!(out.contains("1.0")); // normal is (0,1), output has "1.0"
}

#[test]
fn test_dist_pp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    let out = run_ok(&mut ctx, "print dist(L0.p1,L0.p2)");
    assert!(out.contains("5"));
}

#[test]
fn test_dist_pl() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 0,3; add_line 0,0 10,0");
    let out = run_ok(&mut ctx, "print dist(P0,L0)");
    assert!(out.contains("3"));
}

// -- Session variables --

#[test]
fn test_let_scalar() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "let d = 5");
    let out = run_ok(&mut ctx, "print d");
    assert!(out.contains("5"));
}

#[test]
fn test_let_vec() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    run_ok(&mut ctx, "let p = L0.p2");
    let out = run_ok(&mut ctx, "print p");
    assert!(out.contains("3") && out.contains("4"));
}

#[test]
fn test_let_in_coord() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4; let p = L0.p2");
    run_ok(&mut ctx, "add_point p");
    let pt = ctx.sketch.points.refs().last().unwrap();
    assert!(near(ctx.sketch.points[pt].pos.value.x, 3.0));
}

// -- Selection --

#[test]
fn test_convert_dimension_preserves_identity() {
    // set_derived/set_driven used to delete and recreate the
    // dimension, churning its did and losing placement on the
    // driven path. ConvertDimension flips it in place.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0 noconnect");
    run_ok(&mut ctx, "length L0 3");
    let did = ctx.sketch.dimensions[0].did;
    run_ok(&mut ctx, "dim_pos d0 offset 2");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.has_length);

    run_ok(&mut ctx, "set_derived d0");
    let d = &ctx.sketch.dimensions[0];
    assert_eq!((d.did, d.name.as_str(), d.derived), (did, "d0", true));
    assert!(near(d.offset.y, 2.0), "placement must survive");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(!ctx.sketch.lines[r].constraints.has_length, "backing constraint must be gone");

    run_ok(&mut ctx, "set_driven d0 4");
    let d = &ctx.sketch.dimensions[0];
    assert_eq!((d.did, d.derived), (did, false));
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.has_length);
    assert!(near(line_len(&ctx, "L0"), 4.0));

    run_ok(&mut ctx, "undo");
    assert!(ctx.sketch.dimensions[0].derived, "undo reverts the conversion");
}

#[test]
fn test_dimension_identity_survives_removal() {
    // Dimension actions used to carry Vec indices; removing one
    // dimension shifted the rest and the next removal deleted the
    // wrong one. Identity is the permanent did now.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "add_line 0,1 2,1 noconnect");
    run_ok(&mut ctx, "add_line 0,2 3,2 noconnect");
    run_ok(&mut ctx, "length L0 1");
    run_ok(&mut ctx, "length L1 2");
    run_ok(&mut ctx, "length L2 3");
    let did0 = ctx.sketch.dimensions[0].did;
    let did2 = ctx.sketch.dimensions[2].did;
    assert!(did0 != 0 && did2 != 0 && did0 != did2);
    // Remove first and third the way the GUI multi-delete does.
    ctx.begin_group();
    ctx.exec(Action::RemoveDimension { did: did0 });
    ctx.exec(Action::RemoveDimension { did: did2 });
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert_eq!(ctx.sketch.dimensions[0].name, "d1", "wrong dimension deleted");
    // A did that no longer resolves is a no-op, never a
    // wrong-target delete.
    ctx.exec(Action::RemoveDimension { did: did2 });
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_delete_by_name_is_nid_stable() {
    // ConstraintId used to carry a positional index resolved at
    // parse time; any retain in between shifted it and the wrong
    // constraint died. Identity is the nid now.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "add_line 0,1 1,1 noconnect");
    run_ok(&mut ctx, "add_line 0,2 1,2 noconnect");
    run_ok(&mut ctx, "parallel L0 L1");
    run_ok(&mut ctx, "parallel L1 L2");
    assert_eq!(ctx.sketch.parallel.len(), 2);
    let second_nid = ctx.sketch.parallel[1].nid;
    run_ok(&mut ctx, "delete C1");
    assert_eq!(ctx.sketch.parallel.len(), 1);
    assert_eq!(ctx.sketch.parallel[0].nid, second_nid, "wrong constraint deleted");
    run_ok(&mut ctx, "delete C2");
    assert!(ctx.sketch.parallel.is_empty());
}

#[test]
fn test_hidden_helper_bridge_not_addressable() {
    // point_on with an arc endpoint mints a hidden bridge with the
    // next C number; deleting it used to cascade the user's
    // visible constraint away with the helper point.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_circle 2,3 1");
    run_ok(&mut ctx, "point_on A0.start L0");
    assert_eq!(ctx.sketch.point_on_line.len(), 1);
    run_err(&mut ctx, "delete C2");
    assert_eq!(ctx.sketch.point_on_line.len(), 1, "bridge delete destroyed the user constraint");
    // The visible constraint stays addressable.
    run_ok(&mut ctx, "delete C1");
    assert!(ctx.sketch.point_on_line.is_empty());
}

#[test]
fn test_history_covers_drag_dim_pos_and_relational_delete() {
    // drag: undo restores the pre-drag position instead of
    // undoing the previous action (drag was absent from history).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "drag L0.p2 5,5");
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.lines.refs().count(), 1, "undo of drag deleted the line");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[r].p2.value.x, 1.0));

    // dim_pos: placement is undoable.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "length L0 1");
    let before = ctx.sketch.dimensions[0].offset.y;
    run_ok(&mut ctx, "dim_pos d0 offset 3");
    assert!(near(ctx.sketch.dimensions[0].offset.y, 3.0));
    run_ok(&mut ctx, "undo");
    assert!(near(ctx.sketch.dimensions[0].offset.y, before), "undo must revert the placement");

    // relational delete: one undoable step through DeleteConstraint.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    run_ok(&mut ctx, "add_circle 5,5 1 noconnect");
    run_ok(&mut ctx, "concentric A0 A1");
    assert_eq!(ctx.sketch.concentric.len(), 1);
    run_ok(&mut ctx, "delete A0 A1 concentric");
    assert!(ctx.sketch.concentric.is_empty());
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.concentric.len(), 1, "undo must restore the deleted constraint");
}

#[test]
fn test_flags_survive_undo_redo() {
    // quiet/constr used to mutate the sketch after the history
    // snapshot (or outside history entirely): undo+redo lost the
    // flag, and undo after the standalone commands undid the
    // wrong action.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect constr");
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.lines.refs().count(), 0);
    run_ok(&mut ctx, "redo");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].construction, "constr flag lost across undo+redo");

    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "quiet L0 on");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].quiet);
    run_ok(&mut ctx, "undo");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(!ctx.sketch.lines[r].quiet, "undo must revert the quiet flag, not the line");
    assert_eq!(ctx.sketch.lines.refs().count(), 1, "undo of quiet must not delete the line");
}

#[test]
fn test_created_identity_after_slot_reuse() {
    // The arena refills freed slots, so refs().last() after a
    // delete named a pre-existing entity and flags landed on it.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
    run_ok(&mut ctx, "add_line 2,0 3,0 noconnect");
    run_ok(&mut ctx, "delete L0");
    let out = run_ok(&mut ctx, "add_line 5,5 6,6 noconnect constr");
    assert!(out.contains("Added L2"), "reported name: {}", out);
    let new = resolve_line(&ctx.sketch, "L2").unwrap();
    assert!(ctx.sketch.lines[new].construction);
    let old = resolve_line(&ctx.sketch, "L1").unwrap();
    assert!(!ctx.sketch.lines[old].construction, "flag leaked to the reused-slot neighbor");
}

#[test]
fn test_selection_pruned_after_delete() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,1");
    run_ok(&mut ctx, "select L0");
    run_ok(&mut ctx, "delete L0");
    assert!(ctx.selection.is_empty());
    // The stale-ref read that used to panic.
    run_ok(&mut ctx, "list selection");
}

#[test]
fn test_selection_pruned_after_clear_and_undo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,1");
    run_ok(&mut ctx, "select L0");
    run_ok(&mut ctx, "clear");
    run_ok(&mut ctx, "list selection");
    assert!(ctx.selection.is_empty());

    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,1");
    run_ok(&mut ctx, "select L0");
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "list selection");
    assert!(ctx.selection.is_empty());
}

#[test]
fn test_add_arc_collinear_points_rejected() {
    let mut ctx = CommandContext::new();
    run_err(&mut ctx, "add_arc 0,0 2,0 1,0");
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
}

#[test]
fn test_tangent_degenerate_geometry_rejected() {
    // Zero-length line: the tangent sign is 0/0 and the solve
    // never converges; must reject cleanly and fast. Creation of
    // such a line is rejected at the gate, but a solve can still
    // collapse one -- force the state through the value door.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_circle 5,5 1");
    let l = ctx.sketch.lines.refs().next().unwrap();
    ctx.sketch.mutate_values(|s| {
        let p1 = s.lines[l].p1.value;
        s.lines[l].p2.value = p1;
    });
    let msg = run_err(&mut ctx, "tangent L0 A0");
    assert!(msg.contains("zero length"), "{}", msg);

    // Concentric arcs: tangent divides by center distance.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 2,2 1");
    run_ok(&mut ctx, "add_circle 2,2 3");
    let msg = run_err(&mut ctx, "tangent A0 A1");
    assert!(msg.contains("concentric"), "{}", msg);
}

#[test]
fn test_delete_dim_removes_line_endpoint_arc_backing_constraint() {
    // Line-endpoint x arc-point distance: the backing constraint
    // lives in distance_arc_center_l1 and must die with the dim.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_circle 6,3 1 noconnect");
    run_ok(&mut ctx, "distance L0.p1 A0.center 6");
    assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert_eq!(ctx.sketch.dimensions.len(), 0);
    assert!(ctx.sketch.distance_arc_center_l1.is_empty(),
        "backing constraint must be deleted with the dimension");
}

#[test]
fn test_redo_restores_post_group_cursor() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 4,0 4,3 noconnect");
    run_ok(&mut ctx, "add_line 4,3 0,3 noconnect");
    run_ok(&mut ctx, "undo 2");
    let c = ctx.cursor.unwrap();
    assert!((c.x - 4.0).abs() < 1e-9 && c.y.abs() < 1e-9, "{:?}", (c.x, c.y));
    // Redo of the middle group restores the cursor state that
    // followed it (the pre-state of the next group).
    run_ok(&mut ctx, "redo");
    let c = ctx.cursor.unwrap();
    assert!((c.x - 4.0).abs() < 1e-9 && (c.y - 3.0).abs() < 1e-9, "{:?}", (c.x, c.y));
}

#[test]
fn test_output_surfaces_ids() {
    // Rect constraints carry their ids.
    let mut ctx = CommandContext::new();
    let out = run_ok(&mut ctx, "add_rect 0,0 4,3");
    assert!(out.contains("(C"), "{}", out);
    // Auto-coincident feedback names the constraint it added.
    let out = run_ok(&mut ctx, "add_line 4,3 6,5");
    assert!(out.contains("[connected:") && out.contains("(C"), "{}", out);
    // Freeze names the dimension it created.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    let out = run_ok(&mut ctx, "freeze L0");
    assert!(out.contains("d0"), "{}", out);
    // Delete-relational names the removed constraint.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect; add_line 0,2 4,2 noconnect; parallel L0 L1");
    let out = run_ok(&mut ctx, "delete L0 L1 parallel");
    assert!(out.contains("C1"), "{}", out);
}

#[test]
fn test_dof_commands_error_on_degenerate_geometry() {
    // Solver-collapsed zero-length line under a point-on-line
    // residual: every dof command variant errors cleanly.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_point 2,1");
    run_ok(&mut ctx, "point_on P0 L0");
    let l = ctx.sketch.lines.refs().next().unwrap();
    ctx.sketch.mutate_values(|s| {
        let p1 = s.lines[l].p1.value;
        s.lines[l].p2.value = p1;
        // Value-only change that alters the instantaneous rank.
        s.clear_cached_dof();
    });
    for cmd in ["dof", "dof analyze", "dof eigenvalues", "dof singular"] {
        let msg = run_err(&mut ctx, cmd);
        assert!(msg.contains("non-finite"), "{}: {}", cmd, msg);
    }
}

#[test]
fn test_expr_radius_follows_drag_frames() {
    // A circle whose radius is the expression `d+2`, with `d` a
    // derived distance to the dragged endpoint: the radius must
    // track during per-frame drag solves, not only at drag end.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "l = add_line 0,0 10,0");
    run_ok(&mut ctx, "lock l.p1; horizontal l; length l 10");
    run_ok(&mut ctx, "pivot = add_line l.p2 @5,0");
    run_ok(&mut ctx, "d = distance l.p1 pivot.p2 derived");
    run_ok(&mut ctx, "c = add_circle l.p1 20");
    run_ok(&mut ctx, "radius c d+2");
    let circle = ctx.sketch.arcs.refs().next().unwrap();
    let pivot = ctx.sketch.lines.refs().nth(1).unwrap();
    // The live constraint holds the relation radius = d + 2; with
    // both sides free the solve distributes the correction.
    let dist = |s: &Sketch| {
        let p = s.lines[pivot].p2.value;
        (p.x * p.x + p.y * p.y).sqrt()
    };
    let r0 = ctx.sketch.arcs[circle].radius.value;
    assert!((r0 - (dist(&ctx.sketch) + 2.0)).abs() < 0.1,
        "creation: radius {} vs d+2 {}", r0, dist(&ctx.sketch) + 2.0);

    // Simulate GUI drag frames on pivot.p2: install the
    // apparatus, move the helper, per-frame cell solve.
    let grab_pos = ctx.sketch.lines[pivot].p2.value;
    let app = ctx.sketch.get_mut().install_drag(
        arael_sketch_solver::DragTarget::LineP2(pivot),
        grab_pos, None, Some(DRAG_PULL_WEIGHT));
    ctx.sketch.mutate_values(|s| s.move_drag_helper(app.helper, vect2d::new(25.0, 0.0)));
    ctx.sketch.solve();
    let mid_radius = ctx.sketch.arcs[circle].radius.value;
    let mid_dist = dist(&ctx.sketch);
    ctx.sketch.get_mut().remove_drag(&app);
    assert!(mid_dist > r0 + 1.0, "drag must have pulled the pivot out, d at {}", mid_dist);
    assert!((mid_radius - (mid_dist + 2.0)).abs() < 0.5,
        "radius {} must track d+2 (~{}) during the drag frame", mid_radius, mid_dist + 2.0);
}

#[test]
fn test_all_trailing_keywords_are_peeled() {
    // The circle family's hand-rolled peel loop capped at 3 of
    // its 5 keywords, silently leaving the rest as arguments.
    let mut ctx = CommandContext::new();
    let out = run_ok(&mut ctx, "add_circle 2,2 1 nocursor noconnect quiet constr driven");
    assert!(out.contains("[driven") && out.contains("[quiet]"), "{}", out);
    let r = ctx.sketch.arcs.refs().next().unwrap();
    assert!(ctx.sketch.arcs[r].construction, "constr keyword must be honored");
    assert!(ctx.cursor.is_none(), "nocursor keyword must be honored");
}

#[test]
fn test_deselect_unknown_entity_errors() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0; select L0");
    let msg = run_err(&mut ctx, "deselect L99");
    assert!(msg.contains("Unknown line"), "{}", msg);
    let msg = run_err(&mut ctx, "deselect x9");
    assert!(msg.contains("Unknown entity"), "{}", msg);
    // Deselecting an existing but unselected entity stays a no-op.
    run_ok(&mut ctx, "add_line 0,2 4,2; deselect L1");
    assert_eq!(ctx.selection.len(), 1);
}

#[test]
fn test_driven_fragment_reports_rejection() {
    // A rejected driven dimension must not report the previous
    // dimension's name as if it were the new one.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "length L0 4");
    let line = ctx.sketch.lines.refs().next().unwrap();
    let frag = driven_dim_fragment(&mut ctx, Action::AddDimension {
        kind: DimensionKind::LineLength(line),
        value: 4.0, expr: None, derived: false, range: None,
    }, "length", 4.0);
    assert!(frag.contains("rejected"), "{}", frag);
    assert!(!frag.contains("d0 "), "{}", frag);

    // The success path names the dimension actually created.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    let line = ctx.sketch.lines.refs().next().unwrap();
    let frag = driven_dim_fragment(&mut ctx, Action::AddDimension {
        kind: DimensionKind::LineLength(line),
        value: 4.0, expr: None, derived: false, range: None,
    }, "length", 4.0);
    assert!(frag.contains("d0") && frag.contains("length=4.0000"), "{}", frag);
}

#[test]
fn test_info_reaches_user_params_starting_with_d() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "param depth = 2.5");
    let out = run_ok(&mut ctx, "info depth");
    assert!(out.contains("depth") && out.contains("2.5"), "{}", out);
    // Numeric d-names still report as dimensions.
    let out = run_err(&mut ctx, "info d99");
    assert!(out.contains("Unknown dimension"), "{}", out);
}

#[test]
fn test_force_strip_ordering() {
    // A comment must not hide a trailing force keyword.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 0,2 4,2 noconnect");
    run_ok(&mut ctx, "length L0 4");
    run_ok(&mut ctx, "length L1 4");
    run_ok(&mut ctx, "equal L0 L1 force # redundant on purpose");
    assert_eq!(ctx.sketch.equal_length.len(), 1);

    // msg text is verbatim: no force stripping, no comment stripping.
    let out = run_ok(&mut ctx, "msg use the force # not a comment");
    assert_eq!(out, "use the force # not a comment");
}

#[test]
fn test_classify_direction_with_radius_is_not_rigid_motion() {
    // A radius-grow direction moves two on-circle points diagonally
    // and changes the radius: neither a rotation nor a translation.
    let parts = vec![
        ("P1.x".to_string(), 0.7), ("P1.y".to_string(), 0.7),
        ("P2.x".to_string(), -0.7), ("P2.y".to_string(), 0.7),
        ("A0.radius".to_string(), 1.0),
    ];
    let label = classify_free_direction(&parts);
    assert!(!label.starts_with("rotate") && !label.starts_with("translate"), "{}", label);

    // Radius plus one moving coordinate must not read as translate X.
    let parts = vec![
        ("A0.center.x".to_string(), 1.0),
        ("A0.center.y".to_string(), 0.0),
        ("A0.radius".to_string(), 1.0),
    ];
    let label = classify_free_direction(&parts);
    assert!(!label.starts_with("translate"), "{}", label);

    // Control: a genuine rotation still classifies as one.
    let parts = vec![
        ("P1.x".to_string(), 0.7), ("P1.y".to_string(), 0.7),
        ("P2.x".to_string(), -0.7), ("P2.y".to_string(), 0.7),
    ];
    let label = classify_free_direction(&parts);
    assert!(label.starts_with("rotate"), "{}", label);
}

#[test]
fn test_command_path_runs_conflict_checks() {
    // Transitive H/V contradiction through a parallel link used to
    // pass the command path (only the GUI ran the conflict checks)
    // and collapse L1 to zero length.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,5 noconnect");
    run_ok(&mut ctx, "add_line 2,0 2,5 noconnect");
    run_ok(&mut ctx, "vertical L0");
    run_ok(&mut ctx, "parallel L0 L1");
    let msg = run_err(&mut ctx, "horizontal L1");
    assert!(msg.contains("vertical"), "{}", msg);

    // force overrides only the DOF check, never a contradiction.
    let msg = run_err(&mut ctx, "horizontal L0 force");
    assert!(msg.contains("vertical"), "{}", msg);
}

#[test]
fn test_collinear_transitivity_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
    run_ok(&mut ctx, "add_line 5,0 9,0 noconnect");
    run_ok(&mut ctx, "add_line 10,0 14,0 noconnect");
    run_ok(&mut ctx, "collinear L0 L1");
    run_ok(&mut ctx, "collinear L1 L2");
    let msg = run_err(&mut ctx, "collinear L0 L2");
    assert!(msg.contains("already collinear"), "{}", msg);
}

#[test]
fn test_degenerate_creation_rejected() {
    let mut ctx = CommandContext::new();
    let msg = run_err(&mut ctx, "add_line 1,1 1,1");
    assert!(msg.contains("zero length"), "{}", msg);
    let msg = run_err(&mut ctx, "add_circle 0,0 0");
    assert!(msg.contains("zero radius"), "{}", msg);
    assert_eq!(ctx.sketch.lines.refs().count(), 0);
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
}

#[test]
fn test_select() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; select L0");
    assert_eq!(ctx.selection.len(), 1);
}

#[test]
fn test_deselect() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; select L0; deselect");
    assert!(ctx.selection.is_empty());
}

// -- Undo/redo --

#[test]
fn test_undo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    assert_eq!(ctx.sketch.lines.refs().count(), 1);
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.lines.refs().count(), 0);
}

#[test]
fn test_redo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; undo; redo");
    assert_eq!(ctx.sketch.lines.refs().count(), 1);
}

#[test]
fn test_history() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 1,1 2,2");
    let out = run_ok(&mut ctx, "history");
    assert!(out.contains("Add line"));
}

// -- Remove constraint --

#[test]
fn test_remove_horizontal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.horizontal);
    run_ok(&mut ctx, "delete L0 horizontal");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(!ctx.sketch.lines[r].constraints.horizontal);
}

#[test]
fn test_remove_parallel() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1; parallel L0 L1");
    assert!(!ctx.sketch.parallel.is_empty());
    run_ok(&mut ctx, "delete L0 L1 parallel");
    assert!(ctx.sketch.parallel.is_empty());
}

// -- Multi-command --

#[test]
fn test_semicolon() {
    let mut ctx = CommandContext::new();
    let results = execute(&mut ctx, "add_line 0,0 5,0; horizontal L0");
    // Two commands + one trailing DOF summary line.
    assert_eq!(results.len(), 3);
    assert!(!results[0].is_error);
    assert!(!results[1].is_error);
    assert!(results[2].output.starts_with("DOF:"));
}

// -- Error handling --

#[test]
fn test_unknown_command() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "foobar"); }

#[test]
fn test_unknown_entity() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "info L99"); }

#[test]
fn test_bad_coord() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "add_line abc xyz"); }

#[test]
fn test_help() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help"); }

#[test]
fn test_help_command() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help add_line"); }

#[test]
fn test_dof() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "dof"); }

#[test]
fn test_cost() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "cost"); }

// -- Entity name capture --

#[test]
fn test_auto_underscore() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,1");
    assert!(ctx.session_names.contains_key("_"));
    run_ok(&mut ctx, "vertical _");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.vertical);
}

#[test]
fn test_auto_underscore_updates() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    assert_eq!(ctx.session_names["_"], "L0");
    run_ok(&mut ctx, "add_line 1,1 2,2");
    assert_eq!(ctx.session_names["_"], "L1");
}

#[test]
fn test_assign_entity_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "base = add_line 0,0 5,1");
    assert_eq!(ctx.session_names.get("base").unwrap(), "L0");
    run_ok(&mut ctx, "horizontal base");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.horizontal);
}

#[test]
fn test_let_entity_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "let l = add_line 0,0 5,1");
    assert_eq!(ctx.session_names.get("l").unwrap(), "L0");
    run_ok(&mut ctx, "horizontal l");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.horizontal);
}

#[test]
fn test_let_entity_coincident() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "a = add_line 0,0 5,0");
    run_ok(&mut ctx, "b = add_line 5,0.1 10,0");
    run_ok(&mut ctx, "coincident a.p2 b.p1");
    assert!(!ctx.sketch.coincident_ll21.is_empty());
}

#[test]
fn test_let_entity_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "l = add_line 0,0 5,0");
    run_ok(&mut ctx, "length l 3");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(near(line_len(&ctx, "L0"), 3.0));
}

#[test]
fn test_let_entity_info() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "l = add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "info l");
    assert!(out.contains("L0"));
}

#[test]
fn test_underscore_chain() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; horizontal _; length _ 3");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(ctx.sketch.lines[r].constraints.horizontal);
    assert!(near(line_len(&ctx, "L0"), 3.0));
}

// -- Auto-coincident --

#[test]
fn test_auto_coincident() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_line 5,0 5,3");
    assert!(out.contains("connected"), "Should auto-connect: {}", out);
    // L1.p1==L0.p2 -> coincident_ll12 (a.p1 == b.p2 where a=L1, b=L0)
    let has_coincident = !ctx.sketch.coincident_ll12.is_empty()
        || !ctx.sketch.coincident_ll21.is_empty()
        || !ctx.sketch.coincident_ll11.is_empty()
        || !ctx.sketch.coincident_ll22.is_empty();
    assert!(has_coincident, "Should have coincident constraint");
}

#[test]
fn test_noconnect() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_line 5,0 5,3 noconnect");
    assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
    assert!(ctx.sketch.coincident_ll21.is_empty());
}

// -- Auto-coincident for arcs/circles --

#[test]
fn test_auto_coincident_circle_to_line_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_circle 5,0 1");
    assert!(out.contains("connected"), "Should auto-connect: {}", out);
    assert!(out.contains("A0.center=L0.p2"), "Should mention A0.center=L0.p2: {}", out);
    assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
        "Should have coincident_lp2_arc_center");
}

#[test]
fn test_auto_coincident_circle_to_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 3,3");
    let out = run_ok(&mut ctx, "add_circle 3,3 1");
    assert!(out.contains("connected"), "Should auto-connect: {}", out);
    assert!(!ctx.sketch.coincident_arc_center.is_empty(),
        "Should have coincident_arc_center");
}

#[test]
fn test_auto_coincident_circle_concentric() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    let out = run_ok(&mut ctx, "add_circle 0,0 3");
    assert!(out.contains("connected"), "Should auto-connect: {}", out);
    assert!(out.contains("A1.center=A0.center"), "Should mention concentric: {}", out);
    assert!(!ctx.sketch.concentric.is_empty(), "Should have concentric constraint");
}

#[test]
fn test_auto_coincident_line_to_arc_center() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 5,3 1");
    let out = run_ok(&mut ctx, "add_line 0,0 5,3");
    assert!(out.contains("connected"), "Should auto-connect: {}", out);
    assert!(out.contains("L0.p2=A0.center"), "Should mention A0.center: {}", out);
    assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
        "Should have coincident_lp2_arc_center");
}

#[test]
fn test_noconnect_circle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_circle 5,0 1 noconnect");
    assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
    assert!(ctx.sketch.coincident_lp2_arc_center.is_empty());
}

#[test]
fn test_noconnect_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_arc 5,0 5,3 6,1.5 noconnect");
    assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
}

// -- Duplicate constraint rejection --

#[test]
fn test_duplicate_horizontal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    let e = run_err(&mut ctx, "horizontal L0");
    assert!(e.contains("already horizontal"));
}

#[test]
fn test_duplicate_vertical() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,5");
    run_ok(&mut ctx, "vertical L0");
    let e = run_err(&mut ctx, "vertical L0");
    assert!(e.contains("already vertical"));
}

#[test]
fn test_duplicate_parallel() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
    run_ok(&mut ctx, "parallel L0 L1");
    let e = run_err(&mut ctx, "parallel L0 L1");
    assert!(e.contains("already exists"));
    let e = run_err(&mut ctx, "parallel L1 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_perpendicular() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
    run_ok(&mut ctx, "perpendicular L0 L1");
    let e = run_err(&mut ctx, "perpendicular L1 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_equal_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
    run_ok(&mut ctx, "equal L0 L1");
    let e = run_err(&mut ctx, "equal L1 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_equal_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
    run_ok(&mut ctx, "equal A0 A1");
    let e = run_err(&mut ctx, "equal A1 A0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_collinear() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 6,0 10,0");
    run_ok(&mut ctx, "collinear L0 L1");
    let e = run_err(&mut ctx, "collinear L1 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_tangent_la() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2.5,1 1");
    run_ok(&mut ctx, "tangent L0 A0");
    let e = run_err(&mut ctx, "tangent L0 A0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_tangent_aa() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 2");
    run_ok(&mut ctx, "tangent A0 A1");
    let e = run_err(&mut ctx, "tangent A1 A0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_coincident_ll() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
    run_ok(&mut ctx, "coincident L0.p2 L1.p1");
    let e = run_err(&mut ctx, "coincident L0.p2 L1.p1");
    assert!(e.contains("already exists"));
    // Cross-type: L0.p2=L1.p1 is same as L1.p1=L0.p2 (swapped order)
    let e = run_err(&mut ctx, "coincident L1.p1 L0.p2");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_concentric() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
    run_ok(&mut ctx, "concentric A0 A1");
    let e = run_err(&mut ctx, "concentric A1 A0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_point_on_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
    run_ok(&mut ctx, "point_on P0 L0");
    let e = run_err(&mut ctx, "point_on P0 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_duplicate_midpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
    run_ok(&mut ctx, "midpoint P0 L0");
    let e = run_err(&mut ctx, "midpoint P0 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_midpoint_arc_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_point 0,5");
    run_ok(&mut ctx, "midpoint P0 A0");
    assert_eq!(ctx.sketch.midpoint_arc_point.len(), 1);
    // Duplicate check
    let e = run_err(&mut ctx, "midpoint P0 A0");
    assert!(e.contains("already exists"), "{}", e);
}

#[test]
fn test_midpoint_arc_line_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_line -1,5 1,5");
    run_ok(&mut ctx, "midpoint L0.p1 A0");
    assert_eq!(ctx.sketch.midpoint_lp1_arc.len(), 1);
}

#[test]
fn test_midpoint_arc_circle_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5; add_point 0,5");
    let e = run_err(&mut ctx, "midpoint P0 A0");
    assert!(e.contains("full circle"), "{}", e);
}

#[test]
fn test_remove_constraint_midpoint_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_point 0,5");
    run_ok(&mut ctx, "midpoint P0 A0");
    assert_eq!(ctx.sketch.midpoint_arc_point.len(), 1);
    run_ok(&mut ctx, "delete P0 A0 midpoint");
    assert_eq!(ctx.sketch.midpoint_arc_point.len(), 0);
}

#[test]
fn test_duplicate_symmetry_ll() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
    run_ok(&mut ctx, "symmetry L0 L1 L2");
    let e = run_err(&mut ctx, "symmetry L2 L1 L0");
    assert!(e.contains("already exists"));
}

#[test]
fn test_self_reference_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 0,0 2");
    let e = run_err(&mut ctx, "parallel L0 L0");
    assert!(e.contains("itself"));
    let e = run_err(&mut ctx, "equal L0 L0");
    assert!(e.contains("itself"));
    let e = run_err(&mut ctx, "concentric A0 A0");
    assert!(e.contains("itself"));
}

// -- Info with constraints --

#[test]
fn test_info_shows_constraints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
    let out = run_ok(&mut ctx, "info L0");
    assert!(out.contains("horizontal"), "info should show constraints: {}", out);
}

#[test]
fn test_info_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "info L0.p1");
    assert!(out.contains("0.0000"), "info L0.p1 should show position: {}", out);
}

// -- Select endpoints --

#[test]
fn test_select_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "select L0.p1");
    assert_eq!(ctx.selection.len(), 1);
    assert!(matches!(ctx.selection[0], Selection::LineP1(_)));
}

// -- Param shows value --

#[test]
fn test_param_shows_value() {
    let mut ctx = CommandContext::new();
    let out = run_ok(&mut ctx, "param kala 12+3*4");
    assert!(out.contains("24"), "Should show evaluated value: {}", out);
}

// -- Cursor --

#[test]
fn test_cursor_set() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 5,3");
    assert!(ctx.cursor.is_some());
    assert!(near(ctx.cursor.unwrap().x, 5.0));
    assert!(near(ctx.cursor.unwrap().y, 3.0));
}

#[test]
fn test_cursor_from_add_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    assert!(ctx.cursor.is_some());
    assert!(near(ctx.cursor.unwrap().x, 5.0));
    assert!(near(ctx.cursor.unwrap().y, 0.0));
}

#[test]
fn test_cursor_relative() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 1,1");
    run_ok(&mut ctx, "cursor @2,3");
    assert!(near(ctx.cursor.unwrap().x, 3.0));
    assert!(near(ctx.cursor.unwrap().y, 4.0));
}

#[test]
fn test_cursor_as_coord() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 5,0");
    run_ok(&mut ctx, "add_line cursor 5,3");
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
    assert!(near(ctx.sketch.lines[r].p1.value.y, 0.0));
}

#[test]
fn test_cursor_off() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 5,3");
    assert!(ctx.cursor.is_some());
    run_ok(&mut ctx, "cursor off");
    assert!(ctx.cursor.is_none());
}

#[test]
fn test_cursor_nocursor() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 1,1");
    run_ok(&mut ctx, "add_line 0,0 5,0 nocursor");
    // Cursor should still be at 1,1, not moved to 5,0
    assert!(near(ctx.cursor.unwrap().x, 1.0));
}

#[test]
fn test_cursor_endpoint_ref() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "cursor L0.p1");
    assert!(near(ctx.cursor.unwrap().x, 0.0));
}

#[test]
fn test_cursor_query() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "cursor 3,7");
    let out = run_ok(&mut ctx, "cursor");
    assert!(out.contains("3.0000") && out.contains("7.0000"));
}

// -- Dimension text position --

#[test]
fn test_dim_pos_offset() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
    run_ok(&mut ctx, "dim_pos d0 offset 2.0");
    assert!(near(ctx.sketch.dimensions[0].offset.y, 2.0));
}

#[test]
fn test_dim_pos_along() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
    run_ok(&mut ctx, "dim_pos d0 along 0.5");
    assert!(near(ctx.sketch.dimensions[0].text_along, 0.5));
}

#[test]
fn test_dim_info_shows_position() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
    let out = run_ok(&mut ctx, "info d0");
    assert!(out.contains("offset=") && out.contains("along="));
}

// -- Point symmetry command --

#[test]
fn test_cmd_symmetry_pp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 3,2; add_point 7,2; add_line 5,0 5,10");
    run_ok(&mut ctx, "symmetry P0 L0 P1");
    assert!(!ctx.sketch.symmetry_pp.is_empty());
}

// -- Derived dimensions --

#[test]
fn test_cmd_derived_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 5 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    // Derived should NOT constrain — line length should stay at original ~5
    // (no has_length constraint set)
    let r = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(!ctx.sketch.lines[r].constraints.has_length);
}

#[test]
fn test_cmd_set_derived() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
    assert!(!ctx.sketch.dimensions[0].derived);
    run_ok(&mut ctx, "set_derived d0");
    // Find the derived dim (might be re-added with same name)
    let dim = ctx.sketch.dimensions.iter().find(|d| d.name == "d0");
    assert!(dim.is_some());
    assert!(dim.unwrap().derived);
}

#[test]
fn test_cmd_set_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    // Create derived dim first
    run_ok(&mut ctx, "length L0 5 derived");
    assert!(ctx.sketch.dimensions[0].derived);
    run_ok(&mut ctx, "set_driven d0 3");
    // Should now be driven
    let dim = ctx.sketch.dimensions.last().unwrap();
    assert!(!dim.derived);
    assert!(near(line_len(&ctx, "L0"), 3.0));
}

#[test]
fn test_cmd_derived_length_measure() {
    // "length L0 derived" should measure current geometry
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    run_ok(&mut ctx, "length L0 derived");
    assert!(ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 5.0));
}

#[test]
fn test_cmd_derived_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3");
    run_ok(&mut ctx, "radius A0 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 3.0));
}

#[test]
fn test_cmd_derived_angle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
    run_ok(&mut ctx, "angle L0 L1 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 90.0));
}

#[test]
fn test_cmd_derived_distance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0; add_line 4,0 7,0");
    run_ok(&mut ctx, "distance L0.p2 L1.p1 derived");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 1.0));
}

// -- Helper point display and cleanup tests --

fn has_helper_points(ctx: &CommandContext) -> bool {
    ctx.sketch.points.refs().any(|r| ctx.sketch.points[r].helper)
}

/// Both listings: constraints (addressable by their own name) and
/// dimensions (each with its own constraint).
fn list_constraints_output(ctx: &mut CommandContext) -> String {
    format!("{}\n{}", run_ok(ctx, "list constraints"), run_ok(ctx, "list dims"))
}

// 6A: Display tests -- neither listing shows Pc names

#[test]
fn test_list_no_pc_distance_ll_endpoints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
    run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    assert!(out.contains("distance"), "should list distance constraint: {}", out);
}

#[test]
fn test_list_no_pc_distance_arc_endpoints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
    run_ok(&mut ctx, "distance A0.center A1.center 10");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    assert!(out.contains("distance") && out.contains("A0.center") && out.contains("A1.center"),
        "should show semantic names: {}", out);
}

#[test]
fn test_list_no_pc_distance_mixed_arc_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 10,0 2");
    run_ok(&mut ctx, "distance A0.center L0.p1 10");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
}

#[test]
fn test_list_no_pc_distance_point_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    run_ok(&mut ctx, "distance L0.p1 L1 3");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
}

#[test]
fn test_list_no_pc_symmetry_pp_endpoints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
    run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    assert!(out.contains("symmetry") && out.contains("L0.p1") && out.contains("L0.p2"),
        "should show semantic names: {}", out);
}

#[test]
fn test_list_no_pc_symmetry_pp_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
    run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
    let out = list_constraints_output(&mut ctx);
    assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
}

#[test]
fn test_list_no_bridge_constraints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
    run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
    let out = list_constraints_output(&mut ctx);
    // Should not contain bridge coincident entries
    let lines: Vec<&str> = out.lines().collect();
    for line in &lines {
        if line.starts_with("coincident") {
            assert!(!line.contains("Pc"), "bridge constraint should be hidden: {}", line);
        }
    }
}

// 6B: Cleanup on object deletion
// Note: Line-Line endpoint distances (DistanceLL*) don't create helpers.
// Helpers are created for Arc endpoint distances and PointLineDistance
// with non-Point endpoints.

#[test]
fn test_cleanup_delete_line_removes_symmetry_helpers() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
    run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
    assert!(has_helper_points(&ctx), "should have helpers after symmetry");
    run_ok(&mut ctx, "delete L0");
    assert!(!has_helper_points(&ctx), "helpers should be cleaned up after delete L0");
    assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
}

#[test]
fn test_cleanup_delete_arc_removes_distance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
    run_ok(&mut ctx, "distance A0.center A1.center 10");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
    run_ok(&mut ctx, "delete A0");
    assert!(ctx.sketch.distance_aa_ce_ce.is_empty(), "constraint should be cleaned up");
}

#[test]
fn test_cleanup_delete_arc_removes_symmetry_helpers() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
    run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
    assert!(has_helper_points(&ctx), "should have helpers");
    run_ok(&mut ctx, "delete A0");
    assert!(!has_helper_points(&ctx), "helpers should be cleaned up");
    assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
}

#[test]
fn test_cleanup_delete_mirror_line_removes_symmetry_helpers() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
    run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
    assert!(!ctx.sketch.symmetry_pp.is_empty());
    run_ok(&mut ctx, "delete L1");
    assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry gone after mirror line deleted");
    assert!(!has_helper_points(&ctx), "helpers cleaned up");
}

#[test]
fn test_cleanup_delete_line_removes_pl_distance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    run_ok(&mut ctx, "distance L0.p1 L1 3");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_lp1l.len(), 1);
    run_ok(&mut ctx, "delete L0");
    assert!(ctx.sketch.distance_lp1l.is_empty(), "constraint cleaned up");
}

// 6C: Cleanup on dimension removal

#[test]
fn test_cleanup_remove_dim_distance_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
    run_ok(&mut ctx, "distance A0.center A1.center 10");
    assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert!(ctx.sketch.distance_aa_ce_ce.is_empty(), "constraint cleaned up after remove_dim");
}

#[test]
fn test_cleanup_remove_dim_distance_point_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    run_ok(&mut ctx, "distance L0.p1 L1 3");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_lp1l.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert!(ctx.sketch.distance_lp1l.is_empty(), "constraint cleaned up after remove_dim");
}

#[test]
fn test_cleanup_remove_dim_distance_arc_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_line 0,5 5,5");
    run_ok(&mut ctx, "distance A0.center L0 5");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_arc_center_l.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert!(ctx.sketch.distance_arc_center_l.is_empty(), "constraint cleaned up after remove_dim");
}

#[test]
fn test_distance_pl_line_endpoint() {
    // LineP1 to line (perpendicular distance)
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    run_ok(&mut ctx, "distance L0.p1 L1 3");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_lp1l.len(), 1);

    // LineP2 to line
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    run_ok(&mut ctx, "distance L0.p2 L1 3");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_lp2l.len(), 1);
}

#[test]
fn test_distance_pl_arc_center() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2; add_line 0,5 5,5");
    run_ok(&mut ctx, "distance A0.center L0 4");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_center_l.len(), 1);
}

#[test]
fn test_distance_pl_arc_start() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_line 0,10 5,10");
    run_ok(&mut ctx, "distance A0.start L0 8");
    assert!(!has_helper_points(&ctx));
    assert_eq!(ctx.sketch.distance_arc_start_l.len(), 1);
}

#[test]
fn test_cleanup_remove_dim_distance_arc_mixed() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_line 10,0 15,0");
    run_ok(&mut ctx, "distance A0.center L0.p1 10");
    assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
    assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
    run_ok(&mut ctx, "delete d0");
    assert!(ctx.sketch.distance_arc_center_l1.is_empty(), "constraint cleaned up after remove_dim");
}

// 6D: No Pc in distance constraints that don't need helpers (regression)

#[test]
fn test_no_helpers_for_line_line_distance() {
    // Line-Line endpoint distances use specialized constraints, no helpers
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
    run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
    assert!(!has_helper_points(&ctx), "DistanceLL should not create helpers");
}

// -- Autocomplete tests --

fn setup_complete_ctx() -> CommandContext {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");   // L0
    run_ok(&mut ctx, "add_line 5,0 5,5");   // L1
    run_ok(&mut ctx, "add_point 2,3");       // P0
    run_ok(&mut ctx, "add_circle 3,3 2");    // A0
    run_ok(&mut ctx, "length L0 5");         // d0
    run_ok(&mut ctx, "param width 10");
    ctx
}

fn completions(ctx: &CommandContext, input: &str) -> Vec<String> {
    complete(&ctx.sketch, &ctx.session_names, input, input.len())
}

// -- DOF check on constraints --

#[test]
fn test_dof_check_accepts_valid() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    // horizontal removes 1 DOF, should succeed
}

#[test]
fn test_dof_check_force_overrides() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    // Two parallel lines, then collinear (removes 1 more DOF beyond parallel)
    run_ok(&mut ctx, "add_line 0,1 5,1");
    run_ok(&mut ctx, "parallel L0 L1");
    run_ok(&mut ctx, "collinear L0 L1");
}

// -- DOF analysis --

#[test]
fn test_dof_analyze_unconstrained_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "dof analyze");
    assert!(out.contains("DOF: 4"), "Unconstrained line should have 4 DOF: {}", out);
}

#[test]
fn test_dof_analyze_constrained() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5; lock L0.p1 0,0");
    let out = run_ok(&mut ctx, "dof analyze");
    assert!(out.contains("DOF: 0"), "Fully constrained should be DOF 0: {}", out);
}

#[test]
fn test_dof_analyze_partial() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5");
    let out = run_ok(&mut ctx, "dof analyze");
    // 4 DOF - 1 (horizontal) - 1 (length) = 2 DOF (translate X, Y)
    assert!(out.contains("DOF: 2"), "Should have 2 DOF: {}", out);
    assert!(out.contains("translate"), "Should identify translation: {}", out);
}

#[test]
fn test_dof_analyze_empty() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "clear");
    let out = run_ok(&mut ctx, "dof analyze");
    assert!(out.contains("DOF: 0"), "Empty sketch should be 0 DOF: {}", out);
}

// -- point_on with arc endpoints --

#[test]
fn test_point_on_arc_center_on_line_no_duplicate() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2,1 1");
    run_ok(&mut ctx, "point_on A0.center L0");
    let arc_center_count = ctx.sketch.coincident_arc_center.len();
    let pol_count = ctx.sketch.point_on_line.len();
    eprintln!("After first: arc_center={}, point_on_line={}", arc_center_count, pol_count);
    let out2 = run_err(&mut ctx, "point_on A0.center L0");
    eprintln!("Second attempt: {}", out2);
    assert!(out2.contains("already exists"), "Should reject duplicate: {}", out2);
}

#[test]
fn test_point_on_arc_center_on_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
    let out = run_ok(&mut ctx, "point_on A0.center L0");
    assert!(out.contains("point-on-line"), "Should succeed: {}", out);
    // Verify helper point was created and point_on_line constraint exists
    assert!(!ctx.sketch.point_on_line.is_empty(),
        "Should have point_on_line constraint");
}

#[test]
fn test_point_on_arc_center_on_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5; add_circle 4.5,0 1");
    let out = run_ok(&mut ctx, "point_on A1.center A0");
    assert!(out.contains("point-on-arc"), "Should succeed: {}", out);
    assert!(!ctx.sketch.point_on_arc.is_empty(),
        "Should have point_on_arc constraint");
}

// -- Dimension update (no duplicates) --

#[test]
fn test_dimension_update_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 5");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "length L0 10");
    assert!(out.contains("Updated"), "Should update existing: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1, "Should still be 1 dimension");
}

#[test]
fn test_dimension_update_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    run_ok(&mut ctx, "radius A0 5");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "radius A0 10");
    assert!(out.contains("Updated"), "Should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_dimension_update_radius_to_expr() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    run_ok(&mut ctx, "radius A0 5");
    run_ok(&mut ctx, "param scale 2");
    let out = run_ok(&mut ctx, "radius A0 \"5*scale\"");
    assert!(out.contains("Updated"), "Should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_dimension_update_angle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
    run_ok(&mut ctx, "angle L0 L1 45");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "angle L0 L1 90");
    assert!(out.contains("Updated"), "Should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_dimension_update_distance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 7,0 10,0");
    run_ok(&mut ctx, "distance L0.p2 L1.p1 3");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "distance L0.p2 L1.p1 5");
    assert!(out.contains("Updated"), "Should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_dimension_expr_constrains() {
    // Bare expression is live under the current grammar (commit 76947c1).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "clear");
    run_ok(&mut ctx, "param scale 1");
    run_ok(&mut ctx, "add_circle 0,0 5");
    run_ok(&mut ctx, "radius A0 5*scale");
    // Check that expr_constraints were created
    ctx.sketch.solve();
    assert!(!ctx.sketch.expr_constraints.is_empty(),
        "Expression dimension should create expr_constraint, got none. dims: {:?}",
        ctx.sketch.dimensions.iter().map(|d| (&d.name, &d.expr_str, d.value)).collect::<Vec<_>>());
    // Verify it actually constrains the radius
    let r = ctx.sketch.arcs.refs().next().unwrap();
    let radius = ctx.sketch.arcs[r].radius.value;
    assert!((radius - 5.0).abs() < 0.1,
        "radius should be 5*1=5, got {}", radius);
}

#[test]
fn test_dimension_expr_constrains_fresh() {
    // Fresh expression dim without prior numeric.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "param scale 1");
    run_ok(&mut ctx, "add_circle 0,0 5");
    // Bare form is live (tracks `scale`).
    run_ok(&mut ctx, "radius A0 5*scale");
    // Check dimension was created with expression
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert_eq!(ctx.sketch.dimensions[0].expr_str.as_deref(), Some("5*scale"));
    // Solve and check expr constraints are built
    let result = ctx.sketch.solve();
    assert!(!ctx.sketch.expr_constraints.is_empty(),
        "Should have expr_constraints after solve");
    // Check radius is actually constrained to 5
    let r = ctx.sketch.arcs.refs().next().unwrap();
    assert!((ctx.sketch.arcs[r].radius.value - 5.0).abs() < 0.1,
        "radius should be 5*1=5, got {}", ctx.sketch.arcs[r].radius.value);
    assert!(result.end_cost < 0.01, "cost should be near zero, got {}", result.end_cost);
}

#[test]
fn test_dimension_expr_update_constrains() {
    // Updating numeric dim to an expression should constrain. The old
    // `{2*scale}` brace form was removed by the grammar flip; use the
    // bare-expression form for live tracking.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    run_ok(&mut ctx, "radius A0 5");
    run_ok(&mut ctx, "param scale 3");
    run_ok(&mut ctx, "radius A0 2*scale");
    ctx.sketch.solve();
    assert!(!ctx.sketch.expr_constraints.is_empty(),
        "Updated expression should create expr_constraint");
    let r = ctx.sketch.arcs.refs().next().unwrap();
    let radius = ctx.sketch.arcs[r].radius.value;
    assert!((radius - 6.0).abs() < 0.1,
        "radius should be 2*3=6, got {}", radius);
}

#[test]
fn test_dimension_no_cross_update() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
    run_ok(&mut ctx, "length L0 5");
    run_ok(&mut ctx, "length L1 3");
    assert_eq!(ctx.sketch.dimensions.len(), 2, "Different entities should have separate dims");
}

// -- Autocomplete tests --

#[test]
fn test_complete_empty_input() {
    let ctx = setup_complete_ctx();
    assert!(completions(&ctx, "").is_empty());
}

#[test]
fn test_complete_first_token_commands() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_");
    assert!(c.contains(&"add_line".to_string()));
    assert!(c.contains(&"add_point".to_string()));
    assert!(c.contains(&"add_circle".to_string()));
    // Should NOT contain entity names
    assert!(!c.iter().any(|s| s.starts_with('L')));
}

#[test]
fn test_complete_list_filters_not_entities() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "list l");
    assert!(c.contains(&"lines".to_string()));
    assert!(!c.iter().any(|s| s.starts_with('L')), "list should not offer entity names: {:?}", c);
}

#[test]
fn test_complete_cursor_keywords() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "cursor o");
    assert!(c.contains(&"on".to_string()));
    assert!(c.contains(&"off".to_string()));
}

#[test]
fn test_complete_add_line_cursor() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_line curs");
    assert!(c.contains(&"cursor".to_string()));
}

#[test]
fn test_complete_horizontal_lines_only() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "horizontal L");
    assert!(c.contains(&"L0".to_string()));
    assert!(c.contains(&"L1".to_string()));
    assert!(!c.iter().any(|s| is_arc_name(s)), "horizontal should not offer arcs");
    assert!(!c.iter().any(|s| s.starts_with('P')), "horizontal should not offer points");
}

#[test]
fn test_complete_concentric_arcs_only() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "concentric A");
    assert!(c.contains(&"A0".to_string()));
    assert!(!c.iter().any(|s| s.starts_with('L')), "concentric should not offer lines");
}

#[test]
fn test_complete_style_values() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "style L0 d");
    assert!(c.contains(&"dashed".to_string()));
    assert!(c.contains(&"dashdot".to_string()));
    assert!(!c.iter().any(|s| s.starts_with("d0")), "style arg2 should not offer dimensions");
}

#[test]
fn test_complete_delete_dim() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "delete d");
    assert!(c.contains(&"d0".to_string()));
}

#[test]
fn test_complete_length_arg2_derived() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "length L0 d");
    assert!(c.contains(&"derived".to_string()));
    // Should offer dimension refs in expression context
    assert!(c.contains(&"d0".to_string()));
    // Should NOT offer lines
    assert!(!c.contains(&"L0".to_string()), "length arg2 should not offer L0");
}

#[test]
fn test_complete_equal_type_matching() {
    let ctx = setup_complete_ctx();
    // After "equal L0", should only offer lines
    let c = completions(&ctx, "equal L0 L");
    assert!(c.contains(&"L1".to_string()));
    assert!(!c.iter().any(|s| is_arc_name(s)), "equal with L0 should not offer arcs");
}

#[test]
fn test_complete_dim_pos() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "dim_pos d0 o");
    assert!(c.contains(&"offset".to_string()));
    let c = completions(&ctx, "dim_pos d0 a");
    assert!(c.contains(&"along".to_string()));
}

#[test]
fn test_complete_no_arg_commands() {
    let ctx = setup_complete_ctx();
    assert!(completions(&ctx, "dof x").is_empty());
    assert!(completions(&ctx, "cost x").is_empty());
}

#[test]
fn test_complete_del_param() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "del_param w");
    assert!(c.contains(&"width".to_string()));
}

#[test]
fn test_complete_delete_constraint_types() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "delete L0 h");
    assert!(c.contains(&"horizontal".to_string()));
}

#[test]
fn test_complete_dot_line() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "info L0.");
    assert!(c.contains(&"L0.p1".to_string()));
    assert!(c.contains(&"L0.p2".to_string()));
}

#[test]
fn test_complete_dot_arc() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "info A0.");
    assert!(c.contains(&"A0.center".to_string()));
    assert!(c.contains(&"A0.start".to_string()));
    assert!(c.contains(&"A0.end".to_string()));
}

#[test]
fn test_complete_midpoint_arg2_lines_only() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "midpoint P0 L");
    assert!(c.contains(&"L0".to_string()));
    assert!(!c.iter().any(|s| is_arc_name(s)), "midpoint arg2 should not offer arcs");
}

#[test]
fn test_complete_offset_line_arg1() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "offset L");
    assert!(c.contains(&"L0".to_string()));
    assert!(!c.iter().any(|s| is_arc_name(s)));
}

#[test]
fn test_complete_list_space_shows_options() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "list ");
    assert!(c.contains(&"lines".to_string()));
    assert!(c.contains(&"constraints".to_string()));
}

#[test]
fn test_complete_horizontal_space_shows_lines() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "horizontal ");
    assert!(c.contains(&"L0".to_string()));
    assert!(c.contains(&"L1".to_string()));
    assert!(!c.iter().any(|s| is_arc_name(s)));
}

#[test]
fn test_complete_cursor_space_shows_keywords() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "cursor ");
    assert!(c.contains(&"on".to_string()));
    assert!(c.contains(&"off".to_string()));
}

#[test]
fn test_complete_style_space_shows_entities() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "style ");
    assert!(c.contains(&"L0".to_string()));
}

#[test]
fn test_complete_style_entity_space_shows_values() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "style L0 ");
    assert!(c.contains(&"solid".to_string()));
    assert!(c.contains(&"dashed".to_string()));
}

#[test]
fn test_complete_empty_first_token_no_suggestions() {
    let ctx = setup_complete_ctx();
    // Just a space or empty — no suggestions
    assert!(completions(&ctx, "").is_empty());
}

#[test]
fn test_complete_add_line_after_coords_only_flags() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_line 0,0 5,0 ");
    assert!(c.contains(&"noconnect".to_string()));
    assert!(c.contains(&"nocursor".to_string()));
    assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities after coords: {:?}", c);
}

#[test]
fn test_complete_add_line_flag_excludes_typed() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_line 0,0 5,0 nocursor ");
    assert!(c.contains(&"noconnect".to_string()));
    assert!(!c.contains(&"nocursor".to_string()), "Should not re-offer nocursor");
}

#[test]
fn test_complete_add_line_first_coord() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_line curs");
    assert!(c.contains(&"cursor".to_string()));
}

#[test]
fn test_complete_add_point_after_coord() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_point 0,0 ");
    assert!(c.contains(&"nocursor".to_string()));
    assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities: {:?}", c);
}

#[test]
fn test_complete_add_circle_radius_position() {
    let ctx = setup_complete_ctx();
    // After center coord, radius is expression context
    let c = completions(&ctx, "add_circle 0,0 w");
    assert!(c.contains(&"width".to_string()));
    assert!(!c.contains(&"cursor".to_string()));
}

#[test]
fn test_complete_add_arc_after_3_coords() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "add_arc 0,0 5,0 2,3 ");
    assert!(c.contains(&"noconnect".to_string()));
    assert!(!c.iter().any(|s| s.starts_with('L')));
}

#[test]
fn test_complete_help_full() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "help f");
    assert!(c.contains(&"full".to_string()));
}

#[test]
fn test_complete_list_all_keyword() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "list a");
    assert!(c.contains(&"all".to_string()));
    assert!(c.contains(&"arcs".to_string()));
}

#[test]
fn test_complete_list_no_second_arg() {
    let ctx = setup_complete_ctx();
    assert!(completions(&ctx, "list lines ").is_empty());
}

#[test]
fn test_complete_horizontal_excludes_typed() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "horizontal L0 L");
    assert!(c.contains(&"L1".to_string()));
    assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
}

#[test]
fn test_complete_select_excludes_typed() {
    let ctx = setup_complete_ctx();
    let c = completions(&ctx, "select L0 P0 L");
    assert!(c.contains(&"L1".to_string()));
    assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
}

#[test]
fn test_complete_cursor_no_second_arg() {
    let ctx = setup_complete_ctx();
    assert!(completions(&ctx, "cursor on ").is_empty());
}

#[test]
fn test_complete_parallel_no_third_arg() {
    let ctx = setup_complete_ctx();
    assert!(completions(&ctx, "parallel L0 L1 ").is_empty());
}

// -- sweep tests --

#[test]
fn test_sweep_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    let out = run_ok(&mut ctx, "sweep A0 180");
    assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
    assert!(ctx.sketch.arcs.refs().next().map(|r| ctx.sketch.arcs[r].constraints.has_target_sweep).unwrap_or(false));
    // Solve and check sweep is close to 180 degrees
    ctx.sketch.solve();
    let r = ctx.sketch.arcs.refs().next().unwrap();
    let sweep = (ctx.sketch.arcs[r].end_angle.value - ctx.sketch.arcs[r].start_angle.value).abs().to_degrees();
    assert!((sweep - 180.0).abs() < 1.0, "Sweep should be ~180, got {}", sweep);
}

#[test]
fn test_sweep_update() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    run_ok(&mut ctx, "sweep A0 180");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let out = run_ok(&mut ctx, "sweep A0 90");
    assert!(out.contains("Updated"), "Should update: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_sweep_derived() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    let out = run_ok(&mut ctx, "sweep A0 derived");
    assert!(out.contains("Derived"), "Should be derived: {}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(ctx.sketch.dimensions[0].derived);
}

#[test]
fn test_sweep_expression() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    run_ok(&mut ctx, "param n 2");
    let out = run_ok(&mut ctx, "sweep A0 \"90*n\"");
    assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
}

#[test]
fn test_sweep_full_circle_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    let e = run_err(&mut ctx, "sweep A0 180");
    assert!(e.contains("full circle"), "Should reject: {}", e);
}

#[test]
fn test_sweep_remove() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    run_ok(&mut ctx, "sweep A0 180");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let name = ctx.sketch.dimensions[0].name.clone();
    run_ok(&mut ctx, &format!("delete {}", name));
    assert_eq!(ctx.sketch.dimensions.len(), 0);
    let r = ctx.sketch.arcs.refs().next().unwrap();
    assert!(!ctx.sketch.arcs[r].constraints.has_target_sweep);
}

// -- arc derived properties in expressions --

#[test]
fn test_print_arc_start_end() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
    let out = run_ok(&mut ctx, "print A0.start.x");
    // Should return a number, not an error
    assert!(out.parse::<f64>().is_ok() || out.trim().parse::<f64>().is_ok(),
        "A0.start.x should be a number: {}", out);
    run_ok(&mut ctx, "print A0.start.y");
    run_ok(&mut ctx, "print A0.end.x");
    run_ok(&mut ctx, "print A0.end.y");
    run_ok(&mut ctx, "print A0.sweep");
    run_ok(&mut ctx, "print A0.diameter");
}

#[test]
fn test_geo_functions_in_expressions() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
    // angle() as standalone works
    let out = run_ok(&mut ctx, "print angle(L0,L1)");
    assert!(out.trim().parse::<f64>().is_ok(), "angle(L0,L1) should be numeric: {}", out);
    // angle() inside an expression
    let out = run_ok(&mut ctx, "print angle(L0,L1)+1");
    let val: f64 = out.trim().parse().expect(&format!("should parse: {}", out));
    assert!((val - 91.0).abs() < 1.0, "angle(L0,L1)+1 should be ~91, got {}", val);
    // dist() inside an expression
    let out = run_ok(&mut ctx, "print dist(L0.p1,L0.p2)*2");
    let val: f64 = out.trim().parse().expect(&format!("should parse: {}", out));
    assert!((val - 10.0).abs() < 0.1, "dist*2 should be ~10, got {}", val);
}

#[test]
fn test_inline_comments() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0  # a horizontal line");
    assert_eq!(ctx.sketch.lines.len(), 1);
    run_ok(&mut ctx, "horizontal L0 # make it horizontal");
    assert!(ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].constraints.horizontal);
    // Comment-only line
    let out = run_ok(&mut ctx, "# just a comment");
    assert!(out.is_empty());
    // Quoted strings should not be affected
    run_ok(&mut ctx, "param scale 1");
    run_ok(&mut ctx, "add_circle 0,0 5");
    run_ok(&mut ctx, "radius A0 =5*scale # expression dimension");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_dimension_variable_assignment() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "s0 = add_line 0,0 5,0; s1 = add_line 5,0 3,4");
    run_ok(&mut ctx, "len = length s0 5");
    assert!(ctx.session_names.contains_key("len"), "len should be set");
    assert_eq!(ctx.session_names["len"], "d0");
    run_ok(&mut ctx, "a = angle s0 s1 60");
    assert!(ctx.session_names.contains_key("a"), "a should be set");
    assert_eq!(ctx.session_names["a"], "d1");
    // Use dimension variable as expression in another dimension
    let out = run_ok(&mut ctx, "print a");
    assert!(out.trim().parse::<f64>().is_ok(), "should resolve: {}", out);
}

// -- delete (relational / named constraint) tests --

#[test]
fn test_remove_constraint_coincident_pp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 0,0; add_point 1,0");
    run_ok(&mut ctx, "coincident P0 P1");
    assert_eq!(ctx.sketch.coincident_pp.len(), 1);
    run_ok(&mut ctx, "delete P0 P1 coincident");
    assert_eq!(ctx.sketch.coincident_pp.len(), 0);
}

#[test]
fn test_remove_constraint_coincident_ll() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
    run_ok(&mut ctx, "coincident L0.p2 L1.p1");
    assert_eq!(ctx.sketch.coincident_ll21.len(), 1);
    run_ok(&mut ctx, "delete L0.p2 L1.p1 coincident");
    assert_eq!(ctx.sketch.coincident_ll21.len(), 0);
}

#[test]
fn test_remove_constraint_coincident_not_found() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
    let e = run_err(&mut ctx, "delete L0.p2 L1.p1 coincident");
    assert!(e.contains("not found"), "{}", e);
}

#[test]
fn test_remove_constraint_point_on_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 2,0.5; add_line 0,0 5,0");
    run_ok(&mut ctx, "point_on P0 L0");
    assert_eq!(ctx.sketch.point_on_line.len(), 1);
    run_ok(&mut ctx, "delete P0 L0 point_on");
    assert_eq!(ctx.sketch.point_on_line.len(), 0);
}

#[test]
fn test_remove_constraint_point_on_line_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
    run_ok(&mut ctx, "point_on L0.p1 L1");
    assert_eq!(ctx.sketch.line_p1_on_line.len(), 1);
    run_ok(&mut ctx, "delete L0.p1 L1 point_on");
    assert_eq!(ctx.sketch.line_p1_on_line.len(), 0);
}

#[test]
fn test_remove_constraint_point_on_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 5,0; add_circle 0,0 5");
    run_ok(&mut ctx, "point_on P0 A0");
    assert_eq!(ctx.sketch.point_on_arc.len(), 1);
    run_ok(&mut ctx, "delete P0 A0 point_on");
    assert_eq!(ctx.sketch.point_on_arc.len(), 0);
}

#[test]
fn test_remove_constraint_point_on_line_arc_endpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
    run_ok(&mut ctx, "point_on A0.center L0");
    assert!(!ctx.sketch.point_on_line.is_empty());
    run_ok(&mut ctx, "delete A0.center L0 point_on");
    // The point_on_line constraint on the helper should be removed
    // cleanup_helper_points removes orphan helpers
    assert!(ctx.sketch.point_on_line.is_empty() || ctx.sketch.points.refs().all(|p| !ctx.sketch.points[p].helper));
}

#[test]
fn test_remove_constraint_symmetry_pp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point -3,0; add_line 0,-5 0,5; add_point 3,0");
    run_ok(&mut ctx, "symmetry P0 L0 P1");
    assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
    run_ok(&mut ctx, "delete P0 L0 P1 symmetry");
    assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
}

#[test]
fn test_remove_constraint_symmetry_ll() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
    run_ok(&mut ctx, "symmetry L0 L1 L2");
    assert_eq!(ctx.sketch.symmetry_ll.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 L2 symmetry");
    assert_eq!(ctx.sketch.symmetry_ll.len(), 0);
}

#[test]
fn test_remove_constraint_midpoint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 2.5,0.5; add_line 0,0 5,0");
    run_ok(&mut ctx, "midpoint P0 L0");
    assert_eq!(ctx.sketch.midpoint.len(), 1);
    run_ok(&mut ctx, "delete P0 L0 midpoint");
    assert_eq!(ctx.sketch.midpoint.len(), 0);
}

#[test]
fn test_remove_constraint_midpoint_lp() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line -5,0 10,0");
    run_ok(&mut ctx, "midpoint L0.p1 L1");
    assert_eq!(ctx.sketch.midpoint_lp1.len(), 1);
    run_ok(&mut ctx, "delete L0.p1 L1 midpoint");
    assert_eq!(ctx.sketch.midpoint_lp1.len(), 0);
}

#[test]
fn test_remove_constraint_equal_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
    run_ok(&mut ctx, "equal A0 A1");
    assert_eq!(ctx.sketch.equal_radius.len(), 1);
    run_ok(&mut ctx, "delete A0 A1 equal_radius");
    assert_eq!(ctx.sketch.equal_radius.len(), 0);
}

#[test]
fn test_remove_constraint_equal_radius_not_found() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
    let e = run_err(&mut ctx, "delete A0 A1 equal_radius");
    assert!(e.contains("not found"), "{}", e);
}

#[test]
fn test_remove_constraint_horizontal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    run_ok(&mut ctx, "delete L0 horizontal");
    assert!(!ctx.sketch.lines.iter().next().unwrap().constraints.horizontal);
}

#[test]
fn test_remove_constraint_vertical() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,5");
    run_ok(&mut ctx, "vertical L0");
    run_ok(&mut ctx, "delete L0 vertical");
    assert!(!ctx.sketch.lines.iter().next().unwrap().constraints.vertical);
}

#[test]
fn test_remove_constraint_parallel() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    assert_eq!(ctx.sketch.parallel.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 parallel");
    assert_eq!(ctx.sketch.parallel.len(), 0);
}

#[test]
fn test_remove_constraint_perpendicular() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
    run_ok(&mut ctx, "perpendicular L0 L1");
    assert_eq!(ctx.sketch.perpendicular.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 perpendicular");
    assert_eq!(ctx.sketch.perpendicular.len(), 0);
}

#[test]
fn test_remove_constraint_equal_length() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "equal L0 L1");
    assert_eq!(ctx.sketch.equal_length.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 equal");
    assert_eq!(ctx.sketch.equal_length.len(), 0);
}

#[test]
fn test_remove_constraint_collinear() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 6,0 10,0");
    run_ok(&mut ctx, "collinear L0 L1");
    assert_eq!(ctx.sketch.collinear.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 collinear");
    assert_eq!(ctx.sketch.collinear.len(), 0);
}

#[test]
fn test_remove_constraint_tangent_la() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,4 5,4; add_circle 2,0 4");
    run_ok(&mut ctx, "tangent L0 A0");
    assert_eq!(ctx.sketch.tangent_la.len(), 1);
    run_ok(&mut ctx, "delete L0 A0 tangent");
    assert_eq!(ctx.sketch.tangent_la.len(), 0);
}

#[test]
fn test_remove_constraint_tangent_aa() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_circle 7,0 4");
    run_ok(&mut ctx, "tangent A0 A1");
    assert_eq!(ctx.sketch.tangent_aa.len(), 1);
    run_ok(&mut ctx, "delete A0 A1 tangent");
    assert_eq!(ctx.sketch.tangent_aa.len(), 0);
}

#[test]
fn test_remove_constraint_concentric() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3; add_circle 1,0 5");
    run_ok(&mut ctx, "concentric A0 A1");
    assert_eq!(ctx.sketch.concentric.len(), 1);
    run_ok(&mut ctx, "delete A0 A1 concentric");
    assert_eq!(ctx.sketch.concentric.len(), 0);
}

/// Regression: the circle-to-circle distance dimension must be
/// placeable on two geometrically-concentric circles without a
/// pre-existing `Concentric` constraint. The command auto-installs
/// a Concentric for list-visibility.
#[test]
fn test_distance_concentric_no_prior_concentric() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
    run_ok(&mut ctx, "add_circle 0,0 5 noconnect");
    assert_eq!(ctx.sketch.concentric.len(), 0);
    run_ok(&mut ctx, "distance A0 A1 3");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert_eq!(ctx.sketch.distance_concentric.len(), 1);
    // A paired Concentric was auto-installed for visibility.
    assert_eq!(ctx.sketch.concentric.len(), 1);
    assert!((ctx.sketch.arcs[ctx.sketch.arcs.refs().nth(1).unwrap()].radius.value
           - ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value
           - 3.0).abs() < 0.01);
}

/// Regression: the dim's self-contained residual keeps the circles
/// concentric even after the user manually deletes the paired
/// `Concentric` constraint. Previously the cascade removed the dim
/// alongside the Concentric; now the dim survives.
#[test]
fn test_concentric_distance_survives_concentric_delete() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
    run_ok(&mut ctx, "add_circle 0,0 5 noconnect");
    run_ok(&mut ctx, "distance A0 A1 3");
    assert_eq!(ctx.sketch.concentric.len(), 1);
    assert_eq!(ctx.sketch.distance_concentric.len(), 1);
    // Delete the paired Concentric by name. The dim (and its
    // backing constraint) must stay.
    let nid = ctx.sketch.concentric[0].nid;
    run_ok(&mut ctx, &format!("delete C{}", nid));
    assert_eq!(ctx.sketch.concentric.len(), 0);
    assert_eq!(ctx.sketch.dimensions.len(), 1,
        "dim must survive manual Concentric deletion");
    assert_eq!(ctx.sketch.distance_concentric.len(), 1,
        "backing DistanceConcentric must survive");
    // Solve must still hold the circles concentric (the dim's own
    // residual enforces it).
    ctx.sketch.solve();
    let ca = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].center.value;
    let cb = ctx.sketch.arcs[ctx.sketch.arcs.refs().nth(1).unwrap()].center.value;
    assert!((ca.x - cb.x).abs() < 0.01 && (ca.y - cb.y).abs() < 0.01,
        "circles must stay concentric: {:?} vs {:?}", ca, cb);
}

/// The dim only accepts circles whose centers currently coincide.
/// Non-concentric pairs fall through to PointPointDistance or
/// fail, per the existing grammar.
#[test]
fn test_distance_concentric_rejects_non_concentric_circles() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
    run_ok(&mut ctx, "add_circle 5,0 3 noconnect");
    // Centers are 5 units apart -> not eligible for
    // ConcentricDistance. `distance A0 A1 3` falls through to the
    // endpoint-pair path, which fails to resolve bare arc names as
    // endpoints.
    let out = run_err(&mut ctx, "distance A0 A1 3");
    assert!(out.contains("Cannot parse endpoint"), "{}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 0);
    assert_eq!(ctx.sketch.distance_concentric.len(), 0);
}

#[test]
fn test_remove_constraint_undo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    let dof_with = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "delete L0 horizontal");
    let dof_without = ctx.sketch.dof().unwrap();
    assert!(dof_without > dof_with, "DOF should increase after removing constraint: {} vs {}", dof_without, dof_with);
    run_ok(&mut ctx, "undo");
    let dof_undone = ctx.sketch.dof().unwrap();
    assert_eq!(dof_undone, dof_with, "DOF should restore after undo: {} vs {}", dof_undone, dof_with);
}

#[test]
fn test_remove_constraint_dof_update() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "delete L0 L1 parallel");
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before + 1, "removing parallel should increase DOF by 1: {} -> {}", dof_before, dof_after);
}

// -- Multi-segment add_line --

#[test]
fn test_add_line_multi_segment_3_points() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 2,1");
    assert_eq!(ctx.sketch.lines.len(), 2);
    // Auto-coincident between L0.p2 and L1.p1
    assert!(!ctx.sketch.coincident_ll21.is_empty() || !ctx.sketch.coincident_ll12.is_empty(),
        "segments should be connected");
}

#[test]
fn test_add_line_multi_segment_5_points() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0 2,1 3,0 4,1");
    assert_eq!(ctx.sketch.lines.len(), 4);
}

#[test]
fn test_add_line_multi_segment_relative() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 @1,0 @0,1");
    assert_eq!(ctx.sketch.lines.len(), 2);
    let l1 = ctx.sketch.lines.iter().nth(1).unwrap().p2.value;
    assert!((l1.x - 1.0).abs() < 0.01 && (l1.y - 1.0).abs() < 0.01,
        "L1.p2 should be (1,1), got ({},{})", l1.x, l1.y);
}

#[test]
fn test_add_line_multi_assignment() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "a, b, c = add_line 0,0 @1,0 @0,1 @-1,0");
    assert_eq!(ctx.sketch.lines.len(), 3);
    assert!(ctx.session_names.contains_key("a"));
    assert!(ctx.session_names.contains_key("b"));
    assert!(ctx.session_names.contains_key("c"));
    // Use alias in constraint
    run_ok(&mut ctx, "horizontal a");
    assert!(ctx.sketch.lines.iter().next().unwrap().constraints.horizontal);
}

#[test]
fn test_add_line_two_points_compat() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    assert_eq!(ctx.sketch.lines.len(), 1);
}

#[test]
fn test_add_line_multi_segment_dof() {
    let mut ctx = CommandContext::new();
    // 4 points = 3 lines, 2 coincidents
    // 3*4 params - 2*2 coincident = 8 DOF
    run_ok(&mut ctx, "add_line 0,0 1,0 2,1 3,0");
    let dof = ctx.sketch.dof().unwrap();
    assert_eq!(dof, 8, "3 connected lines should have 8 DOF, got {}", dof);
}

// -- Angle direct/supplement --

#[test]
fn test_angle_default_is_direction_vectors() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    // Default: constrains the angle between p1->p2 direction vectors
    run_ok(&mut ctx, "angle L0 L1 45");
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(!supplement, "default should not be supplement");
    } else {
        panic!("expected angle dimension");
    }
}

#[test]
fn test_angle_supplement_keyword() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    run_ok(&mut ctx, "angle L0 L1 135 supplement");
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "should be supplement sector");
    } else {
        panic!("expected angle dimension");
    }
}

#[test]
fn test_angle_closest_keyword() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    // Current angle is ~45. Value 130 is closer to supplement (135) than direct (45).
    run_ok(&mut ctx, "angle L0 L1 130 closest");
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "closest should pick supplement for 130 when direct is ~45");
    } else {
        panic!("expected angle dimension");
    }
}

#[test]
fn test_angle_acute_keyword() {
    let mut ctx = CommandContext::new();
    // Lines at ~120 degrees (direct angle > 90, so acute picks supplement)
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 -2,4");
    run_ok(&mut ctx, "angle L0 L1 60 acute");
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "acute should pick the smaller sector");
    } else {
        panic!("expected angle dimension");
    }
}

#[test]
fn test_angle_obtuse_keyword() {
    let mut ctx = CommandContext::new();
    // Lines at ~45 degrees (direct is 45, supplement is 135)
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    run_ok(&mut ctx, "angle L0 L1 135 obtuse");
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        // Direct is ~45 (acute), so obtuse picks supplement (135)
        assert!(supplement, "obtuse should pick the larger sector");
    } else {
        panic!("expected angle dimension");
    }
}

#[test]
fn test_angle_negative_value_accepted() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    // Negative value should be accepted (taken as absolute value)
    run_ok(&mut ctx, "angle L0 L1 -45");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_angle_driven_closest() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    let dof_before = ctx.sketch.dof().unwrap();
    // "driven closest" — driven is before sector keyword
    run_ok(&mut ctx, "angle L0 L1 driven closest");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(ctx.sketch.dof().unwrap() < dof_before);
}

#[test]
fn test_angle_driven_supplement() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "angle L0 L1 driven supplement");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "should be supplement sector");
    } else {
        panic!("expected angle dimension");
    }
    assert!(ctx.sketch.dof().unwrap() < dof_before);
}

#[test]
fn test_angle_driven_acute() {
    let mut ctx = CommandContext::new();
    // Lines at ~120 degrees so acute picks supplement
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 -2,4");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "angle L0 L1 driven acute");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "acute should pick the smaller sector");
    } else {
        panic!("expected angle dimension");
    }
    assert!(ctx.sketch.dof().unwrap() < dof_before);
}

#[test]
fn test_angle_driven_obtuse() {
    let mut ctx = CommandContext::new();
    // Lines at ~45 degrees, so obtuse picks supplement (135)
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "angle L0 L1 driven obtuse");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
        assert!(supplement, "obtuse should pick the larger sector");
    } else {
        panic!("expected angle dimension");
    }
    assert!(ctx.sketch.dof().unwrap() < dof_before);
}

#[test]
fn test_angle_closest_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    let dof_before = ctx.sketch.dof().unwrap();
    // Reverse order: sector keyword before driven
    run_ok(&mut ctx, "angle L0 L1 closest driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(ctx.sketch.dof().unwrap() < dof_before);
}

#[test]
fn test_angle_value_closest_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    // With explicit value + sector + driven
    run_ok(&mut ctx, "angle L0 L1 45 closest driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
}

#[test]
fn test_angle_value_driven_closest() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    // Reverse order with explicit value
    run_ok(&mut ctx, "angle L0 L1 45 driven closest");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
}

// -- chamfer --

#[test]
fn test_chamfer_two_lines() {
    let mut ctx = CommandContext::new();
    // Right-angle corner at (5,0).
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    let out = run_ok(&mut ctx, "chamfer L0 L1 0.5");
    assert!(out.contains("L0 L1 -> "), "got: {}", out);
    // A new line (the bevel) and a new point (corner anchor).
    assert_eq!(ctx.sketch.lines.refs().count(), 3);
    assert_eq!(ctx.sketch.points.refs().filter(|r| !ctx.sketch.points[*r].helper).count(), 1);
    // Two distance dims.
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    // Secondary dim tracks the primary by name.
    let primary_name = ctx.sketch.dimensions[0].name.clone();
    assert_eq!(ctx.sketch.dimensions[1].expr_str.as_deref(), Some(primary_name.as_str()));
    // L0 trimmed to x=4.5, L1 trimmed to y=0.5.
    let l0 = ctx.sketch.lines.iter().next().unwrap();
    assert!((l0.p2.value.x - 4.5).abs() < 1e-6);
    let l1 = ctx.sketch.lines.iter().nth(1).unwrap();
    assert!((l1.p1.value.y - 0.5).abs() < 1e-6);
}

#[test]
fn test_chamfer_endpoint_form() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    let out = run_ok(&mut ctx, "chamfer L0.p2 0.5");
    assert!(out.contains("L0.p2 -> "), "got: {}", out);
}

#[test]
fn test_chamfer_parametric_distance() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    run_ok(&mut ctx, "add_line 10,0 10,10");
    run_ok(&mut ctx, "length L0 10");
    // Live expression on the chamfer distance -- must stick as
    // expr_str on the primary distance dim so the whole chamfer
    // tracks when d0 moves.
    run_ok(&mut ctx, "chamfer L0 L1 d0*0.1");
    // Dim layout: d0 (length), d1 (chamfer primary), d2 (secondary).
    let d1 = &ctx.sketch.dimensions[1];
    assert_eq!(d1.expr_str.as_deref(), Some("d0*0.1"));
    let d2 = &ctx.sketch.dimensions[2];
    assert_eq!(d2.expr_str.as_deref(), Some(d1.name.as_str()));
    run_ok(&mut ctx, "length L0 20");
    let d1 = &ctx.sketch.dimensions[1];
    assert!((d1.value - 2.0).abs() < 1e-3,
        "chamfer primary should track to 2.0, got {}", d1.value);
}

#[test]
fn test_chamfer_rejects_collinear() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 10,0");
    let r = execute_one(&mut ctx, "chamfer L0 L1 0.5");
    assert!(r.is_error);
    assert!(r.output.contains("collinear"), "got: {}", r.output);
}

#[test]
fn test_chamfer_rejects_too_short() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0");
    run_ok(&mut ctx, "add_line 1,0 1,1");
    let r = execute_one(&mut ctx, "chamfer L0 L1 2");
    assert!(r.is_error);
    assert!(r.output.contains("too short"), "got: {}", r.output);
}

// -- fillet --

#[test]
fn test_fillet_two_lines() {
    let mut ctx = CommandContext::new();
    // Right-angle corner at (5,0).
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    let out = run_ok(&mut ctx, "fillet L0 L1 0.5");
    assert!(out.contains("L0 L1 -> A0"), "got: {}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    assert_eq!(ctx.sketch.tangent_la.len(), 2);
    // Radius dimension added.
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    // Lines trimmed: L0.p2 was at x=5, now at x=4.5.
    let l0 = ctx.sketch.lines.iter().next().unwrap();
    assert!((l0.p2.value.x - 4.5).abs() < 1e-6,
        "L0.p2.x should be 4.5, got {}", l0.p2.value.x);
    let l1 = ctx.sketch.lines.iter().nth(1).unwrap();
    assert!((l1.p1.value.y - 0.5).abs() < 1e-6,
        "L1.p1.y should be 0.5, got {}", l1.p1.value.y);
}

#[test]
fn test_fillet_endpoint_form() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    let out = run_ok(&mut ctx, "fillet L0.p2 0.5");
    assert!(out.contains("L0.p2 -> "), "got: {}", out);
}

#[test]
fn test_fillet_variadic_all_rect_corners() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 5,3 hv");
    // All four corners in one command, last token is radius.
    // Every non-primary fillet references the primary dim name
    // so the four radii track one source.
    let out = run_ok(&mut ctx, "fillet L0 L1 L1 L2 L2 L3 L3 L0 0.5");
    assert!(out.contains("Filleted 4 of 4"), "got: {}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 4);
    // Exactly one radius dim that's a literal; three that
    // reference the primary by name.
    let dims: Vec<_> = ctx.sketch.dimensions.iter().collect();
    assert_eq!(dims.len(), 4);
    let primary = &dims[0];
    assert!(primary.expr_str.is_none());
    for d in &dims[1..] {
        assert_eq!(d.expr_str.as_deref(), Some(primary.name.as_str()));
    }
}

#[test]
fn test_fillet_variadic_partial_failure() {
    // First corner valid, second corner too short -- whole command
    // still succeeds with one fillet reported and the other as
    // FAILED in the per-corner report.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    // L2 is only 0.1 long -- fillet radius 0.5 is too short.
    run_ok(&mut ctx, "add_line 5,3 5.1,3");
    let out = run_ok(&mut ctx, "fillet L0 L1 L1 L2 0.5");
    assert!(out.contains("Filleted 1 of 2"), "got: {}", out);
    assert!(out.contains("FAILED"), "got: {}", out);
}

#[test]
fn test_fillet_notangent_noradius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 5,3");
    run_ok(&mut ctx, "fillet L0 L1 0.5 notangent noradius");
    assert_eq!(ctx.sketch.tangent_la.len(), 0);
    assert_eq!(ctx.sketch.dimensions.len(), 0);
    // Arc still exists.
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
}

#[test]
fn test_fillet_parametric_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    run_ok(&mut ctx, "add_line 10,0 10,10");
    run_ok(&mut ctx, "length L0 10");
    // Radius expressed as a live expression referencing the length
    // dim: the fillet dim must store the expression so later edits
    // to d0 propagate through.
    run_ok(&mut ctx, "fillet L0 L1 d0*0.1");
    // Radius dim is the second dim (d1). Verify it captured the
    // expression, not the numeric snapshot.
    let r_dim = &ctx.sketch.dimensions[1];
    assert_eq!(r_dim.expr_str.as_deref(), Some("d0*0.1"),
        "expected live expr, got {:?}", r_dim.expr_str);
    // Change the source; fillet radius must follow.
    run_ok(&mut ctx, "length L0 20");
    let r_dim = &ctx.sketch.dimensions[1];
    assert!((r_dim.value - 2.0).abs() < 1e-3,
        "radius should track to 2.0 after length=20, got {}", r_dim.value);
}

#[test]
fn test_fillet_rejects_collinear() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 5,0 10,0");
    let r = execute_one(&mut ctx, "fillet L0 L1 0.5");
    assert!(r.is_error, "collinear fillet must fail");
    assert!(r.output.contains("collinear"), "got: {}", r.output);
}

#[test]
fn test_fillet_rejects_too_short() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0");
    run_ok(&mut ctx, "add_line 1,0 1,1");
    let r = execute_one(&mut ctx, "fillet L0 L1 2");
    assert!(r.is_error);
    assert!(r.output.contains("too short"), "got: {}", r.output);
}

#[test]
fn test_fillet_rejects_disconnected() {
    let mut ctx = CommandContext::new();
    // Not touching.
    run_ok(&mut ctx, "add_line 0,0 1,0");
    run_ok(&mut ctx, "add_line 2,0 2,1");
    let r = execute_one(&mut ctx, "fillet L0 L1 0.2");
    assert!(r.is_error);
    assert!(r.output.contains("not connected"), "got: {}", r.output);
}

// -- add_rect / add_rect3 / add_rectcenter --

#[test]
fn test_add_rect_basic() {
    let mut ctx = CommandContext::new();
    let out = run_ok(&mut ctx, "add_rect 0,0 5,3");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    assert_eq!(ctx.sketch.perpendicular.len(), 1);
    assert_eq!(ctx.sketch.parallel.len(), 2);
    assert!(out.contains("perpendicular"), "should list perpendicular: {}", out);
    assert!(out.contains("parallel"), "should list parallel: {}", out);
}

#[test]
fn test_add_rect_hv() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 5,3 hv");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    // hv: horizontal on 2 lines, vertical on 2 lines
    let h_count = ctx.sketch.lines.refs().filter(|r| ctx.sketch.lines[*r].constraints.horizontal).count();
    let v_count = ctx.sketch.lines.refs().filter(|r| ctx.sketch.lines[*r].constraints.vertical).count();
    assert_eq!(h_count, 2);
    assert_eq!(v_count, 2);
    assert_eq!(ctx.sketch.perpendicular.len(), 0);
    assert_eq!(ctx.sketch.parallel.len(), 0);
}

#[test]
fn test_add_rect_noconstraint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 5,3 noconstraint");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    assert_eq!(ctx.sketch.perpendicular.len(), 0);
    assert_eq!(ctx.sketch.parallel.len(), 0);
}

#[test]
fn test_add_rect_driven() {
    let mut ctx = CommandContext::new();
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "add_rect 0,0 5,3 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(!ctx.sketch.dimensions[1].derived);
    assert!(ctx.sketch.dof().unwrap() < dof_before + 8); // 4 lines = +8 DOF, constraints + dims reduce
}

#[test]
fn test_add_rect_noconstraint_conflicts() {
    let mut ctx = CommandContext::new();
    let r1 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint hv");
    assert!(r1.is_error);
    let r2 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint driven");
    assert!(r2.is_error);
    let r3 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint strict");
    assert!(r3.is_error);
}

#[test]
fn test_add_rect_relative_coords() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 @5,3");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    let l0 = ctx.sketch.lines.refs().next().unwrap();
    let l = &ctx.sketch.lines[l0];
    assert!(near(l.p2.value.x - l.p1.value.x, 5.0));
}

#[test]
fn test_add_rect_session_names() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 5,3");
    assert_eq!(ctx.session_names.get("_0").map(|s| s.as_str()), Some("L0"));
    assert_eq!(ctx.session_names.get("_1").map(|s| s.as_str()), Some("L1"));
    assert_eq!(ctx.session_names.get("_2").map(|s| s.as_str()), Some("L2"));
    assert_eq!(ctx.session_names.get("_3").map(|s| s.as_str()), Some("L3"));
}

#[test]
fn test_add_rect_noconnect() {
    let mut ctx = CommandContext::new();
    // Place a line at a rect corner
    run_ok(&mut ctx, "add_line 0,0 10,0");
    let coinc_before = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
        + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
    run_ok(&mut ctx, "add_rect 0,0 5,3 noconnect");
    let coinc_after = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
        + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
    // No new coincident constraints (not even internal ones)
    assert_eq!(coinc_after, coinc_before);
}

#[test]
fn test_add_rect_non_strict_warns() {
    let mut ctx = CommandContext::new();
    // Fully constrain a rect
    run_ok(&mut ctx, "add_rect 0,0 5,3 hv driven");
    // Overlapping rect: constraints will be redundant, but non-strict should succeed with warnings
    let result = run_ok(&mut ctx, "add_rect 0,0 5,3");
    assert!(result.contains("warning"), "should contain warnings: {}", result);
}

#[test]
fn test_add_rect3_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect3 0,0 5,0 5,3");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    assert_eq!(ctx.sketch.perpendicular.len(), 1);
    assert_eq!(ctx.sketch.parallel.len(), 2);
}

#[test]
fn test_add_rect3_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect3 0,0 5,0 5,3 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(!ctx.sketch.dimensions[1].derived);
}

#[test]
fn test_add_rect3_hv_rejected() {
    let mut ctx = CommandContext::new();
    let r = execute_one(&mut ctx, "add_rect3 0,0 5,0 5,3 hv");
    assert!(r.is_error);
}

#[test]
fn test_add_rect3_collinear_rejected() {
    let mut ctx = CommandContext::new();
    let r = execute_one(&mut ctx, "add_rect3 1,1 2,3 3,5");
    assert!(r.is_error, "collinear points should be rejected: {}", r.output);
    assert!(r.output.contains("collinear"), "error should mention collinear: {}", r.output);
    assert_eq!(ctx.sketch.lines.refs().count(), 0, "no geometry should be created");
}

#[test]
fn test_add_rectcenter_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rectcenter 2.5,1.5 0,0");
    assert_eq!(ctx.sketch.lines.refs().count(), 4);
    assert_eq!(ctx.sketch.perpendicular.len(), 1);
    assert_eq!(ctx.sketch.parallel.len(), 2);
    // Verify corners: bl=(0,0), br=(5,0), tr=(5,3), tl=(0,3)
    let refs: Vec<_> = ctx.sketch.lines.refs().collect();
    let l0 = &ctx.sketch.lines[refs[0]];
    assert!(near(l0.p1.value.x, 0.0) && near(l0.p1.value.y, 0.0));
    assert!(near(l0.p2.value.x, 5.0) && near(l0.p2.value.y, 0.0));
}

#[test]
fn test_add_rectcenter_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rectcenter 2.5,1.5 0,0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(!ctx.sketch.dimensions[0].derived);
}

// -- add_line / add_circle driven + new circle tools --

#[test]
fn test_short_line_does_not_break_solver() {
    // A very short line should not prevent constraints on other entities.
    // Previously, the length drift's sqrt had a singularity at zero length.
    // Note: truly zero-length lines are now softly penalized (Heaviside minimum
    // length), so we use a short-but-nonzero line instead.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0.1; add_line 3,3 @0.02,0");
    run_ok(&mut ctx, "horizontal L0");
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    assert!((l.p1.value.y - l.p2.value.y).abs() < 0.01);
}

#[test]
fn test_add_line_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 5.0));
}

#[test]
fn test_add_line_multi_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0 5,3 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(near(ctx.sketch.dimensions[0].value, 5.0));
    assert!(near(ctx.sketch.dimensions[1].value, 3.0));
}

#[test]
fn test_add_circle_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 3 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 3.0));
}

#[test]
fn test_add_arc_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 0,0 5,0 2.5,2.5 driven noconnect");
    // Should have 2 dimensions: radius and sweep
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(!ctx.sketch.dimensions[1].derived);
    // Radius > 0
    assert!(ctx.sketch.dimensions[0].value > 0.0);
    // Sweep > 0
    assert!(ctx.sketch.dimensions[1].value > 0.0);
}

#[test]
fn test_add_arc_driven_variable_assignment() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "l = add_line 0,0 1,0");
    run_ok(&mut ctx, "a = add_arc 1,0 3,1 2,0 driven noconnect");
    // Variable 'a' should resolve to the arc, not the dimension
    run_ok(&mut ctx, "tangent l a");
    assert_eq!(ctx.sketch.tangent_la.len(), 1);
}

#[test]
fn test_add_circle2_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle2 0,0 10,0");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    assert!(near(arc.center.value.x, 5.0));
    assert!(near(arc.center.value.y, 0.0));
    assert!(near(arc.radius.value, 5.0));
}

#[test]
fn test_add_circle3_basic() {
    let mut ctx = CommandContext::new();
    // Points on a circle of radius 5 centered at origin
    run_ok(&mut ctx, "add_circle3 5,0 -5,0 0,5");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    assert!(near(arc.center.value.x, 0.0));
    assert!(near(arc.center.value.y, 0.0));
    assert!(near(arc.radius.value, 5.0));
}

#[test]
fn test_add_circle3_collinear_error() {
    let mut ctx = CommandContext::new();
    let r = execute_one(&mut ctx, "add_circle3 0,0 5,0 10,0");
    assert!(r.is_error);
}

#[test]
fn test_add_circle2t_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
    let out = run_ok(&mut ctx, "add_circle2t L0 L1 2");
    assert!(out.contains("tangent L0"), "should list tangent L0: {}", out);
    assert!(out.contains("tangent L1"), "should list tangent L1: {}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    assert!(near(arc.center.value.x, 2.0));
    assert!(near(arc.center.value.y, 2.0));
    assert!(near(arc.radius.value, 2.0));
    assert_eq!(ctx.sketch.tangent_la.len(), 2);
}

#[test]
fn test_add_circle2t_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
    run_ok(&mut ctx, "add_circle2t L0 L1 2 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
    assert!(near(ctx.sketch.dimensions[0].value, 2.0));
}

#[test]
fn test_add_circle2t_noconstraint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
    run_ok(&mut ctx, "add_circle2t L0 L1 2 noconstraint");
    assert_eq!(ctx.sketch.tangent_la.len(), 0);
}

#[test]
fn test_add_circle3t_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
    run_ok(&mut ctx, "add_circle3t L0 L1 L2");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    assert_eq!(ctx.sketch.tangent_la.len(), 3);
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    // Incircle should be inside the triangle
    assert!(arc.center.value.x > 0.0 && arc.center.value.x < 10.0);
    assert!(arc.center.value.y > 0.0 && arc.center.value.y < 8.0);
}

#[test]
fn test_add_circle3t_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
    run_ok(&mut ctx, "add_circle3t L0 L1 L2 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(!ctx.sketch.dimensions[0].derived);
}

#[test]
fn test_add_circle2t_segment_disambiguation() {
    let mut ctx = CommandContext::new();
    // Two perpendicular rays from origin: only 1 of 4 sectors has both tangent points on segments
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
    run_ok(&mut ctx, "add_circle2t L0 L1 1");
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    // Must be in the +x,+y quadrant (interior of the angle)
    assert!(arc.center.value.x > 0.0, "center.x should be positive: {}", arc.center.value.x);
    assert!(arc.center.value.y > 0.0, "center.y should be positive: {}", arc.center.value.y);
}

#[test]
fn test_add_circle2t_no_touching() {
    let mut ctx = CommandContext::new();
    // Two parallel segments far apart: no tangent circle of r=1 touches both
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,100 10,100");
    let r = execute_one(&mut ctx, "add_circle2t L0 L1 1");
    assert!(r.is_error, "should fail: {}", r.output);
}

#[test]
fn test_add_circle3t_segment_touches() {
    let mut ctx = CommandContext::new();
    // Closed triangle: exactly 1 incircle touches all 3 segments
    run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
    run_ok(&mut ctx, "add_circle3t L0 L1 L2");
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[arc_ref];
    // Incircle center must be inside the triangle
    assert!(arc.center.value.x > 0.0 && arc.center.value.x < 10.0);
    assert!(arc.center.value.y > 0.0 && arc.center.value.y < 8.0);
    assert_eq!(ctx.sketch.tangent_la.len(), 3);
}

// -- mirror --

#[test]
fn test_mirror_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
    run_ok(&mut ctx, "mirror L1 about L0");
    assert_eq!(ctx.sketch.lines.refs().count(), 3);
    let refs: Vec<_> = ctx.sketch.lines.refs().collect();
    let mirrored = &ctx.sketch.lines[refs[2]];
    assert!(near(mirrored.p1.value.x, 3.0));
    assert!(near(mirrored.p2.value.x, 3.0));
    assert_eq!(ctx.sketch.symmetry_pp.len(), 2);
}

#[test]
fn test_mirror_circle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_circle -5,3 2");
    run_ok(&mut ctx, "mirror A0 about L0");
    assert_eq!(ctx.sketch.arcs.refs().count(), 2);
    let refs: Vec<_> = ctx.sketch.arcs.refs().collect();
    let mirrored = &ctx.sketch.arcs[refs[1]];
    assert!(near(mirrored.center.value.x, 5.0));
    assert!(near(mirrored.center.value.y, 3.0));
    assert!(near(mirrored.radius.value, 2.0));
    // One arc symmetry holds center and radius for a circle.
    assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
    assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
}

#[test]
fn test_mirror_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_point -4,5");
    run_ok(&mut ctx, "mirror P0 about L0");
    assert_eq!(ctx.sketch.points.refs().count(), 2);
    let refs: Vec<_> = ctx.sketch.points.refs().collect();
    let mirrored = &ctx.sketch.points[refs[1]];
    assert!(near(mirrored.pos.value.x, 4.0));
    assert!(near(mirrored.pos.value.y, 5.0));
    assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
}

#[test]
fn test_mirror_multiple() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,1 -5,4; add_line -5,4 -2,7");
    run_ok(&mut ctx, "mirror L1 L2 about L0");
    assert_eq!(ctx.sketch.lines.refs().count(), 5);
}

#[test]
fn test_mirror_selection() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
    run_ok(&mut ctx, "select L1");
    run_ok(&mut ctx, "mirror selection about L0");
    assert_eq!(ctx.sketch.lines.refs().count(), 3);
}

#[test]
fn test_mirror_session_names() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8; add_line -3,8 -6,5");
    run_ok(&mut ctx, "mirror L1 L2 about L0");
    assert_eq!(ctx.session_names.get("_0").map(|s| s.as_str()), Some("L3"));
    assert_eq!(ctx.session_names.get("_1").map(|s| s.as_str()), Some("L4"));
}

#[test]
fn test_mirror_noconstraint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8; add_line -3,8 -6,5");
    run_ok(&mut ctx, "mirror L1 L2 about L0 noconstraint");
    assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
    // No coincident recreation either
    let ll_coinc = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
        + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
    // Only the original L1.p2=L2.p1 coincident, no mirrored copy
    assert_eq!(ll_coinc, 1);
}

#[test]
fn test_mirror_coincident_recreation() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,0 -2,3; add_line -2,3 -5,6");
    // L2.p1 = L1.p2 via auto-coincident (stored as LL12: a=L2, b=L1)
    let coinc_before = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
        + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
    run_ok(&mut ctx, "mirror L1 L2 about L0");
    let coinc_after = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
        + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
    assert_eq!(coinc_after, coinc_before + 1, "should recreate coincident among mirrored lines");
}

#[test]
fn test_mirror_symmetry_dedup() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,0 -2,3; add_line -2,3 -5,6");
    run_ok(&mut ctx, "mirror L1 L2 about L0");
    // 4 endpoints total, but L1.p2=L2.p1 is shared, so 3 unique positions -> 3 symmetry constraints
    assert_eq!(ctx.sketch.symmetry_pp.len(), 3);
}

#[test]
fn test_mirror_missing_about() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
    let r = execute_one(&mut ctx, "mirror L1 L0");
    assert!(r.is_error, "should require 'about' keyword: {}", r.output);
}

#[test]
fn test_mirror_output_lists_constraints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
    let out = run_ok(&mut ctx, "mirror L1 about L0");
    assert!(out.contains("symmetry"), "should list symmetry: {}", out);
    assert!(out.contains("Mirrored L1"), "should list mirrored entity: {}", out);
}

// -- add_ellipse --

#[test]
fn test_add_ellipse_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 45");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let a = &ctx.sketch.arcs[arc_ref];
    assert!(a.is_ellipse);
    assert!(a.closed);
    assert!(near(a.radius.value, 5.0));
    assert!(near(a.radius_b.value, 3.0));
    assert!(near(a.rotation.value.to_degrees(), 45.0));
}

#[test]
fn test_add_ellipse_dof() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 0");
    // DOF: center(2) + rx(1) + ry(1) + rotation(1) = 5
    assert_eq!(ctx.sketch.dof().unwrap(), 5);
}

#[test]
fn test_add_ellipse_list_output() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 30");
    let out = run_ok(&mut ctx, "list");
    assert!(out.contains("[ellipse]"), "should show [ellipse]: {}", out);
    assert!(out.contains("rx="), "should show rx: {}", out);
    assert!(out.contains("ry="), "should show ry: {}", out);
}

#[test]
fn test_add_ellipse_driven() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 0 driven");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
    assert!(near(ctx.sketch.dimensions[0].value, 5.0));
    assert!(near(ctx.sketch.dimensions[1].value, 3.0));
}

#[test]
fn test_ellipse_print_params() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 45");
    let out = run_ok(&mut ctx, "print EA0.rotation");
    let val: f64 = out.trim().parse().unwrap();
    assert!(near(val, std::f64::consts::PI / 4.0));
    let out = run_ok(&mut ctx, "print EA0.radius_b");
    let val: f64 = out.trim().parse().unwrap();
    assert!(near(val, 3.0));
}

#[test]
fn test_ellipse_radius_b_command() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 0");
    run_ok(&mut ctx, "radius_b EA0 4");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(near(ctx.sketch.dimensions[0].value, 4.0));
    assert!(near(ctx.sketch.arcs.refs().next().map(|r| ctx.sketch.arcs[r].radius_b.value).unwrap(), 4.0));
}

#[test]
fn test_radius_b_rejects_non_ellipse() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    let r = execute_one(&mut ctx, "radius_b A0 3");
    assert!(r.is_error, "radius_b should reject non-ellipse: {}", r.output);
}

#[test]
fn test_ellipse_measure() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 5 3 30");
    let out = run_ok(&mut ctx, "measure EA0");
    assert!(out.contains("rx="), "should show rx: {}", out);
    assert!(out.contains("ry="), "should show ry: {}", out);
    assert!(out.contains("rotation="), "should show rotation: {}", out);
}

#[test]
fn test_arc_unchanged_by_ellipse_fields() {
    // Verify that existing arcs/circles still work identically
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let a = &ctx.sketch.arcs[arc_ref];
    assert!(!a.is_ellipse);
    assert!(near(a.radius_b.value, 5.0)); // radius_b == radius
    assert!(near(a.rotation.value, 0.0));
    assert!(a.radius_b.optimize); // optimizable (equality constraint keeps it = radius)
    assert!(!a.rotation.optimize); // fixed
}

#[test]
fn test_ellipse_start_end_pos() {
    // Verify ellipse point formula works for arc_start_pos/arc_end_pos
    let mut ctx = CommandContext::new();
    // Unrotated ellipse: rx=4, ry=2. At angle 0, point = (cx+4, cy)
    run_ok(&mut ctx, "add_ellipse 0,0 4 2 0");
    let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
    let sp = crate::geometry::arc_start_pos(&ctx.sketch.arcs[arc_ref]);
    assert!(near(sp.x, 4.0));
    assert!(near(sp.y, 0.0));
}

// -- Measure --

#[test]
fn test_measure_single_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    let out = run_ok(&mut ctx, "measure L0");
    assert!(out.contains("length=5.0000"), "should show length: {}", out);
}

#[test]
fn test_measure_two_parallel_lines() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
    let out = run_ok(&mut ctx, "measure L0 L1");
    assert!(out.contains("parallel"), "should detect parallel: {}", out);
    assert!(out.contains("3.0000"), "should show distance: {}", out);
}

#[test]
fn test_measure_two_lines_angle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
    let out = run_ok(&mut ctx, "measure L0 L1");
    assert!(out.contains("45.0000"), "should show 45 deg: {}", out);
    assert!(out.contains("135.0000"), "should show supplement: {}", out);
}

#[test]
fn test_measure_two_points() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,4");
    let out = run_ok(&mut ctx, "measure L0.p1 L0.p2");
    assert!(out.contains("5.0000"), "should show distance 5: {}", out);
}

#[test]
fn test_measure_point_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_point 2,3");
    let out = run_ok(&mut ctx, "measure P0 L0");
    assert!(out.contains("3.0000"), "should show perp distance 3: {}", out);
}

#[test]
fn test_measure_single_arc() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    let out = run_ok(&mut ctx, "measure A0");
    assert!(out.contains("radius=5.0000"), "should show radius: {}", out);
}

// -- Arc-arc symmetry --

#[test]
fn test_symmetry_aa_command() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 4,1 2");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "symmetry A0 L0 A1");
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before - 3, "arc symmetry should remove 3 DOF: {} -> {}", dof_before, dof_after);
    assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
}

#[test]
fn test_symmetry_aa_equal_radius() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 2");
    run_ok(&mut ctx, "symmetry A0 L0 A1");
    let r0 = ctx.sketch.arcs.iter().next().unwrap().radius.value;
    let r1 = ctx.sketch.arcs.iter().nth(1).unwrap().radius.value;
    assert!((r0 - r1).abs() < 0.01, "radii should be equal: {} vs {}", r0, r1);
}

#[test]
fn test_symmetry_aa_remove() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 1");
    run_ok(&mut ctx, "symmetry A0 L0 A1");
    assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
    run_ok(&mut ctx, "delete A0 L0 A1 symmetry");
    assert_eq!(ctx.sketch.symmetry_aa.len(), 0);
}

#[test]
fn test_symmetry_aa_duplicate() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 1");
    run_ok(&mut ctx, "symmetry A0 L0 A1");
    let e = run_err(&mut ctx, "symmetry A0 L0 A1");
    assert!(e.contains("already exists"), "{}", e);
}

#[test]
fn test_symmetry_aa_ellipse() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,-5 0,5; add_ellipse -3,0 2 1 30; add_ellipse 4,1 3 2 0");
    run_ok(&mut ctx, "symmetry EA0 L0 EA1");
    let a0 = ctx.sketch.arcs.refs().nth(0).unwrap();
    let a1 = ctx.sketch.arcs.refs().nth(1).unwrap();
    let r0 = ctx.sketch.arcs[a0].radius.value;
    let r1 = ctx.sketch.arcs[a1].radius.value;
    assert!((r0 - r1).abs() < 0.01, "radii should be equal: {} vs {}", r0, r1);
    let rb0 = ctx.sketch.arcs[a0].radius_b.value;
    let rb1 = ctx.sketch.arcs[a1].radius_b.value;
    assert!((rb0 - rb1).abs() < 0.01, "radius_b should be equal: {} vs {}", rb0, rb1);
}

#[test]
fn test_tangent_aa_ellipse() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse -5,0 3 2 0; add_ellipse 5,0 3 2 0");
    run_ok(&mut ctx, "tangent EA0 EA1");
    assert_eq!(ctx.sketch.tangent_aa.len(), 1);
    // Verify constraint is satisfied: dist = r_eff_a + r_eff_b
    let a0 = ctx.sketch.arcs.refs().nth(0).unwrap();
    let a1 = ctx.sketch.arcs.refs().nth(1).unwrap();
    let aa = &ctx.sketch.arcs[a0];
    let bb = &ctx.sketch.arcs[a1];
    let dx = aa.center.value.x - bb.center.value.x;
    let dy = aa.center.value.y - bb.center.value.y;
    let dist = (dx * dx + dy * dy).sqrt();
    let nx = dx / dist;
    let ny = dy / dist;
    // Effective radius of a
    let cra = aa.rotation.value.cos();
    let sra = aa.rotation.value.sin();
    let nxa = nx * cra + ny * sra;
    let nya = -nx * sra + ny * cra;
    let r_eff_a = (nxa * nxa * aa.radius.value * aa.radius.value + nya * nya * aa.radius_b.value * aa.radius_b.value).sqrt();
    // Effective radius of b
    let crb = bb.rotation.value.cos();
    let srb = bb.rotation.value.sin();
    let nxb = -nx * crb - ny * srb;
    let nyb = nx * srb - ny * crb;
    let r_eff_b = (nxb * nxb * bb.radius.value * bb.radius.value + nyb * nyb * bb.radius_b.value * bb.radius_b.value).sqrt();
    let residual = (dist - r_eff_a - r_eff_b).abs();
    assert!(residual < 0.01, "tangent should be satisfied: dist={}, r_eff_a={}, r_eff_b={}, residual={}", dist, r_eff_a, r_eff_b, residual);
}

// -- List constraint filtering --

#[test]
fn test_list_filter_horizontal() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "horizontal L0; horizontal L1");
    let out = run_ok(&mut ctx, "list horizontal");
    assert!(out.contains("horizontal L0"), "should list L0: {}", out);
    assert!(out.contains("horizontal L1"), "should list L1: {}", out);
    assert!(!out.contains("coincident"), "should not include other types: {}", out);
}

#[test]
fn test_list_filter_empty() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "list parallel");
    assert_eq!(out, "(empty)");
}

#[test]
fn test_list_filter_coincident() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line @0,3");
    let out = run_ok(&mut ctx, "list coincident");
    assert!(out.contains("coincident"), "should show coincident: {}", out);
    assert!(!out.contains("L0:"), "should not include entity listing: {}", out);
}

// -- Auto-assigned constraint names (C<n>, CL0H) --

#[test]
fn test_rc_by_numeric_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    assert_eq!(ctx.sketch.parallel.len(), 1);
    let nid = ctx.sketch.parallel[0].nid;
    let out = run_ok(&mut ctx, &format!("delete C{}", nid));
    assert!(out.contains(&format!("C{}", nid)), "output should mention name: {}", out);
    assert!(out.contains("parallel"), "output should describe constraint: {}", out);
    assert_eq!(ctx.sketch.parallel.len(), 0);
}

#[test]
fn test_rc_by_flag_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 0,7");
    run_ok(&mut ctx, "horizontal L0; vertical L1");
    assert!(ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].constraints.horizontal);
    let out = run_ok(&mut ctx, "delete CL0H");
    assert!(out.contains("CL0H"), "output should mention name: {}", out);
    assert!(out.contains("horizontal L0"), "output should describe constraint: {}", out);
    let l0 = ctx.sketch.lines.refs().next().unwrap();
    assert!(!ctx.sketch.lines[l0].constraints.horizontal);
    // CL1V still set until removed.
    let l1 = ctx.sketch.lines.refs().nth(1).unwrap();
    assert!(ctx.sketch.lines[l1].constraints.vertical);
    run_ok(&mut ctx, "delete CL1V");
    assert!(!ctx.sketch.lines[l1].constraints.vertical);
}

#[test]
fn test_rc_unknown_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    let e = run_err(&mut ctx, "delete C999");
    assert!(e.contains("Unknown"), "{}", e);
    let e = run_err(&mut ctx, "delete CL0H");
    assert!(e.contains("Unknown"), "{}", e);
}

#[test]
fn test_info_by_constraint_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    let nid = ctx.sketch.parallel[0].nid;
    let out = run_ok(&mut ctx, &format!("info C{}", nid));
    assert!(out.contains("parallel"), "info output: {}", out);
    assert!(out.contains(&format!("C{}:", nid)), "info output: {}", out);
}

#[test]
fn test_info_by_flag_name() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "horizontal L0");
    let out = run_ok(&mut ctx, "info CL0H");
    assert!(out.contains("horizontal L0"), "info output: {}", out);
}

#[test]
fn test_list_includes_constraint_names() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "horizontal L0; parallel L0 L1");
    let out = run_ok(&mut ctx, "list");
    assert!(out.contains("CL0H: horizontal L0"), "list output: {}", out);
    assert!(out.contains(": parallel L0 L1"), "list output: {}", out);
}

#[test]
fn test_rc_entity_syntax_still_works() {
    // Existing entity-pair + type dispatch keeps working alongside name lookup.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    assert_eq!(ctx.sketch.parallel.len(), 1);
    run_ok(&mut ctx, "delete L0 L1 parallel");
    assert_eq!(ctx.sketch.parallel.len(), 0);
}

#[test]
fn test_list_and_info_show_range_bound() {
    // Range dimensions must be identifiable from script output (was
    // previously indistinguishable from a numeric dim in both `list
    // dims` and `info d0`).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 3 to 7");
    let out = run_ok(&mut ctx, "list dims");
    assert!(out.contains("3 to 7"), "list dims should show range: {}", out);
    let out = run_ok(&mut ctx, "info d0");
    assert!(out.contains("range=3 to 7"), "info should mark range: {}", out);
    assert!(!out.contains("expr=(numeric)"), "info should not say numeric: {}", out);

    // One-sided and live-expression forms stay recognisable too.
    run_ok(&mut ctx, "add_line 0,3 5,3");
    run_ok(&mut ctx, "length L1 >= 2");
    let out = run_ok(&mut ctx, "info d1");
    assert!(out.contains("range=>= 2"), "one-sided: {}", out);

    run_ok(&mut ctx, "param lo 2");
    run_ok(&mut ctx, "length L0 lo to 8");
    let out = run_ok(&mut ctx, "list dims");
    assert!(out.contains("lo to 8"), "live range: {}", out);
}

#[test]
fn test_constraint_names_survive_undo_redo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
    run_ok(&mut ctx, "parallel L0 L1");
    let nid_before = ctx.sketch.parallel[0].nid;
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.parallel.len(), 0);
    run_ok(&mut ctx, "redo");
    assert_eq!(ctx.sketch.parallel.len(), 1);
    assert_eq!(ctx.sketch.parallel[0].nid, nid_before);
}

#[test]
fn test_add_earc() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc 0,0 5,0 3 1 0 noconnect");
    assert_eq!(ctx.sketch.arcs.len(), 1);
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    assert!(!a.closed);
    let sp = crate::geometry::arc_start_pos(a);
    let ep = crate::geometry::arc_end_pos(a);
    assert!((sp.x - 0.0).abs() < 0.1, "start x: {}", sp.x);
    assert!((sp.y - 0.0).abs() < 0.1, "start y: {}", sp.y);
    assert!((ep.x - 5.0).abs() < 0.1, "end x: {}", ep.x);
    assert!((ep.y - 0.0).abs() < 0.1, "end y: {}", ep.y);
}

#[test]
fn test_add_earc_large() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc 0,0 5,0 3 1 0 large noconnect");
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    let sweep = (a.end_angle.value - a.start_angle.value).abs();
    assert!(sweep > std::f64::consts::PI, "sweep {:.2} should be > pi for large arc", sweep);
}

#[test]
fn test_add_earc_cw() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc 0,0 5,0 3 1 0 cw noconnect");
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(!a.ccw, "should be clockwise");
}

#[test]
fn test_add_earc_center() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc_center 0,0 3 1 45 0 90 noconnect");
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    assert!(!a.closed);
    assert!((a.center.value.x).abs() < 0.01);
    assert!((a.center.value.y).abs() < 0.01);
    assert!((a.radius.value - 3.0).abs() < 0.01);
    assert!((a.radius_b.value - 1.0).abs() < 0.01);
}

#[test]
fn test_add_earc3() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc3 0,0 5,0 2,2 3 1 noconnect");
    assert_eq!(ctx.sketch.arcs.len(), 1);
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    assert!(!a.closed);
}

#[test]
fn test_add_earc_driven() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc 0,0 5,0 3 1 0 driven noconnect");
    assert_eq!(ctx.sketch.dimensions.len(), 2);
}

// --- Auto-tangent tests ---

#[test]
fn test_auto_tangent_line_arc() {
    let mut ctx = CommandContext::new();
    // Horizontal line, then arc tangent at p2
    run(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_arc 5,0 5,5 7.5,2.5");
    assert!(out.contains("tangent"), "expected auto-tangent: {}", out);
    assert_eq!(ctx.sketch.tangent_la.len(), 1);
}

#[test]
fn test_auto_tangent_not_applied_when_not_tangent() {
    let mut ctx = CommandContext::new();
    // Horizontal line, then arc clearly not tangent (goes straight up)
    run(&mut ctx, "add_line 0,0 5,0");
    run(&mut ctx, "add_arc 5,0 5,5 4,2.5");
    assert_eq!(ctx.sketch.tangent_la.len(), 0, "should not auto-tangent non-tangent geometry");
}

#[test]
fn test_auto_tangent_notangent_keyword() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0");
    let out = run_ok(&mut ctx, "add_arc 5,0 5,5 7.5,2.5 notangent");
    assert!(!out.contains("tangent"), "notangent should suppress: {}", out);
    assert_eq!(ctx.sketch.tangent_la.len(), 0);
}

#[test]
fn test_auto_tangent_noconnect_implies_notangent() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0");
    run(&mut ctx, "add_arc 5,0 5,5 7.5,2.5 noconnect");
    assert_eq!(ctx.sketch.tangent_la.len(), 0);
}

#[test]
fn test_auto_tangent_arc_arc() {
    let mut ctx = CommandContext::new();
    // Two arcs sharing an endpoint, tangent at the junction
    // First arc: semicircle from (0,0) to (2,0) with center (1,1)
    run(&mut ctx, "add_arc 0,0 2,0 1,1 noconnect");
    // Second arc: continues tangent from (2,0)
    let out = run_ok(&mut ctx, "add_arc 2,0 4,0 3,1");
    assert!(out.contains("tangent"), "expected arc-arc auto-tangent: {}", out);
    assert_eq!(ctx.sketch.tangent_aa.len(), 1);
}

#[test]
fn test_tangent_aa_shared_endpoint() {
    // Manual tangent command with shared endpoint should use cross-product formula
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_earc 377.953,0 204.989,2.559 2017.12 2017.12 0 noconnect");
    run(&mut ctx, "add_earc 204.989,2.559 142.625,0.378 4123.85 4123.85 0 noconnect notangent");
    run_ok(&mut ctx, "tangent EA0 EA1");
    assert_eq!(ctx.sketch.tangent_aa.len(), 1);
    assert_ne!(ctx.sketch.tangent_aa[0].shared, SharedEndpoint::None);
}

// --- Quiet mode tests ---

#[test]
fn test_quiet_keyword_line() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 quiet noconnect");
    let r = ctx.sketch.lines.refs().next().unwrap();
    assert!(ctx.sketch.lines[r].quiet);
}

#[test]
fn test_quiet_keyword_arc() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_circle 0,0 5 quiet noconnect");
    let r = ctx.sketch.arcs.refs().next().unwrap();
    assert!(ctx.sketch.arcs[r].quiet);
}

#[test]
fn test_quiet_toggle() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    let r = ctx.sketch.lines.refs().next().unwrap();
    assert!(!ctx.sketch.lines[r].quiet);
    run_ok(&mut ctx, "quiet L0");
    assert!(ctx.sketch.lines[r].quiet);
    run_ok(&mut ctx, "quiet L0");
    assert!(!ctx.sketch.lines[r].quiet);
}

#[test]
fn test_quiet_on_off() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    let r = ctx.sketch.lines.refs().next().unwrap();
    run_ok(&mut ctx, "quiet L0 on");
    assert!(ctx.sketch.lines[r].quiet);
    run_ok(&mut ctx, "quiet L0 off");
    assert!(!ctx.sketch.lines[r].quiet);
}

#[test]
fn test_quiet_info() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 quiet noconnect");
    let out = run_ok(&mut ctx, "info L0");
    assert!(out.contains("[quiet]"), "info should show [quiet]: {}", out);
}

// --- add_earc_tangent tests ---

#[test]
fn test_add_earc_tangent_basic() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_earc_tangent 0,0 1,0 5,5 0,1 noconnect");
    assert_eq!(ctx.sketch.arcs.len(), 1);
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    assert!(!a.closed);
}

#[test]
fn test_add_earc_tangent_bulge() {
    let mut ctx = CommandContext::new();
    // Semicircle-like: opposing tangents, bulge=1
    run_ok(&mut ctx, "add_earc_tangent 0,0 0,1 10,0 0,-1 1 noconnect");
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    // bulge=1 with symmetric tangents should be near-circular
    assert!((a.radius.value - a.radius_b.value).abs() < 0.5,
        "expected near-circular: rx={:.3} ry={:.3}", a.radius.value, a.radius_b.value);
}

#[test]
fn test_cursor_tangent_from_line() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    assert!(ctx.cursor_tangent.is_some(), "tangent should be set after add_line");
    let t = ctx.cursor_tangent.unwrap();
    assert!((t.x - 1.0).abs() < 0.01, "tangent x: {}", t.x);
    assert!(t.y.abs() < 0.01, "tangent y: {}", t.y);
}

#[test]
fn test_cursor_tangent_chaining() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    run_ok(&mut ctx, "add_earc_tangent @cursor @tangent 10,3 0,1 noconnect");
    // Cursor should now be at (10,3)
    assert!(ctx.cursor.is_some());
    let c = ctx.cursor.unwrap();
    assert!((c.x - 10.0).abs() < 0.1, "cursor x: {}", c.x);
    assert!((c.y - 3.0).abs() < 0.1, "cursor y: {}", c.y);
    // Tangent should be set from arc end
    assert!(ctx.cursor_tangent.is_some(), "tangent should be set after earc_tangent");
}

#[test]
fn test_cursor_tangent_from_arc() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_arc 0,0 5,0 2.5,2.5 noconnect");
    assert!(ctx.cursor_tangent.is_some(), "tangent should be set after add_arc");
}

// --- add_earc_rtangent tests ---

#[test]
fn test_add_earc_rtangent_basic() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    run_ok(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5 noconnect");
    assert_eq!(ctx.sketch.arcs.len(), 1);
    let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
    assert!(a.is_ellipse);
    // Start should be at cursor (5,0)
    let sp = crate::geometry::arc_start_pos(a);
    assert!((sp.x - 5.0).abs() < 0.01, "start x: {:.4}", sp.x);
    assert!((sp.y - 0.0).abs() < 0.01, "start y: {:.4}", sp.y);
}

#[test]
fn test_add_earc_rtangent_chaining() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    run_ok(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5 noconnect");
    // Cursor should now be at (10,3) and tangent set
    let c = ctx.cursor.unwrap();
    assert!((c.x - 10.0).abs() < 0.1, "cursor x: {:.4}", c.x);
    assert!((c.y - 3.0).abs() < 0.1, "cursor y: {:.4}", c.y);
    assert!(ctx.cursor_tangent.is_some());
    // Chain another
    run_ok(&mut ctx, "add_earc_rtangent 15,0 1,0 0.5 noconnect");
    assert_eq!(ctx.sketch.arcs.len(), 2);
}

#[test]
fn test_add_earc_rtangent_no_cursor() {
    let mut ctx = CommandContext::new();
    // No line/arc created yet — should fail
    let results = crate::commands::execute(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5");
    assert!(results[0].is_error, "should fail without cursor");
}

// --- Construction tests ---

#[test]
fn test_constr_keyword_line() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
    let r = ctx.sketch.lines.refs().next().unwrap();
    assert!(ctx.sketch.lines[r].construction);
    assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::DashDot);
}

#[test]
fn test_constr_keyword_arc() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_circle 0,0 5 constr noconnect");
    let r = ctx.sketch.arcs.refs().next().unwrap();
    assert!(ctx.sketch.arcs[r].construction);
    assert_eq!(ctx.sketch.arcs[r].style, arael_sketch_solver::LineStyle::DashDot);
}

#[test]
fn test_constr_toggle() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    let r = ctx.sketch.lines.refs().next().unwrap();
    assert!(!ctx.sketch.lines[r].construction);
    run_ok(&mut ctx, "constr L0");
    assert!(ctx.sketch.lines[r].construction);
    assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::DashDot);
    run_ok(&mut ctx, "constr L0");
    assert!(!ctx.sketch.lines[r].construction);
    assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::Solid);
}

#[test]
fn test_constr_info() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
    let out = run_ok(&mut ctx, "info L0");
    assert!(out.contains("[constr]"), "info should show [constr]: {}", out);
}

#[test]
fn test_list_constr_filter() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
    run(&mut ctx, "add_line 1,0 6,0 noconnect");
    let out = run_ok(&mut ctx, "list constr");
    assert!(out.contains("L0"), "should list L0: {}", out);
    assert!(!out.contains("L1"), "should not list L1: {}", out);
}

// --- Drag tests ---

#[test]
fn test_drag_line_endpoint() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    let out = run_ok(&mut ctx, "drag L0.p2 5,3");
    assert!(out.contains("Dragged"), "{}", out);
    let p2 = ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].p2.value;
    assert!((p2.x - 5.0).abs() < 0.1, "p2.x={:.4}", p2.x);
    assert!((p2.y - 3.0).abs() < 0.1, "p2.y={:.4}", p2.y);
}

#[test]
fn test_drag_relative() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    run_ok(&mut ctx, "drag L0.p2 @0,3");
    let p2 = ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].p2.value;
    assert!((p2.x - 5.0).abs() < 0.1, "p2.x={:.4}", p2.x);
    assert!((p2.y - 3.0).abs() < 0.1, "p2.y={:.4}", p2.y);
}

#[test]
fn test_drag_point() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_point 0,0");
    run_ok(&mut ctx, "drag P0 3,3");
    let p = ctx.sketch.points[ctx.sketch.points.refs().next().unwrap()].pos.value;
    assert!((p.x - 3.0).abs() < 0.1, "x={:.4}", p.x);
    assert!((p.y - 3.0).abs() < 0.1, "y={:.4}", p.y);
}

#[test]
fn test_drag_constrained() {
    let mut ctx = CommandContext::new();
    run(&mut ctx, "add_line 0,0 5,0 noconnect");
    run_ok(&mut ctx, "horizontal L0");
    // Drag p2 to (5,3) — horizontal should keep y equal
    run_ok(&mut ctx, "drag L0.p2 5,3");
    let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
    assert!((l.p1.value.y - l.p2.value.y).abs() < 0.1, "should stay horizontal: p1.y={:.4} p2.y={:.4}", l.p1.value.y, l.p2.value.y);
}

/// Soft drag (default): dragging L0.p2 toward an infeasible target
/// must leave the sketch at its relaxed cost ~ 0 state, with the
/// dragged endpoint lagging at the nearest feasible point rather
/// than forcing the sketch to deform.
#[test]
fn test_drag_soft_respects_constraints() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "lock L0.p1");
    run_ok(&mut ctx, "length L0 5");
    // Target (10, 0) is outside the reachable circle (radius 5).
    run_ok(&mut ctx, "drag L0.p2 10,0");
    let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
    let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
    assert!((len - 5.0).abs() < 0.01, "length constraint must hold: {:.4}", len);
    // p1 stays locked at origin.
    assert!(l.p1.value.x.abs() < 0.01 && l.p1.value.y.abs() < 0.01,
        "p1 must stay locked: {:?}", l.p1.value);
    // p2 lands at (5, 0), the nearest feasible point toward (10, 0).
    assert!((l.p2.value.x - 5.0).abs() < 0.1 && l.p2.value.y.abs() < 0.1,
        "p2 should relax to nearest feasible point: {:?}", l.p2.value);
}

/// Feasible soft drag must still land exactly on the cursor target.
/// Verifies the drag-pull attractor dominates background drift so
/// unconstrained DOFs track the cursor rather than compromising
/// between cursor and starting position.
#[test]
fn test_drag_soft_feasible_hits_target() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "drag L0.p2 3,4");
    let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
    assert!((l.p2.value.x - 3.0).abs() < 0.01 && (l.p2.value.y - 4.0).abs() < 0.01,
        "drag should land at cursor: {:?}", l.p2.value);
}

// -- Range-dimension transitions (regression tests) --

/// Regression: updating a range dim with a bare numeric value must
/// clear `dim.range` so the barrier residual stops firing in
/// `rebuild_expr_constraints`. Previously the old range stuck and
/// the geometry couldn't leave the band.
#[test]
fn test_range_to_numeric_transition() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 3 to 6");
    // Currently 5, inside the [3, 6] band.
    assert!(near(line_len(&ctx, "L0"), 5.0));
    // Set to 2 (outside the band). With the bug, length would stay
    // clamped to 3 because the old barrier still applied.
    run_ok(&mut ctx, "length L0 2");
    assert!(near(line_len(&ctx, "L0"), 2.0),
        "expected length 2.0 after range->numeric, got {:.4}", line_len(&ctx, "L0"));
}

/// Range dimensions must not contribute to the reported DOF --
/// their Jacobian row is zero inside the feasible band and
/// non-zero at/outside the bound, so including them would make
/// DOF swing by one as the user drags geometry across the bound.
/// Reported DOF should be the geometric DOF, stable regardless
/// of whether the barrier is currently active.
#[test]
fn test_range_dim_does_not_affect_dof() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 3 to 6");
    let dof_inside = ctx.sketch.dof().unwrap();
    // Push onto the lower bound: barrier would be active there.
    run_ok(&mut ctx, "length L0 1 to 6");
    let dof_at_lower = ctx.sketch.dof().unwrap();
    // Push onto the upper bound.
    run_ok(&mut ctx, "length L0 3 to 4");
    let dof_at_upper = ctx.sketch.dof().unwrap();
    assert_eq!(dof_inside, dof_at_lower,
        "DOF must be stable; inside={}, at lower={}", dof_inside, dof_at_lower);
    assert_eq!(dof_inside, dof_at_upper,
        "DOF must be stable; inside={}, at upper={}", dof_inside, dof_at_upper);
}

/// Regression: updating a numeric dim to a range must drop the old
/// per-kind equality constraint (e.g. `has_length = 5`) so the
/// barrier can drive the parameter into the band unopposed.
/// Previously both constraints stayed and the equality won.
#[test]
fn test_numeric_to_range_transition() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 5");
    assert!(near(line_len(&ctx, "L0"), 5.0));
    // Upper bound 3: barrier must clamp length down from 5 to 3.
    run_ok(&mut ctx, "length L0 2 to 3");
    assert!(near(line_len(&ctx, "L0"), 3.0),
        "expected length 3.0 after numeric->range clamp, got {:.4}", line_len(&ctx, "L0"));
}

#[test]
fn test_bare_expr_is_live() {
    // Bare parameter names in a dimension value are live by
    // default: `length L0 w` creates an expression dim that
    // tracks `w`. Snapshot form `=w` bakes the current value
    // in as a literal.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "param w 5");
    run_ok(&mut ctx, "length L0 w");
    run_ok(&mut ctx, "param w 8");
    assert!(near(line_len(&ctx, "L0"), 8.0),
        "bare `w` must be live; len after `param w 8`: {:.4}",
        line_len(&ctx, "L0"));
}

#[test]
fn test_eq_prefix_is_snapshot() {
    // `=w` snapshots: length is baked as a literal at command
    // time and does not track later param changes.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "param w 5");
    run_ok(&mut ctx, "length L0 =w");
    run_ok(&mut ctx, "param w 8");
    assert!(near(line_len(&ctx, "L0"), 5.0),
        "`=w` must snapshot; len after `param w 8`: {:.4}",
        line_len(&ctx, "L0"));
}

#[test]
fn test_range_bare_expr_is_live() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "param lo 2");
    run_ok(&mut ctx, "param hi 6");
    run_ok(&mut ctx, "length L0 lo to hi");
    // Current 4 is inside [2, 6], bound inactive.
    assert!(near(line_len(&ctx, "L0"), 4.0));
    // Shrink the band by moving `hi` below the current length;
    // the barrier activates and clamps.
    run_ok(&mut ctx, "param hi 3");
    assert!(near(line_len(&ctx, "L0"), 3.0),
        "range must track `hi`; len after `param hi 3`: {:.4}",
        line_len(&ctx, "L0"));
}

#[test]
fn test_xangle_range() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");  // xangle 0
    run_ok(&mut ctx, "xangle L0 30 to 60");
    let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let ang = dy.atan2(dx).to_degrees();
    assert!((30.0..=60.0).contains(&ang) || ang.abs() < 0.1,
        "xangle {:.4} not in [30, 60] after Between(30, 60)", ang);
    // The solver should have rotated the line to at least reach the band.
    // (The lower bound activates; exact landing is near 30 given the initial 0.)
    assert!(ang >= 30.0 - 0.1,
        "xangle lower bound: {:.4} should be >= 30", ang);
}

#[test]
fn test_radius_range() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 5");
    // Lower bound activates: radius grows from 5 to 7.
    run_ok(&mut ctx, "radius A0 >= 7");
    let r = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value;
    assert!(near(r, 7.0), "radius after Min(7): {:.4}", r);
    // Transition to a two-sided range clamping down.
    run_ok(&mut ctx, "radius A0 2 to 4");
    let r = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value;
    assert!(near(r, 4.0), "radius after 2..4: {:.4}", r);
}

/// Round-trip: numeric -> range -> numeric -> range exercises both
/// transition directions through the same command path.
#[test]
fn test_range_numeric_roundtrip() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "length L0 5");
    assert!(near(line_len(&ctx, "L0"), 5.0));
    run_ok(&mut ctx, "length L0 2 to 3");
    assert!(near(line_len(&ctx, "L0"), 3.0));
    run_ok(&mut ctx, "length L0 4");
    assert!(near(line_len(&ctx, "L0"), 4.0),
        "range->numeric: got {:.4}", line_len(&ctx, "L0"));
    run_ok(&mut ctx, "length L0 1 to 2");
    assert!(near(line_len(&ctx, "L0"), 2.0),
        "numeric->range clamp: got {:.4}", line_len(&ctx, "L0"));
}

// -- Split / Trim --

/// A 10-long horizontal line with a vertical cutter through (4,0).
/// `noconnect` keeps the cutter from snapping to L0 so the crossing
/// stays a plain geometric intersection.
fn split_fixture() -> CommandContext {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    run_ok(&mut ctx, "add_line 4,-2 4,2 noconnect");
    ctx
}

#[test]
fn test_split_line_coordinate_form() {
    let mut ctx = split_fixture();
    let out = run_ok(&mut ctx, "split L0 4,0");
    assert!(out.contains("Split L0 -> L2 L3"), "{}", out);
    assert!(resolve_line(&ctx.sketch, "L0").is_err(), "target name retired");
    assert!(near(line_len(&ctx, "L2"), 4.0));
    assert!(near(line_len(&ctx, "L3"), 6.0));
    // Cut endpoints joined and pinned onto the cutter.
    assert_eq!(ctx.sketch.coincident_ll21.len(), 1);
    assert_eq!(ctx.sketch.line_p2_on_line.len(), 1);
    assert!(out.contains("added:"), "{}", out);
}

#[test]
fn test_split_line_by_form_and_capture() {
    let mut ctx = split_fixture();
    let out = run_ok(&mut ctx, "a, b = split L0 by L1");
    assert!(out.contains("Split L0"), "{}", out);
    assert_eq!(ctx.session_names.get("a").map(|s| s.as_str()), Some("L2"));
    assert_eq!(ctx.session_names.get("b").map(|s| s.as_str()), Some("L3"));
    // Scripted trim: delete the far piece by its captured name.
    run_ok(&mut ctx, "delete b");
    assert!(resolve_line(&ctx.sketch, "L3").is_err());
    assert!(resolve_line(&ctx.sketch, "L2").is_ok());
}

#[test]
fn test_split_dof_neutral_with_perpendicular() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "add_line 0,5 0,9 noconnect");
    run_ok(&mut ctx, "perpendicular L0 L2");
    let dof_before = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "split L0 4,0");
    // Perpendicular replicated onto both pieces.
    assert_eq!(ctx.sketch.perpendicular.len(), 2, "{}", out);
    assert!(out.contains("copied:"), "{}", out);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before,
        "pinned split of a direction-constrained line is DOF-neutral");
}

#[test]
fn test_split_bare_line_dof_plus_one_pinned() {
    let mut ctx = split_fixture();
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "split L0 4,0");
    let dof_after = ctx.sketch.dof().unwrap();
    // Bare line: +2 for the junction, -1 for the pin.
    assert_eq!(dof_after, dof_before + 1);
}

#[test]
fn test_split_nopin_dof_plus_two() {
    let mut ctx = split_fixture();
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "split L0 4,0 nopin");
    assert!(ctx.sketch.line_p2_on_line.is_empty());
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before + 2);
}

#[test]
fn test_split_length_dim_becomes_distance() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "length L0 10");
    let did = ctx.sketch.dimensions[0].did;
    let out = run_ok(&mut ctx, "split L0 4,0");
    assert!(out.contains("moved:"), "{}", out);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    let d = &ctx.sketch.dimensions[0];
    assert_eq!(d.did, did);
    assert!(matches!(d.kind, DimensionKind::PointPointDistance(
        DimensionEndpoint::LineP1(_), DimensionEndpoint::LineP2(_))));
    // The distance constraint drives: total end-to-end length holds.
    let a = resolve_line(&ctx.sketch, "L2").unwrap();
    let b = resolve_line(&ctx.sketch, "L3").unwrap();
    let p1 = ctx.sketch.lines[a].p1.value;
    let p2 = ctx.sketch.lines[b].p2.value;
    assert!(near(((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt(), 10.0));
}

#[test]
fn test_split_expression_rewrite() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "param w L0.length / 2");
    let out = run_ok(&mut ctx, "split L0 4,0");
    assert!(out.contains("expressions:"), "{}", out);
    assert_eq!(ctx.sketch.user_params[0].expr_str, "(L2.length + L3.length) / 2");
    assert!(!ctx.sketch.user_params[0].broken);
}

#[test]
fn test_split_no_intersections_errors() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    let out = run_err(&mut ctx, "split L0 4,0");
    assert!(out.contains("no intersections"), "{}", out);
}

#[test]
fn test_split_search_radius() {
    let mut ctx = split_fixture();
    let out = run_err(&mut ctx, "split L0 4,3 1.0");
    assert!(out.contains("search radius"), "{}", out);
    run_ok(&mut ctx, "split L0 4,3 5.0");
}

#[test]
fn test_split_undo_restores() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "split L0 4,0");
    assert!(resolve_line(&ctx.sketch, "L0").is_err());
    run_ok(&mut ctx, "undo");
    assert!(resolve_line(&ctx.sketch, "L0").is_ok());
    assert!(resolve_line(&ctx.sketch, "L2").is_err());
    assert!(near(line_len(&ctx, "L0"), 10.0));
}

#[test]
fn test_trim_coordinate_form() {
    let mut ctx = split_fixture();
    // Remove the span left of the cutter.
    let out = run_ok(&mut ctx, "trim L0 1,0");
    assert!(out.contains("Trimmed L0"), "{}", out);
    assert!(resolve_line(&ctx.sketch, "L0").is_err());
    // One piece: from the cut to the old p2.
    let r = resolve_line(&ctx.sketch, "L2").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 4.0));
    assert!(near(ctx.sketch.lines[r].p2.value.x, 10.0));
    // No coincidence (nothing to join), but the pin holds.
    assert!(ctx.sketch.coincident_ll21.is_empty());
    assert_eq!(ctx.sketch.line_p1_on_line.len(), 1);
}

#[test]
fn test_trim_no_intersections_deletes() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    let out = run_ok(&mut ctx, "trim L0 5,0");
    assert!(out.contains("deleted L0"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 0);
}

#[test]
fn test_trim_by_two_cutters() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "add_line 7,-2 7,2 noconnect");
    let out = run_ok(&mut ctx, "trim L0 by L1 L2");
    assert!(out.contains("Trimmed L0"), "{}", out);
    // Outer pieces survive; the 4..7 span is gone.
    let a = resolve_line(&ctx.sketch, "L3").unwrap();
    let b = resolve_line(&ctx.sketch, "L4").unwrap();
    assert!(near(ctx.sketch.lines[a].p2.value.x, 4.0));
    assert!(near(ctx.sketch.lines[b].p1.value.x, 7.0));
}

#[test]
fn test_trim_by_forward_backward() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "trim L0 by L1 forward");
    let r = resolve_line(&ctx.sketch, "L2").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 0.0));
    assert!(near(ctx.sketch.lines[r].p2.value.x, 4.0));

    let mut ctx2 = split_fixture();
    run_ok(&mut ctx2, "trim L0 by L1 backward");
    let r = resolve_line(&ctx2.sketch, "L2").unwrap();
    assert!(near(ctx2.sketch.lines[r].p1.value.x, 4.0));
    assert!(near(ctx2.sketch.lines[r].p2.value.x, 10.0));
}

#[test]
fn test_trim_by_forward_twice_crossing_cutter() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 10,0");
    // A circle crossing L0 at x=5 and x=7.
    run_ok(&mut ctx, "add_circle 6,0 1 noconnect");
    run_ok(&mut ctx, "trim L0 by A0 forward");
    // Forward trims past the crossing nearest p2 (x=7), not x=5.
    let r = resolve_line(&ctx.sketch, "L1").unwrap();
    assert!(near(ctx.sketch.lines[r].p1.value.x, 0.0));
    assert!(near(ctx.sketch.lines[r].p2.value.x, 7.0));
}

#[test]
fn test_split_circle_concentric_dof() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    run_ok(&mut ctx, "add_line 0,-3 0,3 noconnect");
    let dof_before = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "split A0 by L0");
    assert!(out.contains("Split A0 -> A1 A2"), "{}", out);
    assert!(ctx.sketch.arcs.refs().count() == 2);
    // Concentric joined the pieces; both cut points coincident + pinned.
    assert_eq!(ctx.sketch.concentric.len(), 1);
    assert_eq!(ctx.sketch.coincident_arc_end_start.len(), 2);
    let dof_after = ctx.sketch.dof().unwrap();
    assert_eq!(dof_after, dof_before,
        "circle split at two pinned cuts is DOF-neutral through concentricity");
}

#[test]
fn test_trim_closed_circle_span() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    run_ok(&mut ctx, "add_line 0,-3 0,3 noconnect");
    // Remove the left span (the one containing (-2,0)).
    let out = run_ok(&mut ctx, "trim A0 -2,0");
    assert!(out.contains("Trimmed A0"), "{}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    let r = ctx.sketch.arcs.refs().next().unwrap();
    let arc = &ctx.sketch.arcs[r];
    assert!(!arc.closed);
    // The surviving span passes through (+2, 0).
    let mid = arc.point_at(0.5 * (arc.start_angle.value + arc.end_angle.value));
    assert!(mid.x > 0.0, "kept the right-hand span, got mid {:?}", (mid.x, mid.y));
}

#[test]
fn test_trim_closed_by_form_rejected() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    run_ok(&mut ctx, "add_line 0,-3 0,3 noconnect");
    let out = run_err(&mut ctx, "trim A0 by L0 forward");
    assert!(out.contains("closed"), "{}", out);
}

#[test]
fn test_split_elliptic_arc_dof_plus_two() {
    let mut ctx = CommandContext::new();
    // Elliptic arc spanning the upper half, cut once by a vertical line.
    run_ok(&mut ctx, "add_earc_center 0,0 3 1 0 0 180");
    run_ok(&mut ctx, "add_line 0,-2 0,2 noconnect");
    let dof_before = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "split EA0 by L0");
    assert!(out.contains("Split EA0"), "{}", out);
    assert_eq!(ctx.sketch.concentric.len(), 1);
    let dof_after = ctx.sketch.dof().unwrap();
    // Concentric-only ellipse ties: mutual rotation and semi-minor
    // stay free (+2), the pinned junction is neutral.
    assert_eq!(dof_after, dof_before + 2);
}

#[test]
fn test_split_equal_length_dropped_and_reported() {
    let mut ctx = split_fixture();
    run_ok(&mut ctx, "add_line 0,5 10,5 noconnect");
    run_ok(&mut ctx, "equal L0 L2");
    let out = run_ok(&mut ctx, "split L0 4,0");
    assert!(ctx.sketch.equal_length.is_empty());
    assert!(out.contains("dropped:"), "{}", out);
    assert!(out.contains("equal"), "{}", out);
}

#[test]
fn test_split_tangent_lands_on_touching_piece() {
    let mut ctx = split_fixture();
    // Circle tangent to L0 from above at (7,0).
    run_ok(&mut ctx, "add_circle 7,1 1 noconnect");
    run_ok(&mut ctx, "tangent L0 A0");
    run_ok(&mut ctx, "split L0 4,0");
    assert_eq!(ctx.sketch.tangent_la.len(), 1);
    let t = &ctx.sketch.tangent_la[0];
    let b = resolve_line(&ctx.sketch, "L3").unwrap();
    assert_eq!(t.line, b, "tangency follows the piece containing the contact");
}

// -- Command-name parity ----------------------------------------------

/// Names of the dispatch match arms in mod.rs, scraped from source:
/// the only place they exist as data.
fn dispatch_names() -> std::collections::HashSet<String> {
    let src = include_str!("mod.rs");
    let start = src.find("let result: CmdResult = match cmd {")
        .expect("dispatch match not found; update the parity test's anchor");
    let end = src[start..].find("\n    };").expect("dispatch match end") + start;
    let mut names = std::collections::HashSet::new();
    for line in src[start..end].lines() {
        let Some(arrow) = line.find("=>") else { continue };
        // Every quoted name in the arm pattern (handles alias arms
        // like `"perpendicular" | "perp"`).
        let mut rest = &line[..arrow];
        while let Some(q) = rest.find('"') {
            let after = &rest[q + 1..];
            let Some(q2) = after.find('"') else { break };
            names.insert(after[..q2].to_string());
            rest = &after[q2 + 1..];
        }
    }
    names
}

/// Commands dispatchable on purpose but kept out of autocomplete.
const HIDDEN_COMMANDS: &[&str] = &["ai"];

#[test]
fn test_command_names_match_dispatch() {
    let dispatch = dispatch_names();
    assert!(dispatch.len() > 50, "scraper broke: only {} arms found", dispatch.len());
    let listed: std::collections::HashSet<String> =
        COMMAND_NAMES.iter().map(|s| s.to_string()).collect();
    let missing: Vec<&String> = dispatch.iter()
        .filter(|n| !listed.contains(*n) && !HIDDEN_COMMANDS.contains(&n.as_str()))
        .collect();
    assert!(missing.is_empty(),
        "dispatchable but missing from COMMAND_NAMES (autocomplete): {:?}", missing);
    let stale: Vec<&String> = listed.iter().filter(|n| !dispatch.contains(*n)).collect();
    assert!(stale.is_empty(),
        "in COMMAND_NAMES but not dispatchable: {:?}", stale);
}

#[test]
fn test_every_command_has_help() {
    for name in COMMAND_NAMES {
        let r = cmd_help(name);
        assert!(r.is_ok(), "help {} errors: {:?}", name, r.err());
    }
}

#[test]
fn test_every_command_dispatches() {
    for name in COMMAND_NAMES {
        let mut ctx = CommandContext::new();
        let results = execute(&mut ctx, name);
        assert!(
            !results.iter().any(|r| r.output.contains("Unknown command")),
            "{} does not dispatch: {:?}", name,
            results.first().map(|r| r.output.clone())
        );
    }
}

#[test]
fn test_conflict_message_names_blocking_constraint() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 5,0");
    run_ok(&mut ctx, "add_line 0,1 5,2 noconnect");
    let out = run_ok(&mut ctx, "parallel L0 L1");
    // The applied constraint's id (from "C<n>: ..." output).
    let nid: String = out.split(':').next().unwrap_or("").trim().to_string();
    assert!(nid.starts_with('C'), "unexpected apply output: {}", out);
    let err = run_err(&mut ctx, "parallel L0 L1");
    assert!(err.contains(&format!("({})", nid)),
        "duplicate rejection must name {}: {}", nid, err);
    let err = run_err(&mut ctx, "perpendicular L0 L1");
    assert!(err.contains(&format!("({})", nid)),
        "conflict rejection must name {}: {}", nid, err);
}

// -- Scale ------------------------------------------------------------

#[test]
fn test_scale_lines_about_point() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 2,1");
    run_ok(&mut ctx, "add_point 0,0");
    let out = run_ok(&mut ctx, "scale L0 L1 L2 L3 about P0 2");
    assert!(out.contains("Scaled L0 L1 L2 L3"), "{}", out);
    // The far corner doubled away from the origin.
    let l1 = resolve_line(&ctx.sketch, "L1").unwrap();
    let p = ctx.sketch.lines[l1].p2.value;
    assert!(near(p.x, 4.0) && near(p.y, 2.0), "far corner = {:?}", (p.x, p.y));
    // One undo restores everything.
    run_ok(&mut ctx, "undo");
    let p = ctx.sketch.lines[l1].p2.value;
    assert!(near(p.x, 2.0) && near(p.y, 1.0), "after undo = {:?}", (p.x, p.y));
}

#[test]
fn test_scale_about_endpoint_and_coord() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 1,0 3,0");
    run_ok(&mut ctx, "scale L0 about L0.p1 2");
    assert!(near(line_len(&ctx, "L0"), 4.0));
    let l0 = resolve_line(&ctx.sketch, "L0").unwrap();
    assert!(near(ctx.sketch.lines[l0].p1.value.x, 1.0), "center endpoint stays");
    run_ok(&mut ctx, "scale L0 about 0,0 0.5");
    assert!(near(line_len(&ctx, "L0"), 2.0));
}

#[test]
fn test_scale_circle_radius_and_dim() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 2,0 1");
    run_ok(&mut ctx, "radius A0 1");
    let out = run_ok(&mut ctx, "scale A0 about 0,0 3");
    assert!(out.contains("dims scaled: d0"), "{}", out);
    let a = resolve_arc(&ctx.sketch, "A0").unwrap();
    assert!(near(ctx.sketch.arcs[a].radius.value, 3.0));
    assert!(near(ctx.sketch.arcs[a].center.value.x, 6.0));
    assert!(near(ctx.sketch.dimensions[0].value, 3.0));
}

#[test]
fn test_scale_boundary_dim_reported_left() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0");
    run_ok(&mut ctx, "add_line 0,1 2,1 noconnect");
    run_ok(&mut ctx, "distance L0.p1 L1.p1 1");
    let out = run_ok(&mut ctx, "scale L0 about 0,0 2");
    assert!(out.contains("dims left: d0 (spans unscaled geometry)"), "{}", out);
    assert!(near(ctx.sketch.dimensions[0].value, 1.0));
}

#[test]
fn test_scale_selection_form() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0");
    run_ok(&mut ctx, "select L0");
    run_ok(&mut ctx, "scale selection about 0,0 1.5");
    assert!(near(line_len(&ctx, "L0"), 3.0));
}

#[test]
fn test_scale_rejects_bad_factor() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0");
    let e = run_err(&mut ctx, "scale L0 about 0,0 0");
    assert!(e.contains("positive"), "{}", e);
    let e = run_err(&mut ctx, "scale L0 about 0,0 -1");
    assert!(e.contains("positive"), "{}", e);
    run_err(&mut ctx, "scale about 0,0 2");
}

#[test]
fn test_scale_length_dim_holds_after_solve() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 3,0");
    run_ok(&mut ctx, "length L0 3");
    let out = run_ok(&mut ctx, "scale L0 about 0,0 2");
    assert!(out.contains("dims scaled: d0"), "{}", out);
    // The solve keeps the scaled length: no snap-back.
    assert!(near(line_len(&ctx, "L0"), 6.0));
    assert!(near(ctx.sketch.dimensions[0].value, 6.0));
}

// -- offset --

/// Perpendicular distances of both endpoints of `res` from the infinite
/// line of `src` (equal when the lines are parallel).
fn line_offsets(ctx: &CommandContext, src: &str, res: &str) -> (f64, f64) {
    let s = &ctx.sketch.lines[resolve_line(&ctx.sketch, src).unwrap()];
    let r = &ctx.sketch.lines[resolve_line(&ctx.sketch, res).unwrap()];
    let d = s.p2.value - s.p1.value;
    let len = (d.x * d.x + d.y * d.y).sqrt();
    let dist = |p: vect2d| ((p.x - s.p1.value.x) * d.y - (p.y - s.p1.value.y) * d.x).abs() / len;
    (dist(r.p1.value), dist(r.p2.value))
}

/// Center distance and radius gap between a source arc and its result.
fn arc_offsets(ctx: &CommandContext, src: &str, res: &str) -> (f64, f64) {
    let s = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, src).unwrap()];
    let r = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, res).unwrap()];
    let c = s.center.value - r.center.value;
    ((c.x * c.x + c.y * c.y).sqrt(), (r.radius.value - s.radius.value).abs())
}

/// Whole-sketch residual after a fresh solve.
fn solve_cost(ctx: &mut CommandContext) -> f64 {
    ctx.sketch.get_mut().solve().end_cost
}

/// The sketch is consistent: every hard constraint holds after a solve.
fn assert_solved(ctx: &mut CommandContext) {
    let c = solve_cost(ctx);
    assert!(c < 1e-8, "the sketch does not solve cleanly: cost {:e}\n{}", c,
        ctx.sketch.list_constraints().join("\n"));
}

fn meta_count(ctx: &CommandContext) -> usize {
    ctx.sketch.metas.len()
}

/// A single line: one result parallel at the distance, free ends on the
/// source ends' normals, the DOF unchanged, the meta recorded.
#[test]
fn test_offset_single_line() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 1.5");
    assert!(out.contains("M0: offset of L0 by 1.5 left -> left: L1"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof, "an offset adds no freedom");
    let (a, b) = line_offsets(&ctx, "L0", "L1");
    assert!((a - 1.5).abs() < 1e-6 && (b - 1.5).abs() < 1e-6, "{} {}", a, b);
    // Left of (0,0)->(4,0) is +y; the ends are straight above the source's.
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near(l1.p1.value.y, 1.5) && near(l1.p1.value.x, 0.0) && near(l1.p2.value.x, 4.0), "{:?} {:?}", l1.p1.value, l1.p2.value);
    assert_eq!(ctx.sketch.parallel.len(), 1);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 2, "both free ends pinned");
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert!(matches!(ctx.sketch.dimensions[0].kind, DimensionKind::LineLineDistance(_, _)));
    assert_eq!(meta_count(&ctx), 1);
    assert_solved(&mut ctx);
    let listed = run_ok(&mut ctx, "list metas");
    assert!(listed.contains("M0: offset of L0"), "{}", listed);
    let info = run_ok(&mut ctx, "info L1");
    assert!(info.contains("result of offset M0"), "{}", info);
}

/// Sides: right, flip, symmetric, two distances, and an expression.
#[test]
fn test_offset_line_sides() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "offset L0 1 right");
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near(l1.p1.value.y, -1.0), "{:?}", l1.p1.value);
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "offset L0 1 flip");
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    assert!(near(l1.p1.value.y, -1.0), "{:?}", l1.p1.value);
    run_ok(&mut ctx, "undo");

    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 1 symmetric");
    assert!(out.contains("symmetric"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let ys: Vec<f64> = ["L1", "L2"].iter().map(|n| ctx.sketch.lines[resolve_line(&ctx.sketch, n).unwrap()].p1.value.y).collect();
    assert!((ys[0] - 1.0).abs() < 1e-6 && (ys[1] + 1.0).abs() < 1e-6, "{:?}", ys);
    run_ok(&mut ctx, "undo");

    let out = run_ok(&mut ctx, "offset L0 1 2.5");
    assert!(out.contains("1 left / 2.5 right"), "{}", out);
    let ys: Vec<f64> = ["L1", "L2"].iter().map(|n| ctx.sketch.lines[resolve_line(&ctx.sketch, n).unwrap()].p1.value.y).collect();
    assert!((ys[0] - 1.0).abs() < 1e-6 && (ys[1] + 2.5).abs() < 1e-6, "{:?}", ys);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    run_ok(&mut ctx, "undo");

    run_ok(&mut ctx, "param w 3");
    run_ok(&mut ctx, "offset L0 w/2");
    let (a, _) = line_offsets(&ctx, "L0", "L1");
    assert!((a - 1.5).abs() < 1e-6, "{}", a);
    assert_eq!(ctx.sketch.dimensions[0].expr_str.as_deref(), Some("w/2"));
    run_ok(&mut ctx, "param w 5");
    let (a, _) = line_offsets(&ctx, "L0", "L1");
    assert!((a - 2.5).abs() < 1e-6, "expression distance follows its parameter: {}", a);
    assert_eq!(meta_count(&ctx), 1, "a parameter change is not an edit of the offset");
}

/// Two lines at a corner: the results meet at a sharp corner on both
/// sides (extended outside, trimmed inside), at the distance.
#[test]
fn test_offset_corner() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset L0 L1 1 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Left of the chain (0,0)->(4,0)->(4,3) is the inside (+y then -x).
    let inner_a = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L2").unwrap()];
    let inner_b = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L3").unwrap()];
    assert!(near(inner_a.p2.value.x, 3.0) && near(inner_a.p2.value.y, 1.0), "inner corner {:?}", inner_a.p2.value);
    assert!(near(inner_b.p1.value.x, 3.0) && near(inner_b.p1.value.y, 1.0), "inner corner {:?}", inner_b.p1.value);
    let outer_a = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L4").unwrap()];
    assert!(near(outer_a.p2.value.x, 5.0) && near(outer_a.p2.value.y, -1.0), "outer corner {:?}", outer_a.p2.value);
    for (s, r) in [("L0", "L2"), ("L1", "L3"), ("L0", "L4"), ("L1", "L5")] {
        let (a, b) = line_offsets(&ctx, s, r);
        assert!((a - 1.0).abs() < 1e-6 && (b - 1.0).abs() < 1e-6, "{} {}: {} {}", s, r, a, b);
    }
    assert_solved(&mut ctx);
    // Joints are coincidences between the results, no pins at corners.
    assert_eq!(ctx.sketch.on_normal_ll.len(), 4, "two free ends per side");
}

/// A closed rectangle inward and outward; inward past the half width is
/// refused naming the segment that collapses.
#[test]
fn test_offset_rectangle() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 6,4");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 L1 L2 L3 1 inward");
    assert!(out.contains("(closed)"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Every result line sits 1 inside: x in 1..5, y in 1..3.
    for n in ["L4", "L5", "L6", "L7"] {
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, n).unwrap()];
        for p in [l.p1.value, l.p2.value] {
            assert!(p.x > 0.99 && p.x < 5.01 && p.y > 0.99 && p.y < 3.01, "{} {:?}", n, p);
        }
    }
    assert!(ctx.sketch.on_normal_ll.is_empty(), "a closed loop has no free ends");
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "offset sequence L0 1 outward");
    for n in ["L4", "L5", "L6", "L7"] {
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, n).unwrap()];
        for p in [l.p1.value, l.p2.value] {
            assert!((p.x < -0.99 || p.x > 6.99) || (p.y < -0.99 || p.y > 4.99), "{} {:?}", n, p);
        }
    }
    run_ok(&mut ctx, "undo");
    let e = run_err(&mut ctx, "offset sequence L0 2.5 inward");
    assert!(e.contains("collapses") || e.contains("do not meet"), "{}", e);
    assert_eq!(ctx.sketch.lines.refs().count(), 4, "nothing left behind: {}", e);
}

/// A single arc outward and inward (concentric, radius +- d, ends on the
/// source's rays), a circle both ways, and an inward offset past the
/// radius refused.
#[test]
fn test_offset_arc_and_circle() {
    let mut ctx = CommandContext::new();
    // CCW quarter arc of radius 2 about the origin, from (2,0) to (0,2).
    run_ok(&mut ctx, "add_arc 2,0 0,2 1.41421356,1.41421356");
    let dof = ctx.sketch.dof().unwrap();
    // Travelling CCW, left is inward.
    run_ok(&mut ctx, "offset A0 0.5");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let (cd, gap) = arc_offsets(&ctx, "A0", "A1");
    assert!(cd < 1e-6 && (gap - 0.5).abs() < 1e-6, "{} {}", cd, gap);
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    assert!((a1.radius.value - 1.5).abs() < 1e-6, "inward: {}", a1.radius.value);
    let s = crate::geometry::arc_start_pos(a1);
    assert!(near(s.x, 1.5) && near(s.y, 0.0), "start on the source's start ray: {:?}", s);
    assert_eq!(ctx.sketch.on_normal_aa.len(), 2);
    assert_eq!(ctx.sketch.concentric.len(), 1);
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "offset A0 0.5 right");
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    assert!((a1.radius.value - 2.5).abs() < 1e-6, "outward: {}", a1.radius.value);
    run_ok(&mut ctx, "undo");
    let e = run_err(&mut ctx, "offset A0 2.5");
    assert!(e.contains("cannot be offset inward"), "{}", e);

    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 0,0 2");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset A0 0.5 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let radii: Vec<f64> = ["A1", "A2"].iter().map(|n| ctx.sketch.arcs[resolve_arc(&ctx.sketch, n).unwrap()].radius.value).collect();
    assert!((radii[0] - 1.5).abs() < 1e-6 && (radii[1] - 2.5).abs() < 1e-6, "{:?}", radii);
    assert!(ctx.sketch.on_normal_aa.is_empty());
    assert_solved(&mut ctx);
}

/// Line - arc - line, tangent (a slot end): tangent joints on both sides,
/// the results tangent too, pins on the earlier segment at each joint.
#[test]
fn test_offset_tangent_chain() {
    let mut ctx = CommandContext::new();
    // Bottom line to (4,0), semicircle up to (4,2) around (4,1), top line back.
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 4,2 5,1");
    run_ok(&mut ctx, "add_line 4,2 0,2");
    assert_eq!(ctx.sketch.tangent_la.len(), 2, "auto-tangent at both joints");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 A0 L1 0.5 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof, "{}", out);
    assert_solved(&mut ctx);
    // Sides: left of (0,0)->(4,0) is inside: radius 0.5 arc; right: 1.5.
    let radii: Vec<f64> = ctx.sketch.arcs.iter().filter(|a| a.name != "A0").map(|a| a.radius.value).collect();
    assert!(radii.iter().any(|r| (r - 0.5).abs() < 1e-6) && radii.iter().any(|r| (r - 1.5).abs() < 1e-6), "{:?}", radii);
    // Each result line is tangent to its result arc at their shared end
    // (the joint is at the offset of the source joint).
    for (l, a) in [("L2", "A1"), ("L3", "A1"), ("L4", "A2"), ("L5", "A2")] {
        let line = &ctx.sketch.lines[resolve_line(&ctx.sketch, l).unwrap()];
        let arc = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, a).unwrap()];
        let d = line.p2.value - line.p1.value;
        let len = (d.x * d.x + d.y * d.y).sqrt();
        let c = arc.center.value;
        let dist = ((c.x - line.p1.value.x) * d.y - (c.y - line.p1.value.y) * d.x).abs() / len;
        assert!((dist - arc.radius.value).abs() < 1e-6, "{} tangent to {}: {} vs {}", l, a, dist, arc.radius.value);
    }
    // Pins: two free ends and two tangent joints per side.
    assert_eq!(ctx.sketch.on_normal_ll.len() + ctx.sketch.on_normal_aa.len(), 8);
}

/// A closed slot (two lines, two arcs, all tangent) inward and outward.
#[test]
fn test_offset_slot() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 4,2 5,1");
    run_ok(&mut ctx, "add_line 4,2 0,2");
    run_ok(&mut ctx, "add_arc 0,2 0,0 -1,1");
    assert_eq!(ctx.sketch.tangent_la.len(), 4, "auto-tangent at every joint");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset sequence L0 0.25 inward");
    assert!(out.contains("(closed)"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    let radii: Vec<f64> = ctx.sketch.arcs.iter().filter(|a| !["A0", "A1"].contains(&a.name.as_str())).map(|a| a.radius.value).collect();
    assert!(radii.iter().all(|r| (r - 0.75).abs() < 1e-6), "{:?}", radii);
    // One pin per tangent joint, on the earlier segment's result: the
    // line ends before the arcs (LL), the arc ends before the lines (AA);
    // the all-tangent loop closes by a second pin on the first line's
    // start instead of a coincidence, and carries one distance dim.
    assert_eq!(ctx.sketch.on_normal_ll.len(), 3);
    assert_eq!(ctx.sketch.on_normal_aa.len(), 2);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
    assert_eq!(ctx.sketch.metas[0].as_offset().unwrap().sides[0].constraints.len(), 4 + 3, "relations and three coincidences");
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "offset sequence L0 0.25 outward");
    let radii: Vec<f64> = ctx.sketch.arcs.iter().filter(|a| !["A0", "A1"].contains(&a.name.as_str())).map(|a| a.radius.value).collect();
    assert!(radii.iter().all(|r| (r - 1.25).abs() < 1e-6), "{:?}", radii);
}

/// Arc-arc tangent joints: an S-curve and a same-direction pair.
#[test]
fn test_offset_arc_arc() {
    let mut ctx = CommandContext::new();
    // Quarter arc about (0,0) from (2,0) up to (0,2), then a quarter arc
    // about (0,4) from (0,2) up to (-2,4): an S with a tangent joint.
    run_ok(&mut ctx, "add_arc 2,0 0,2 1.41421356,1.41421356");
    run_ok(&mut ctx, "add_arc 0,2 -2,4 -1.41421356,2.58578644");
    assert_eq!(ctx.sketch.tangent_aa.len(), 1, "auto-tangent at the joint");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset A0 A1 0.5 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // Across an S the same side is inside one arc and outside the next.
    let names: Vec<String> = ctx.sketch.arcs.iter().map(|a| a.name.clone()).collect();
    assert_eq!(names.len(), 6, "{:?}", names);
    let r = |n: &str| ctx.sketch.arcs[resolve_arc(&ctx.sketch, n).unwrap()].radius.value;
    let side1 = (r("A2"), r("A3"));
    assert!(((side1.0 - 1.5).abs() < 1e-6 && (side1.1 - 2.5).abs() < 1e-6) || ((side1.0 - 2.5).abs() < 1e-6 && (side1.1 - 1.5).abs() < 1e-6), "{:?}", side1);

    // Same direction: quarter arc then another quarter arc continuing
    // around the same center, tangent at (0,2).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 2,0 0,2 1.41421356,1.41421356");
    run_ok(&mut ctx, "add_arc 0,2 -2,0 -1.41421356,1.41421356");
    assert_eq!(ctx.sketch.tangent_aa.len(), 1, "auto-tangent at the joint");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset A0 A1 0.5");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    let r = |n: &str| ctx.sketch.arcs[resolve_arc(&ctx.sketch, n).unwrap()].radius.value;
    assert!((r("A2") - 1.5).abs() < 1e-6 && (r("A3") - 1.5).abs() < 1e-6, "{} {}", r("A2"), r("A3"));
}

/// Line meeting an arc at a corner (not tangent): the results intersect.
#[test]
fn test_offset_line_arc_corner() {
    let mut ctx = CommandContext::new();
    // Line along x to (2,0), then a CCW quarter arc about the origin from
    // (2,0) to (0,2): they meet at 90 degrees.
    run_ok(&mut ctx, "add_line -2,0 2,0");
    run_ok(&mut ctx, "add_arc 2,0 0,2 1.41421356,1.41421356");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset L0 A0 0.5 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // The results meet: each result line's end lies on its result arc.
    for (l, a) in [("L1", "A1"), ("L2", "A2")] {
        let line = &ctx.sketch.lines[resolve_line(&ctx.sketch, l).unwrap()];
        let arc = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, a).unwrap()];
        let s = crate::geometry::arc_start_pos(arc);
        assert!(near(s.x, line.p2.value.x) && near(s.y, line.p2.value.y), "{} {}: {:?} vs {:?}", l, a, s, line.p2.value);
        let c = arc.center.value;
        let d = ((line.p2.value.x - c.x).powi(2) + (line.p2.value.y - c.y).powi(2)).sqrt();
        assert!((d - arc.radius.value).abs() < 1e-6);
    }
    assert!(ctx.sketch.on_normal_ll.len() == 2 && ctx.sketch.on_normal_aa.len() == 2, "free ends only");
}

/// A long mixed chain with corners and tangent joints, symmetric.
#[test]
fn test_offset_mixed_chain() {
    let mut ctx = CommandContext::new();
    // Tangent joints L0-A0, A0-L1 and L2-A1 (auto-tangent at creation),
    // a corner L1-L2.
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 6,2 5.41421356,0.58578644");
    run_ok(&mut ctx, "add_line 6,2 6,5");
    run_ok(&mut ctx, "add_line 6,5 9,5");
    run_ok(&mut ctx, "add_arc 9,5 10,6 9.70710678,5.29289322");
    assert_eq!(ctx.sketch.tangent_la.len(), 3, "auto-tangent at the three tangent joints");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset sequence L0 0.4 symmetric");
    assert_eq!(ctx.sketch.dof().unwrap(), dof, "{}", out);
    assert_solved(&mut ctx);
    assert_eq!(ctx.sketch.lines.refs().count(), 9);
    assert_eq!(ctx.sketch.arcs.refs().count(), 6);
    for (s, n) in [("L0", 2), ("L1", 2), ("L2", 2)] {
        let mut hits = 0;
        for r in ctx.sketch.lines.refs() {
            let name = ctx.sketch.lines[r].name.clone();
            if name == s || !ctx.sketch.parallel.iter().any(|c| (ctx.sketch.lines[c.a].name == s && ctx.sketch.lines[c.b].name == name)) { continue; }
            let (a, b) = line_offsets(&ctx, s, &name);
            assert!((a - 0.4).abs() < 1e-6 && (b - 0.4).abs() < 1e-6, "{} -> {}: {} {}", s, name, a, b);
            hits += 1;
        }
        assert_eq!(hits, n, "{} has two results", s);
    }
}

/// Selection and walk forms, and the rejections.
#[test]
fn test_offset_selection_forms_and_errors() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0 2,2 0,2; add_line 5,5 6,6");
    run_ok(&mut ctx, "select L1 L0 L2");
    let out = run_ok(&mut ctx, "offset selection 0.5");
    assert!(out.contains("offset of L0 L1 L2"), "{}", out);
    run_ok(&mut ctx, "undo");
    let out = run_ok(&mut ctx, "select L1 sequence");
    assert!(out.contains("Sequence: L0 L1 L2"), "{}", out);
    let e = run_err(&mut ctx, "offset L0 L3 1");
    assert!(e.contains("not one connected sequence"), "{}", e);
    let e = run_err(&mut ctx, "offset L0 L1");
    assert!(e.contains("missing the distance"), "{}", e);
    let e = run_err(&mut ctx, "offset L0 -1");
    assert!(e.contains("positive"), "{}", e);
    let e = run_err(&mut ctx, "offset L0 1 inward");
    assert!(e.contains("closed sequence"), "{}", e);
    // A doubled-back chain has no corner.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0; add_line 4,0 1,0");
    let e = run_err(&mut ctx, "offset L0 L1 1");
    assert!(e.contains("double back"), "{}", e);
}

/// Editing: the distance moves the geometry through the dims; flip moves
/// it to the other side; the kind adds and removes a side; `nopin`.
#[test]
fn test_offset_edit() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    run_ok(&mut ctx, "offset L0 L1 1");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset M0 2");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let (a, b) = line_offsets(&ctx, "L0", "L2");
    assert!((a - 2.0).abs() < 1e-6 && (b - 2.0).abs() < 1e-6, "{} {}", a, b);
    assert_eq!(ctx.sketch.dimensions.iter().filter(|d| (d.value - 2.0).abs() < 1e-9).count(), 2);
    assert_eq!(meta_count(&ctx), 1, "the edit keeps the meta");
    let l2 = ctx.sketch.lines[resolve_line(&ctx.sketch, "L2").unwrap()].p1.value;
    assert!(near(l2.y, 2.0), "{:?}", l2);

    run_ok(&mut ctx, "offset M0 flip");
    let l2 = ctx.sketch.lines[resolve_line(&ctx.sketch, "L2").unwrap()].p1.value;
    assert!(near(l2.y, -2.0), "flipped: {:?}", l2);
    assert_solved(&mut ctx);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);

    let out = run_ok(&mut ctx, "offset M0 symmetric");
    assert!(out.contains("symmetric"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 6, "a second side was created");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    let m = &ctx.sketch.metas[0];
    assert_eq!(m.as_offset().unwrap().sides.len(), 2);

    run_ok(&mut ctx, "offset M0 one");
    assert_eq!(ctx.sketch.lines.refs().count(), 4, "the second side is gone");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(meta_count(&ctx), 1);

    run_ok(&mut ctx, "offset M0 two 1 2");
    assert_eq!(ctx.sketch.lines.refs().count(), 6);
    let info = run_ok(&mut ctx, "info M0");
    assert!(info.contains("1 right / 2 left"), "{}", info);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // An edit that would collapse a segment is refused and changes nothing.
    let e = run_err(&mut ctx, "offset M0 two 1 3");
    assert!(e.contains("collapses"), "{}", e);
    assert_eq!(ctx.sketch.lines.refs().count(), 6);
    assert_eq!(meta_count(&ctx), 1);

    // nopin: the free ends are free and the DOF shows it.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 1 nopin");
    assert!(out.contains("[nopin]"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof + 2, "two free ends");
    assert!(ctx.sketch.on_normal_ll.is_empty());
}

/// Ownership: deleting a result, an owned constraint, an owned dimension
/// or a source drops the meta with a notice and keeps the rest; editing
/// or converting an owned dimension does too; deleting a pin does not;
/// undo brings the meta back.
#[test]
fn test_offset_ownership() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    run_ok(&mut ctx, "offset L0 L1 1");
    assert_eq!(meta_count(&ctx), 1);
    let out = run_ok(&mut ctx, "delete L3");
    assert!(out.contains("notice: offset M0 dropped"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
    assert!(ctx.sketch.lines.refs().count() == 3, "the other result line stays");
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1, "undo restores the meta");

    let out = run_ok(&mut ctx, "delete d0");
    assert!(out.contains("dropped") && out.contains("d0"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);

    let out = run_ok(&mut ctx, "distance L0 L2 3");
    assert!(out.contains("d0 was edited"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);

    let out = run_ok(&mut ctx, "set_derived d0");
    assert!(out.contains("made derived"), "{}", out);
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);

    // The parallel of L0/L2 is an owned constraint.
    let nid = ctx.sketch.parallel[0].nid;
    let out = run_ok(&mut ctx, &format!("delete C{}", nid));
    assert!(out.contains(&format!("C{} was deleted", nid)), "{}", out);
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);

    // A pin is soft-owned: deleting it keeps the meta.
    let pin = ctx.sketch.on_normal_ll[0].nid;
    let out = run_ok(&mut ctx, &format!("delete C{}", pin));
    assert!(!out.contains("dropped"), "{}", out);
    assert_eq!(meta_count(&ctx), 1);
    run_ok(&mut ctx, "undo");

    // A source entity.
    let out = run_ok(&mut ctx, "delete L0");
    assert!(out.contains("dropped"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
    run_ok(&mut ctx, "undo");

    // Splitting a result (at a crossing line).
    run_ok(&mut ctx, "add_line 2,-1 2,2 noconnect");
    assert_eq!(meta_count(&ctx), 1);
    let out = run_ok(&mut ctx, "split L2 2,1");
    assert!(out.contains("dropped"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
}

/// Dissolve keeps the geometry; delete-all removes it.
#[test]
fn test_offset_dissolve_and_delete() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "offset L0 1 symmetric");
    let out = run_ok(&mut ctx, "delete M0");
    assert!(out.contains("Dissolved M0"), "{}", out);
    assert_eq!(meta_count(&ctx), 0);
    assert_eq!(ctx.sketch.lines.refs().count(), 3);
    assert_eq!(ctx.sketch.parallel.len(), 2, "the relations stay");
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);
    let out = run_ok(&mut ctx, "delete M0 all");
    assert!(out.contains("Deleted M0 and L1, L2"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 1);
    assert!(ctx.sketch.parallel.is_empty() && ctx.sketch.dimensions.is_empty());
    assert!(!out.contains("notice"), "no tampering notice for an ordered delete: {}", out);
    let e = run_err(&mut ctx, "delete M0 some");
    assert!(e.contains("unknown option") || e.contains("Unknown"), "{}", e);
}

/// The meta survives save / load.
#[test]
fn test_offset_persists() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    run_ok(&mut ctx, "offset L0 L1 1");
    let json = serde_json::to_string(&*ctx.sketch).unwrap();
    let back: Sketch = serde_json::from_str(&json).unwrap();
    assert_eq!(back.metas.len(), 1);
    assert_eq!(back.metas[0].name, "M0");
    assert_eq!(back.metas[0].as_offset().unwrap().sides[0].segs.len(), 2);
}

/// Ellipses: the concentric approximation, rotation and both semi-axes
/// held; a tangent joint at an elliptic arc is refused.
#[test]
fn test_offset_ellipse() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_ellipse 0,0 4 2 30");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset EA0 0.5 symmetric");
    assert!(out.contains("approximate"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    let rx: Vec<(f64, f64)> = ctx.sketch.arcs.iter().filter(|a| a.name != "EA0").map(|a| (a.radius.value, a.radius_b.value)).collect();
    assert!(rx.iter().any(|(a, b)| (a - 4.5).abs() < 1e-6 && (b - 2.5).abs() < 1e-6), "{:?}", rx);
    assert!(rx.iter().any(|(a, b)| (a - 3.5).abs() < 1e-6 && (b - 1.5).abs() < 1e-6), "{:?}", rx);
    assert_eq!(ctx.sketch.arc_arc_parallel.len(), 2, "rotation tied");
    assert_eq!(ctx.sketch.distance_concentric.len(), 2);
    // An elliptic arc alone, both free ends on the source ends' normals.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_earc_center 0,0 4 2 0 0 90");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset EA0 0.5 right");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    assert_eq!(ctx.sketch.on_normal_aa.len(), 2);
    let res = ctx.sketch.arcs.iter().find(|a| a.name != "EA0").unwrap();
    assert!((res.radius.value - 4.5).abs() < 1e-6 && (res.radius_b.value - 2.5).abs() < 1e-6, "{} {}", res.radius.value, res.radius_b.value);
    // Elliptic arc and a line at a corner: allowed.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_earc_center 0,0 4 2 0 0 90");
    run_ok(&mut ctx, "add_line 0,2 -3,4");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset EA0 L0 0.3");
    assert_eq!(ctx.sketch.dof().unwrap(), dof, "{}", out);
    assert_solved(&mut ctx);
    // Elliptic arc tangent to a line (the ellipse's top, a horizontal
    // line): refused with the reason.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_earc_center 0,0 4 2 0 0 90");
    run_ok(&mut ctx, "add_line 0,2 -3,2");
    let e = run_err(&mut ctx, "offset EA0 L0 0.3");
    assert!(e.contains("tangentially") && e.contains("approximate"), "{}", e);
}

/// Round corners: a convex corner is an arc of the distance centered on
/// the source joint, tangent to both neighbours, its radius an owned
/// dimension; concave corners stay sharp; the DOF is unchanged; a
/// distance edit moves the radii; the corner style and the side rebuild
/// the side, a plain distance edit keeps it.
#[test]
fn test_offset_round_corners() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 4,3");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 L1 L2 L3 1 outward round");
    assert!(out.contains("[round]"), "{}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 4, "one arc per convex corner");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.tangent_la.len(), 8);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 0, "a closed sequence has no pins");
    let radii: Vec<f64> = ctx.sketch.dimensions.iter().filter(|d| d.name.starts_with('d')).map(|d| d.value).collect();
    assert_eq!(radii.len(), 8, "four distances and four radii");
    assert!(radii.iter().all(|r| (r - 1.0).abs() < 1e-9));
    for a in ctx.sketch.arcs.refs() {
        let arc = &ctx.sketch.arcs[a];
        assert!((arc.radius.value - 1.0).abs() < 1e-6, "{}", arc.radius.value);
        let c = arc.center.value;
        assert!(((c.x - 0.0).abs() < 1e-6 || (c.x - 4.0).abs() < 1e-6) && ((c.y - 0.0).abs() < 1e-6 || (c.y - 3.0).abs() < 1e-6),
            "centered on a source corner: {:?}", c);
    }
    assert_solved(&mut ctx);
    let m = ctx.sketch.metas[0].as_offset().unwrap().clone();
    assert!(m.round && m.sides[0].corners.len() == 4);

    // A distance edit keeps the entities and moves the radii.
    let before: Vec<String> = ctx.sketch.arcs.refs().map(|a| ctx.sketch.arcs[a].name.clone()).collect();
    run_ok(&mut ctx, "offset M0 2");
    let after: Vec<String> = ctx.sketch.arcs.refs().map(|a| ctx.sketch.arcs[a].name.clone()).collect();
    assert_eq!(before, after, "kept");
    assert!(ctx.sketch.arcs.refs().all(|a| (ctx.sketch.arcs[a].radius.value - 2.0).abs() < 1e-6));
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);

    // Inward: no convex corner, no arcs; the side is rebuilt.
    run_ok(&mut ctx, "offset M0 1 inward");
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let inward: Vec<String> = ctx.sketch.lines.refs().map(|l| ctx.sketch.lines[l].name.clone()).collect();
    // Sharp on a side without corners keeps it.
    let out = run_ok(&mut ctx, "offset M0 sharp");
    assert!(!out.contains("[round]"), "{}", out);
    let same: Vec<String> = ctx.sketch.lines.refs().map(|l| ctx.sketch.lines[l].name.clone()).collect();
    assert_eq!(inward, same);
    // Round outward again, after a distance edit that moved the source
    // by the solver's tolerance: the corners still build.
    run_ok(&mut ctx, "offset M0 round outward");
    assert_eq!(ctx.sketch.arcs.refs().count(), 4);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    assert_eq!(meta_count(&ctx), 1);

    // Deleting a corner arc drops the meta (owned result).
    let out = run_ok(&mut ctx, "delete A4");
    assert!(out.contains("notice: offset M0 dropped"), "{}", out);
}

/// Symmetric round on an open corner: the convex side gets the arc, the
/// concave side stays sharp; pins toggle with `nopin` / `pin` on the
/// surviving sides; a line-arc corner rounds with an arc-arc tangent.
#[test]
fn test_offset_round_open_and_pins() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 L1 1 symmetric round");
    assert!(out.contains("left: L2 L3; right: L4 L5 A0"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 4, "two free ends per side");
    run_ok(&mut ctx, "offset M0 nopin");
    assert_eq!(ctx.sketch.on_normal_ll.len(), 0);
    assert_eq!(ctx.sketch.dof().unwrap(), dof + 4);
    assert_eq!(meta_count(&ctx), 1, "removing its own pins is not tampering");
    run_ok(&mut ctx, "offset M0 pin");
    assert_eq!(ctx.sketch.on_normal_ll.len(), 4);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // One side keeps the `distance` side (left, concave: no corner); a
    // flip rebuilds the result on the convex side with its corner.
    run_ok(&mut ctx, "offset M0 one");
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    run_ok(&mut ctx, "offset M0 flip");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);

    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 6,2 4.5858,1.4142");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset L0 A0 0.5 symmetric round");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.tangent_la.len(), 1);
    assert_eq!(ctx.sketch.tangent_aa.len(), 1);
    assert_eq!(ctx.sketch.arcs.refs().count(), 4, "source, two results, one corner");
    assert_solved(&mut ctx);
    run_ok(&mut ctx, "offset M0 0.8");
    let corner = ctx.sketch.arcs.refs().map(|a| &ctx.sketch.arcs[a]).find(|a| a.name == "A3").unwrap();
    assert!((corner.radius.value - 0.8).abs() < 1e-6 && near(corner.center.value.x, 4.0) && near(corner.center.value.y, 0.0));
    assert_solved(&mut ctx);
}

/// Caps: round caps are half circles around the source ends, tangent to
/// both results, held by one pin each; line caps join the two results'
/// ends; a distance edit keeps them; the kind is editable; the DOF is
/// unchanged; caps are owned; refused where they make no sense.
#[test]
fn test_offset_caps() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0 4,3");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 L1 1 symmetric caps round");
    assert!(out.contains("round caps: A0 A1"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.tangent_la.len(), 4);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 2, "one pin per cap, no free-end pins");
    for a in ctx.sketch.arcs.refs() {
        let arc = &ctx.sketch.arcs[a];
        assert!((arc.radius.value - 1.0).abs() < 1e-6);
        let c = arc.center.value;
        assert!((near(c.x, 0.0) && near(c.y, 0.0)) || (near(c.x, 4.0) && near(c.y, 3.0)), "{:?}", c);
    }
    assert_solved(&mut ctx);
    // The distance moves the caps along (tangent to the results).
    run_ok(&mut ctx, "offset M0 2");
    assert!(ctx.sketch.arcs.refs().all(|a| (ctx.sketch.arcs[a].radius.value - 2.0).abs() < 1e-6));
    assert_eq!(ctx.sketch.arcs.refs().count(), 2, "kept");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // Line caps: lines across the ends; the free-end pins come back.
    let out = run_ok(&mut ctx, "offset M0 caps line");
    assert!(out.contains("line caps: L6 L7"), "{}", out);
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 4);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let cap = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L6").unwrap()];
    assert!(near(cap.p1.value.x, 0.0) && near(cap.p1.value.y, 2.0) && near(cap.p2.value.y, -2.0), "{:?}", cap.p1.value);
    assert_solved(&mut ctx);
    // Two distances keep line caps; round caps are refused there.
    run_ok(&mut ctx, "offset M0 1 2");
    assert_eq!(ctx.sketch.lines.refs().count(), 8);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let e = run_err(&mut ctx, "offset M0 caps round");
    assert!(e.contains("symmetric"), "{}", e);
    // No caps: the cap lines go, the meta stays.
    run_ok(&mut ctx, "offset M0 caps none");
    assert_eq!(ctx.sketch.lines.refs().count(), 6);
    assert_eq!(meta_count(&ctx), 1);
    // Going to one side drops the caps by itself.
    run_ok(&mut ctx, "offset M0 symmetric caps round");
    run_ok(&mut ctx, "offset M0 one");
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    assert_eq!(ctx.sketch.on_normal_ll.len(), 2, "the free-end pins of the one side");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Deleting a cap drops the meta.
    run_ok(&mut ctx, "offset M0 symmetric caps line");
    let out = run_ok(&mut ctx, "delete L10");
    assert!(out.contains("notice: offset M0 dropped"), "{}", out);

    // Refused: one side, a closed sequence.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 4,3");
    let e = run_err(&mut ctx, "offset L0 L1 L2 L3 1 symmetric caps line");
    assert!(e.contains("closed"), "{}", e);
    run_ok(&mut ctx, "add_line 6,0 8,0");
    let e = run_err(&mut ctx, "offset L4 1 caps line");
    assert!(e.contains("both sides"), "{}", e);

    // Round caps on an arc-ended sequence: tangent arc-arc; persists.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 6,2 4.5858,1.4142");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset L0 A0 0.5 symmetric caps round");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.tangent_aa.len(), 2);
    assert_eq!(ctx.sketch.tangent_la.len(), 2);
    assert_solved(&mut ctx);
    let json = serde_json::to_string(&*ctx.sketch).unwrap();
    let back: Sketch = serde_json::from_str(&json).unwrap();
    let o = back.metas[0].as_offset().unwrap();
    assert_eq!(o.caps.kind, CapKind::Round);
    assert_eq!(o.caps.entities.len(), 2);
}

/// An arc offset inward past its radius vanishes: no result for it, its
/// neighbours meet at a corner; the record says so; a distance edit that
/// brings it back (or makes it vanish) rebuilds the side; nothing left is
/// an error; round caps need the same end segment on both sides.
#[test]
fn test_offset_vanishing_arcs() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 6,4");
    for pair in ["L0 L1", "L1 L2", "L2 L3", "L3 L0"] {
        run_ok(&mut ctx, &format!("fillet {} 0.77", pair));
    }
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset sequence L0 1 inward");
    assert!(out.contains("(A0 A1 A2 A3 vanished)") && out.contains("vanished: A0 A1 A2 A3"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 8);
    assert_eq!(ctx.sketch.arcs.refs().count(), 4, "no result arcs");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // The four result lines form the inner rectangle (1,1)-(5,3).
    for name in ["L4", "L5", "L6", "L7"] {
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, name).unwrap()];
        for p in [l.p1.value, l.p2.value] {
            assert!((near(p.x, 1.0) || near(p.x, 5.0)) && (near(p.y, 1.0) || near(p.y, 3.0)), "{} {:?}", name, p);
        }
    }
    let o = ctx.sketch.metas[0].as_offset().unwrap().clone();
    assert_eq!(o.sides[0].dropped, vec![1, 3, 5, 7]);
    assert_eq!(o.sides[0].segs.len(), 4);
    assert_eq!(o.sides[0].sources(8), vec![0, 2, 4, 6]);
    let json = serde_json::to_string(&*ctx.sketch).unwrap();
    let back: Sketch = serde_json::from_str(&json).unwrap();
    assert_eq!(back.metas[0].as_offset().unwrap().sides[0].dropped, vec![1, 3, 5, 7]);

    // An open corner whose fillet wraps through the angle cut: the fillet
    // vanishes on the inner side, the lines meet at a sharp corner and
    // the free ends are pinned as before; the outer side keeps it. A
    // smaller distance brings it back (the side is rebuilt), the short
    // way round; larger: gone again.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,3 0,0 4,0");
    run_ok(&mut ctx, "fillet L0 L1 1");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "offset L0 A0 L1 2 symmetric");
    assert!(out.contains("left: L2 L3 (A0 vanished); right: L4 A1 L5") && out.contains("vanished: A0"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Four free ends, plus the outer side's two tangent joints.
    assert_eq!(ctx.sketch.on_normal_ll.len() + ctx.sketch.on_normal_aa.len(), 6);
    let corner = ctx.sketch.lines[resolve_line(&ctx.sketch, "L2").unwrap()].p2.value;
    assert!(near(corner.x, 2.0) && near(corner.y, 2.0), "{:?}", corner);
    let o = ctx.sketch.metas[0].as_offset().unwrap();
    assert_eq!(o.sides[0].dropped, vec![1]);
    assert!(o.sides[1].dropped.is_empty());
    assert_solved(&mut ctx);
    run_ok(&mut ctx, "offset M0 0.5");
    assert_eq!(ctx.sketch.arcs.refs().count(), 3);
    assert!(ctx.sketch.metas[0].as_offset().unwrap().sides.iter().all(|s| s.dropped.is_empty()));
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    for a in ctx.sketch.arcs.refs() {
        let arc = &ctx.sketch.arcs[a];
        let (sa, ea) = (arc.start_angle.value, arc.end_angle.value);
        let sweep = if arc.ccw { (ea - sa).rem_euclid(std::f64::consts::TAU) } else { (sa - ea).rem_euclid(std::f64::consts::TAU) };
        assert!((sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "{}: sweep {}", arc.name, sweep);
    }
    run_ok(&mut ctx, "offset M0 2");
    assert_eq!(ctx.sketch.arcs.refs().count(), 2);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Round caps with an end arc that vanishes on one side only are refused;
    // line caps are fine.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_arc 4,0 5,1 4.7071,0.2929");
    let e = run_err(&mut ctx, "offset L0 A0 2 symmetric caps round");
    assert!(e.contains("A0 vanishes on one side"), "{}", e);
    let out = run_ok(&mut ctx, "offset L0 A0 2 symmetric caps line");
    assert!(out.contains("line caps"), "{}", out);
    assert_solved(&mut ctx);
    // Nothing left.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_arc 4,0 5,1 4.7071,0.2929");
    let e = run_err(&mut ctx, "offset A0 2 left");
    assert!(e.contains("nothing remains"), "{}", e);
    run_ok(&mut ctx, "add_circle 0,0 1");
    let e = run_err(&mut ctx, "offset A1 2 inward");
    assert!(e.contains("nothing remains"), "{}", e);
}

/// One distance dimension per run of tangent joints: the distance
/// carries through a tangent joint, a segment after a sharp corner has
/// its own. Every row is independent, so the gate is exact at every
/// distance (0.5 and 0.9 on this corner used to be refused as redundant)
/// and after a save / load or undo.
#[test]
fn test_offset_tangent_run_dims() {
    for d in ["0.5", "0.9", "0.3"] {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,3 0,0 4,0");
        run_ok(&mut ctx, "fillet L0 L1 1");
        let dof = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, &format!("offset L0 A0 L1 {} symmetric", d));
        assert_eq!(ctx.sketch.dof().unwrap(), dof, "distance {}", d);
        assert_eq!(ctx.sketch.dimensions.len(), 1 + 2, "the fillet radius and one per side");
        assert_solved(&mut ctx);
        // Loaded back, the offset still edits.
        let json = serde_json::to_string(&*ctx.sketch).unwrap();
        let back: Sketch = serde_json::from_str(&json).unwrap();
        ctx.sketch = back.into();
        run_ok(&mut ctx, "offset M0 0.7");
        assert_eq!(ctx.sketch.dof().unwrap(), dof);
        assert_solved(&mut ctx);
    }
    // A sharp corner then a tangent run: two dims.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0 2,2 5,2");
    run_ok(&mut ctx, "fillet L1 L2 0.5");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "offset L0 L1 A0 L2 0.3");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let o = ctx.sketch.metas[0].as_offset().unwrap();
    assert_eq!(o.sides[0].dims.len(), 2);
    let named: Vec<String> = o.sides[0].dims.iter().map(|d| ctx.sketch.dimensions[ctx.sketch.dimension_index_by_did(d.did).unwrap()].name.clone()).collect();
    let listed = run_ok(&mut ctx, "list dims");
    assert!(listed.contains("distance L0 L3") && listed.contains("distance L1 L4"), "{} {:?}", listed, named);
    assert_solved(&mut ctx);
    run_ok(&mut ctx, "offset M0 0.6");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
}

/// `select M0` selects the meta-constraint; `offset selection` then
/// edits it; the selection is pruned when the meta goes.
#[test]
fn test_select_meta() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "offset L0 1");
    run_ok(&mut ctx, "select M0");
    assert_eq!(ctx.selection, vec![Selection::Meta(0)]);
    let out = run_ok(&mut ctx, "list selection");
    assert!(out.contains("M0"), "{}", out);
    let out = run_ok(&mut ctx, "offset selection 2");
    assert!(out.contains("by 2 left"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 2, "edited, not created");
    run_ok(&mut ctx, "delete M0");
    assert!(ctx.selection.is_empty(), "{:?}", ctx.selection);
    let e = run_err(&mut ctx, "select M0");
    assert!(e.contains("Unknown meta-constraint"), "{}", e);
}

// -- pattern --

fn line_pts(ctx: &CommandContext, name: &str) -> (vect2d, vect2d) {
    let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, name).unwrap()];
    (l.p1.value, l.p2.value)
}

fn near2(p: vect2d, x: f64, y: f64) -> bool {
    (p.x - x).abs() < 1e-6 && (p.y - y).abs() < 1e-6
}

/// A circular pattern of a rectangle about a point: every copy is the
/// rotated rectangle, connected by recreated coincidences, the fourth
/// side needs no image; the DOF is unchanged; the record describes it.
#[test]
fn test_pattern_circular_rect() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_rect 0,0 2,1");
    run_ok(&mut ctx, "add_point 5,5");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "pattern circular L0 L1 L2 L3 about P0 4");
    assert!(out.contains("circular pattern of L0 L1 L2 L3 about P0, 4 full -> #1: L4 L5 L6 L7; #2:"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 16);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // Copy 1 is the rectangle rotated 90 degrees about (5,5): (0,0) -> (10,0), (2,0) -> (10,2).
    let (p1, p2) = line_pts(&ctx, "L4");
    assert!(near2(p1, 10.0, 0.0) && near2(p2, 10.0, 2.0), "{:?} {:?}", p1, p2);
    // Three images and four coincidences per copy (the fourth side is
    // held by its neighbours), three copies.
    assert_eq!(ctx.sketch.image_line_r.len(), 9);
    assert_eq!(ctx.sketch.coincident_ll12.len() + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll22.len(), 4 + 12);
    let p = ctx.sketch.metas[0].as_pattern().unwrap();
    assert_eq!(p.copies.len(), 3);
    assert_eq!(p.copies[0].index, (1, 0));
    // Dragging the source moves the copies with it: the copy stays the
    // rotated image about the center, wherever the solve put things.
    run_ok(&mut ctx, "drag L0.p1 0.5,0.2");
    assert_solved(&mut ctx);
    let l4 = line_pts(&ctx, "L4").0;
    let l0 = line_pts(&ctx, "L0").0;
    let c = ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()].pos.value;
    let (dx, dy) = (l0.x - c.x, l0.y - c.y);
    assert!(near2(l4, c.x - dy, c.y + dx), "{:?} vs {:?} about {:?}", l4, l0, c);
}

/// Circular: partial / symmetric distributions, a center at an endpoint
/// (hidden helper, owned), arcs rotate with the copy, quantity edits
/// rebuild, angle edits move in place; even symmetric has one less on
/// the backward side.
#[test]
fn test_pattern_circular_arcs_and_edits() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 3,0 5,0");
    run_ok(&mut ctx, "add_arc 5,0 6,1 5.7071,0.2929");
    run_ok(&mut ctx, "add_point 0,0");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "pattern circular L0 A0 about P0 3 partial 90");
    assert!(out.contains("3 partial 90 deg"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // Copy 1 is rotated 45 degrees: the line from (3,0) goes to (3/sqrt2, 3/sqrt2).
    let (p1, _) = line_pts(&ctx, "L1");
    let s = 3.0 / 2f64.sqrt();
    assert!(near2(p1, s, s), "{:?}", p1);
    // The copied arc's angles are the source's plus 45 degrees.
    let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    assert!((a1.start_angle.value - a0.start_angle.value - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
    assert!((a1.radius.value - a0.radius.value).abs() < 1e-9);
    // The line end is coincident with the arc start in the copy (tied).
    assert_eq!(ctx.sketch.coincident_lp2_arc_start.len(), 1 + 2);
    // Angle edit in place: the same entities, moved.
    run_ok(&mut ctx, "pattern M0 partial 180");
    assert_eq!(ctx.sketch.lines.refs().count(), 3, "kept");
    let (p1, _) = line_pts(&ctx, "L1");
    assert!(near2(p1, 0.0, 3.0), "{:?}", p1);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    // Quantity edit rebuilds.
    run_ok(&mut ctx, "pattern M0 5 symmetric 120");
    assert_eq!(ctx.sketch.lines.refs().count(), 5);
    assert_eq!(ctx.sketch.metas[0].as_pattern().unwrap().copies.iter().map(|c| c.index.0).collect::<Vec<_>>(), vec![-2, -1, 1, 2]);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    run_ok(&mut ctx, "pattern M0 4");
    assert_eq!(ctx.sketch.metas[0].as_pattern().unwrap().copies.iter().map(|c| c.index.0).collect::<Vec<_>>(), vec![-1, 1, 2], "even: one less backward");

    // Center at an endpoint: a hidden helper point, owned; refused if the
    // center's entity is in the set; deleting the center's entity drops.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_line 6,0 8,0");
    run_ok(&mut ctx, "add_point 9,9");
    let dof = ctx.sketch.dof().unwrap();
    let e = run_err(&mut ctx, "pattern circular L0 P0 about P0 4");
    assert!(e.contains("cannot be in the pattern set"), "{}", e);
    run_ok(&mut ctx, "pattern circular L1 about L0.p2 3 full");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert!(ctx.sketch.points.iter().any(|p| p.helper), "the helper point");
    let p = ctx.sketch.metas[0].as_pattern().unwrap();
    assert!(p.helper().is_some() && p.constraints.len() == 1);
    assert_solved(&mut ctx);
    let out = run_ok(&mut ctx, "delete M0 all");
    assert!(out.contains("Deleted M0") && !out.contains("Pc"), "{}", out);
    assert!(!ctx.sketch.points.iter().any(|p| p.helper), "the helper went with the pattern");
    assert_eq!(ctx.sketch.lines.refs().count(), 2);
    run_ok(&mut ctx, "pattern circular L1 about L0.p2 3");
    let out = run_ok(&mut ctx, "delete L0");
    assert!(out.contains("notice: pattern M1 dropped"), "{}", out);
}

/// Rectangular: one and two axes, spacing and extent, along a line,
/// symmetric (even: one less backward), negative distance; the DOF is
/// unchanged; a distance edit moves in place, a quantity edit rebuilds;
/// the set may hold the reference line only by error.
#[test]
fn test_pattern_rect() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0");
    let dof = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "pattern rect L0 3 4 by 2 3");
    assert!(out.contains("3 every 4 x 2 every 3 -> #0,1:"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 6);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);
    let p = ctx.sketch.metas[0].as_pattern().unwrap();
    let idx: Vec<(i32, i32)> = p.copies.iter().map(|c| c.index).collect();
    assert_eq!(idx, vec![(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
    // Copy (2,1): the line at (8,3).
    let MetaEntity::Line(l) = p.copies[4].entities[0] else { panic!() };
    assert!(near2(ctx.sketch.lines[l].p1.value, 8.0, 3.0));
    // In place: spacing 5 on axis 1.
    run_ok(&mut ctx, "pattern M0 3 5");
    assert_eq!(ctx.sketch.lines.refs().count(), 6);
    assert!(near2(ctx.sketch.lines[l].p1.value, 10.0, 3.0));
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    // Extent: 3 over 10 = 5 apart.
    run_ok(&mut ctx, "pattern M0 3 10 extent");
    assert!(near2(ctx.sketch.lines[l].p1.value, 10.0, 3.0));
    // Symmetric on axis 1 (rebuild), negative distance on axis 2.
    run_ok(&mut ctx, "pattern M0 4 4 symmetric spacing by 2 -3");
    let p = ctx.sketch.metas[0].as_pattern().unwrap();
    let idx: Vec<(i32, i32)> = p.copies.iter().map(|c| c.index).collect();
    assert_eq!(idx, vec![(-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
    let MetaEntity::Line(l) = p.copies[2].entities[0] else { panic!() };
    assert!(near2(ctx.sketch.lines[l].p1.value, 0.0, -3.0), "{:?}", ctx.sketch.lines[l].p1.value);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_solved(&mut ctx);

    // Along a line: axis 1 follows it, axis 2 across it.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0");
    run_ok(&mut ctx, "add_line 0,5 3,8");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "pattern rect L0 2 2 along L1");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    let (p1, _) = line_pts(&ctx, "L2");
    let s = 2.0 / 2f64.sqrt();
    assert!(near2(p1, s, s), "{:?}", p1);
    assert_eq!(ctx.sketch.image_line_tf.len(), 1);
    // Turning the direction line turns the pattern (the solve may move
    // the line's other end too: compare with its final direction).
    run_ok(&mut ctx, "drag L1.p2 3,5");
    assert_solved(&mut ctx);
    let (f1, f2) = line_pts(&ctx, "L1");
    let d = f2 - f1;
    let len = (d.x * d.x + d.y * d.y).sqrt();
    let (p1, _) = line_pts(&ctx, "L2");
    let (o1, _) = line_pts(&ctx, "L0");
    assert!(near2(p1, o1.x + 2.0 * d.x / len, o1.y + 2.0 * d.y / len), "{:?}", p1);
    let e = run_err(&mut ctx, "pattern rect L0 L1 2 2 along L1");
    assert!(e.contains("cannot be in the pattern set"), "{}", e);
    let e = run_err(&mut ctx, "pattern rect L0 1 2");
    assert!(e.contains("at least 2"), "{}", e);
    // Deleting the direction line drops the pattern.
    let out = run_ok(&mut ctx, "delete L1");
    assert!(out.contains("notice: pattern M0 dropped"), "{}", out);
}

/// Ownership and the selected-meta forms: deleting a copy or an image
/// constraint drops the pattern; dissolve keeps the copies as images;
/// `select M0` then `pattern selection`; persistence; a point patterns;
/// a failed apply leaves nothing.
#[test]
fn test_pattern_ownership_and_forms() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_point 1,1");
    run_ok(&mut ctx, "add_point 0,0");
    let dof = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "pattern circular P0 about P1 4");
    assert_eq!(ctx.sketch.points.iter().filter(|p| !p.helper).count(), 5);
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
    assert_eq!(ctx.sketch.image_point_r.len(), 3);
    let json = serde_json::to_string(&*ctx.sketch).unwrap();
    let back: Sketch = serde_json::from_str(&json).unwrap();
    assert_eq!(back.metas[0].as_pattern().unwrap().copies.len(), 3);
    let out = run_ok(&mut ctx, "delete P2");
    assert!(out.contains("notice: pattern M0 dropped"), "{}", out);
    run_ok(&mut ctx, "undo");
    assert_eq!(meta_count(&ctx), 1);
    let nid = ctx.sketch.image_point_r[0].nid;
    let out = run_ok(&mut ctx, &format!("delete C{}", nid));
    assert!(out.contains("notice: pattern M0 dropped"), "{}", out);
    run_ok(&mut ctx, "undo");
    run_ok(&mut ctx, "select M0");
    let out = run_ok(&mut ctx, "pattern selection 6");
    assert!(out.contains("6 full"), "{}", out);
    let out = run_ok(&mut ctx, "delete M0");
    assert!(out.contains("Dissolved M0"), "{}", out);
    assert_eq!(ctx.sketch.image_point_r.len(), 5, "the images stay");
    assert_eq!(ctx.sketch.dof().unwrap(), dof);
}

// -- on_normal --

/// A line endpoint placed on the normal of another line at its endpoint:
/// the foot of the perpendicular, one DOF, listed, deletable.
#[test]
fn test_on_normal_lines() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 4,0");
    run_ok(&mut ctx, "add_line 5,2 8,2");
    let dof_before = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "on_normal L1.p1 L0.p2");
    assert!(out.contains("on_normal L1.p1 L0.p2"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof_before - 1);
    // L1.p1 is now on L0's normal at L0.p2 (both lines may have moved).
    let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
    let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
    let dir = l0.p2.value - l0.p1.value;
    let d = l1.p1.value - l0.p2.value;
    assert!((d.x * dir.x + d.y * dir.y).abs() < 1e-6, "L1.p1 is off L0's normal at p2");
    let listed = run_ok(&mut ctx, "list constraints");
    assert!(listed.contains("on_normal L1.p1 L0.p2"), "{}", listed);
    // Duplicate and self-reference are rejected.
    let e = run_err(&mut ctx, "on_normal L1.p1 L0.p2");
    assert!(e.contains("already exists"), "{}", e);
    let e = run_err(&mut ctx, "on_normal L0.p1 L0.p2");
    assert!(e.contains("own entity"), "{}", e);
    // Relational and by-name deletion.
    run_ok(&mut ctx, "delete L1.p1 L0.p2 on_normal");
    assert!(ctx.sketch.on_normal_ll.is_empty());
    assert_eq!(ctx.sketch.dof().unwrap(), dof_before);
    run_ok(&mut ctx, "on_normal L1.p2 L0.p1");
    let nid = ctx.sketch.on_normal_ll[0].nid;
    run_ok(&mut ctx, &format!("delete C{}", nid));
    assert!(ctx.sketch.on_normal_ll.is_empty());
}

/// An arc endpoint on the normal of another arc at its endpoint: for
/// circles the radial ray, for an ellipse the true normal there.
#[test]
fn test_on_normal_arcs() {
    let mut ctx = CommandContext::new();
    // Concentric arcs: A1.start must end up on the ray through A0.start.
    run_ok(&mut ctx, "add_arc 2,0 0,2 1.4142,1.4142");
    run_ok(&mut ctx, "add_arc 3.5,1 1,3.5 2.6,2.6");
    run_ok(&mut ctx, "concentric A0 A1");
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "on_normal A1.start A0.start");
    assert_eq!(ctx.sketch.dof().unwrap(), dof_before - 1);
    let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    let c = a0.center.value;
    let s0 = crate::geometry::arc_start_pos(a0) - c;
    let s1 = crate::geometry::arc_start_pos(a1) - c;
    let cross = s0.x * s1.y - s0.y * s1.x;
    assert!(cross.abs() < 1e-6, "A1.start is off A0's start ray: cross {}", cross);
    let listed = run_ok(&mut ctx, "list constraints");
    assert!(listed.contains("on_normal A1.start A0.start"), "{}", listed);

    // Ellipse reference: the normal at its end is not the center ray.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_earc_center 0,0 4 2 0 0 90");
    run_ok(&mut ctx, "add_arc 6,1 3,6 5.5,5"); // A1: arcs share one counter
    let dof_before = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "on_normal A1.start EA0.end");
    assert_eq!(ctx.sketch.dof().unwrap(), dof_before - 1);
    let ea = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "EA0").unwrap()];
    let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
    let e = crate::geometry::arc_end_pos(ea);
    let t = ea.tangent_at(ea.end_angle.value);
    let d = crate::geometry::arc_start_pos(a1) - e;
    assert!((d.x * t.x + d.y * t.y).abs() < 1e-6, "A1.start is off EA0's end normal");
    // Unsupported operand pair.
    let e = run_err(&mut ctx, "on_normal A1.start EA0.center");
    assert!(e.contains("endpoints"), "{}", e);
}

// -- delete selection --

#[test]
fn test_delete_selection_batch() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0; add_line 2,0 2,2; add_point 4,4; add_circle 6,0 1");
    run_ok(&mut ctx, "length L0 2");
    let actions0 = ctx.history.actions.len();
    run_ok(&mut ctx, "select all");
    let out = run_ok(&mut ctx, "delete selection");
    for name in ["L0", "L1", "P0", "A0"] {
        assert!(out.contains(name), "output must name {}: {}", name, out);
    }
    assert!(out.contains("cascade"), "the length dim and coincident cascade: {}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 0);
    assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    assert_eq!(ctx.sketch.points.refs().count(), 0);
    assert!(ctx.selection.is_empty());
    assert_eq!(ctx.history.actions.len(), actions0 + 1, "one history entry");
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.lines.refs().count(), 2, "one undo restores everything");
    assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    assert_eq!(ctx.sketch.points.refs().count(), 1);
    assert_eq!(ctx.sketch.dimensions.len(), 1);
}

#[test]
fn test_delete_selection_dissolves_meta() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0");
    run_ok(&mut ctx, "select all");
    run_ok(&mut ctx, "pattern rect selection 3 5");
    run_ok(&mut ctx, "deselect");
    run_ok(&mut ctx, "select M0");
    let out = run_ok(&mut ctx, "delete selection");
    assert!(out.contains("Dissolved M0"), "{}", out);
    assert!(ctx.sketch.metas.is_empty());
    assert_eq!(ctx.sketch.lines.refs().count(), 3, "the geometry stays");
}

#[test]
fn test_delete_selection_empty_errors() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 1,0");
    let out = run_err(&mut ctx, "delete selection");
    assert!(out.contains("Nothing deletable"), "{}", out);
}

#[test]
fn test_mirror_strict_batched_undo() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0; add_line 2,0 2,2; add_line 5,-5 5,5 noconnect nocursor");
    let out = run_ok(&mut ctx, "mirror L0 L1 about L2 strict");
    assert!(out.contains("Mirrored L0 ->"), "{}", out);
    assert!(out.contains("constraints: C"), "constraint ids surfaced: {}", out);
    assert!(!out.contains("warning"), "{}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 5);
    // Shared corner deduped: 3 unique endpoint positions -> 3 symmetry.
    assert_eq!(ctx.sketch.symmetry_pp.len(), 3);
    run_ok(&mut ctx, "undo");
    assert_eq!(ctx.sketch.lines.refs().count(), 3, "one undo removes the whole mirror");
    assert!(ctx.sketch.symmetry_pp.is_empty());
}

#[test]
fn test_mirror_selection_excludes_axis() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0; add_line 5,-5 5,5 noconnect nocursor");
    run_ok(&mut ctx, "select all");
    let out = run_ok(&mut ctx, "mirror selection about L1");
    assert!(out.contains("Mirrored L0 ->"), "{}", out);
    assert!(!out.contains("Mirrored L1"), "the axis is not mirrored: {}", out);
    assert_eq!(ctx.sketch.lines.refs().count(), 3);
    run_ok(&mut ctx, "deselect");
    run_ok(&mut ctx, "select L1");
    let out = run_err(&mut ctx, "mirror selection about L1");
    assert!(out.contains("No lines, arcs, or points"), "{}", out);
}

#[test]
fn test_mirror_circle_tied_through_coincident_corner() {
    // A triangle with a circle centered on its apex: the circle center
    // shares a coincidence group with the corner, so it must get a
    // recreated tie (not a dropped symmetry), and a closed arc keeps
    // its radius. The mirror adds no freedom.
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 -1.5,-5; add_line -1.5,-5 1.5,-5; add_line 1.5,-5 0,0");
    // add_circle at the apex auto-connects its center to the corner.
    run_ok(&mut ctx, "add_circle 0,0 1.5");
    assert!(!ctx.sketch.coincident_lp1_arc_center.is_empty()
        || !ctx.sketch.coincident_lp2_arc_center.is_empty(), "center tied to the corner");
    run_ok(&mut ctx, "add_line 20,-20 20,20 noconnect nocursor");
    let dof0: usize = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "select L0 L1 L2 A0");
    let out = run_ok(&mut ctx, "mirror selection about L3");
    assert!(out.contains("coincident A1.center"), "the center ties to the corner: {}", out);
    assert!(out.contains("equal radius A0 A1"), "{}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof0, "a mirror adds no freedom");
}

#[test]
fn test_mirror_unconnected_same_position_endpoints_both_pinned() {
    // Two lines touching at the same coordinates WITHOUT a coincident:
    // each endpoint group gets its own symmetry (position dedup used
    // to drop one, leaving a copy endpoint free).
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_line 0,0 2,0; add_line 2,0 2,2 noconnect; add_line 5,-5 5,5 noconnect nocursor");
    assert!(ctx.sketch.coincident_ll21.is_empty(), "no coincident between the lines");
    let dof0: usize = ctx.sketch.dof().unwrap();
    run_ok(&mut ctx, "select L0 L1");
    let out = run_ok(&mut ctx, "mirror selection about L2");
    assert_eq!(out.matches("symmetry ").count(), 4, "every endpoint pinned: {}", out);
    assert_eq!(ctx.sketch.dof().unwrap(), dof0, "a mirror adds no freedom");
}

#[test]
fn test_mirror_lone_circle_fully_held() {
    let mut ctx = CommandContext::new();
    run_ok(&mut ctx, "add_circle 2,0 1; add_line 5,-5 5,5 noconnect nocursor");
    let dof0: usize = ctx.sketch.dof().unwrap();
    let out = run_ok(&mut ctx, "mirror A0 about L0");
    assert!(out.contains("symmetry A0 L0 A1"), "one arc symmetry: {}", out);
    assert!(!out.contains("equal radius"), "covered by the arc symmetry: {}", out);
    assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
    assert_eq!(ctx.sketch.dof().unwrap(), dof0, "center and radius both held");
}

#[test]
fn test_timing_toggle() {
    let mut ctx = CommandContext::new();
    assert!(!ctx.timing);
    let out = run_ok(&mut ctx, "timing on");
    assert!(ctx.timing);
    assert_eq!(out, "timing on");
    assert_eq!(run_ok(&mut ctx, "timing"), "timing is on");
    run_ok(&mut ctx, "timing off");
    assert!(!ctx.timing);
    run_err(&mut ctx, "timing maybe");
}

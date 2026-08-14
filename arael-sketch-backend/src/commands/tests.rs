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

fn list_constraints_output(ctx: &mut CommandContext) -> String {
    run_ok(ctx, "list constraints")
}

// 6A: Display tests -- list constraints shows no Pc names

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
    assert_eq!(ctx.sketch.symmetry_pp.len(), 1); // center only for circles
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

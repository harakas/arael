// Constraint-registry regressions: the bug class where a constraint
// collection was missing from one of the hand-maintained walks
// (delete cascades, helper cleanup, consolidation, naming). The
// registry makes the walks total; these tests pin the previously
// broken instances.

use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch_solver::{
    ArcLineParallel, AxisDistanceLP1, CoincidentArcCenterStart, CoincidentArcStartCenter,
    CoincidentLP1, MidpointArcPoint, MidpointLP1, Parallel, PointOnLine, Sketch, SymmetryPP,
    registry::CONSTRAINT_COLLECTION_COUNT,
};

#[test]
fn walker_visits_every_collection() {
    let mut s = Sketch::new();
    let mut n = 0;
    s.for_each_constraint_collection(|_, _, _| n += 1);
    assert_eq!(n, CONSTRAINT_COLLECTION_COUNT);
    let mut n_ref = 0;
    s.for_each_constraint_collection_ref(|_, _, _| n_ref += 1);
    assert_eq!(n_ref, CONSTRAINT_COLLECTION_COUNT);
}

// Deleting the arc used to leave arc_line_parallel holding a dangling
// ref (the collection was missing from delete_arc), and the constraint
// never received a name (missing from assign_constraint_names).
#[test]
fn arc_line_parallel_is_named_and_cascades_on_delete() {
    let mut s = Sketch::new();
    let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0));
    let e = s.add_ellipse(vect2d::new(2.0, 3.0), 2.0, 1.0, 0.3, true);
    s.arc_line_parallel.push(ArcLineParallel {
        arc: e,
        line: l,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    s.assign_constraint_names();
    assert_ne!(s.arc_line_parallel[0].nid, 0, "constraint must be named");
    assert!(
        s.constraint_nid_cid_pairs()
            .iter()
            .any(|&(nid, _)| nid == s.arc_line_parallel[0].nid),
        "constraint must be visible to nid/cid consumers"
    );

    s.delete_arc(e);
    assert!(s.arc_line_parallel.is_empty(), "cascade must drop the constraint");
    // The old bug: list_constraints panicked on the dangling ref.
    let _ = s.list_constraints();
    let _ = s.solve();
}

// A helper whose only purpose is an axis-distance constraint was
// classified purposeless (the purpose set missed nine collections) and
// deleted, silently destroying the user's dimension.
#[test]
fn helper_with_axis_distance_purpose_survives_cleanup() {
    let mut s = Sketch::new();
    let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 1.0));
    let h = s.add_helper_point(vect2d::new(5.0, 5.0));
    s.coincident_lp1.push(CoincidentLP1 {
        line: l,
        point: h,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    s.axis_distance_lp1.push(AxisDistanceLP1 {
        line: l,
        point: h,
        distance: 2.0,
        horizontal: true,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    s.cleanup_helper_points();
    assert!(s.points.get(h).is_some(), "helper with a purpose must survive");
    assert_eq!(s.axis_distance_lp1.len(), 1);

    // Control: drop the purpose and the helper goes, taking the bridge.
    s.axis_distance_lp1.clear();
    s.cleanup_helper_points();
    assert!(s.points.get(h).is_none(), "purposeless helper must be removed");
    assert!(s.coincident_lp1.is_empty());
}

// Consolidation merges same-position helpers; midpoint_arc_point was
// missing from the remap and kept a ref to the removed twin.
#[test]
fn consolidation_remaps_midpoint_arc_point() {
    let mut s = Sketch::new();
    let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let a = s.add_arc(vect2d::new(0.0, 2.0), 1.0, 0.0, std::f64::consts::PI, false);
    let h1 = s.add_helper_point(vect2d::new(1.0, 1.0));
    let h2 = s.add_helper_point(vect2d::new(1.0, 1.0));
    // Give both helpers a bridge so cleanup keeps them apart from the merge.
    s.coincident_lp1.push(CoincidentLP1 { line: l, point: h1, nid: 0, cid: 0, hb: CrossBlock::new() });
    s.coincident_lp1.push(CoincidentLP1 { line: l, point: h2, nid: 0, cid: 0, hb: CrossBlock::new() });
    s.midpoint_arc_point.push(MidpointArcPoint {
        point: h2,
        arc: a,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    s.consolidate_helper_constraints();
    let p = s.midpoint_arc_point[0].point;
    assert!(
        s.points.get(p).is_some(),
        "midpoint_arc_point must be remapped to the surviving helper"
    );
}

// The old hand-written list walk emitted every midpoint_lp1/lp2/
// arc_start/arc_end constraint twice, the second copy with swapped
// operands. The registry lists each constraint exactly once, in the
// order the residual means: line.p1 sits at the midpoint of target.
#[test]
fn midpoint_lp1_is_listed_exactly_once() {
    let mut s = Sketch::new();
    let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 1.0));
    let l1 = s.add_line(vect2d::new(0.0, 2.0), vect2d::new(2.0, 2.0));
    s.midpoint_lp1.push(MidpointLP1 {
        line: l0, target: l1, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    s.assign_constraint_names();
    let entries: Vec<String> = s.list_constraints()
        .into_iter().filter(|e| e.contains("midpoint")).collect();
    assert_eq!(entries.len(), 1, "{:?}", entries);
    assert!(entries[0].ends_with("midpoint L0.p1 L1"), "{}", entries[0]);
}

// Coincidence collections share one dedup key space: the same
// endpoint pair expressed through two different collections is a
// duplicate. The old per-collection dedup kept both.
#[test]
fn coincidence_dedup_is_cross_collection() {
    let mut s = Sketch::new();
    let a0 = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, 1.0, false);
    let a1 = s.add_arc(vect2d::new(3.0, 0.0), 1.0, 0.0, 1.0, false);
    s.coincident_arc_start_center.push(CoincidentArcStartCenter {
        a: a0, b: a1, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    s.coincident_arc_center_start.push(CoincidentArcCenterStart {
        a: a1, b: a0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    s.dedup_constraints();
    // Exactly one survives; which collection keeps it follows the
    // registry walk order, not push order.
    assert_eq!(
        s.coincident_arc_start_center.len() + s.coincident_arc_center_start.len(),
        1,
        "cross-collection duplicate must be removed"
    );
}

// Symmetric pairs dedup order-free; symmetry constraints dedup with
// the sides swappable and the mirror exact. symmetry_pp previously
// had no dedup at all.
#[test]
fn symmetric_and_mirror_dedup_keys() {
    let mut s = Sketch::new();
    let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0));
    let l1 = s.add_line(vect2d::new(0.0, 1.0), vect2d::new(1.0, 1.0));
    s.parallel.push(Parallel { a: l0, b: l1, nid: 0, cid: 0, hb: CrossBlock::new() });
    s.parallel.push(Parallel { a: l1, b: l0, nid: 0, cid: 0, hb: CrossBlock::new() });
    s.dedup_constraints();
    assert_eq!(s.parallel.len(), 1);

    let p0 = s.add_point(vect2d::new(0.0, 2.0));
    let p1 = s.add_point(vect2d::new(2.0, 2.0));
    let mirror = s.add_line(vect2d::new(1.0, 0.0), vect2d::new(1.0, 4.0));
    let sym = |a, c| SymmetryPP {
        a, c, line: mirror, nid: 0, cid: 0,
        hb_ac: CrossBlock::new(), hb_al: CrossBlock::new(), hb_cl: CrossBlock::new(),
    };
    s.symmetry_pp.push(sym(p0, p1));
    s.symmetry_pp.push(sym(p1, p0));
    s.dedup_constraints();
    assert_eq!(s.symmetry_pp.len(), 1, "swapped sides are the same symmetry");
}

// The horizontal flag is identity for axis distances: hdistance and
// vdistance on the same referents are different constraints, not
// duplicates of each other.
#[test]
fn axis_distance_flag_is_dedup_identity() {
    let mut s = Sketch::new();
    let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 1.0));
    let p = s.add_point(vect2d::new(5.0, 5.0));
    for horizontal in [true, false] {
        s.axis_distance_lp1.push(AxisDistanceLP1 {
            line: l, point: p, distance: 2.0, horizontal, nid: 0, cid: 0,
            hb: CrossBlock::new(),
        });
    }
    s.dedup_constraints();
    assert_eq!(s.axis_distance_lp1.len(), 2, "hdistance and vdistance are distinct");
}

// A point-on-line residual divides by line length. Collapse the line
// after creation -- something a solve can produce -- and every DOF /
// analysis path must return an error, not panic on a NaN sort.
#[test]
fn degenerate_geometry_errors_instead_of_panicking() {
    let mut s = Sketch::new();
    let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0));
    let p = s.add_point(vect2d::new(2.0, 1.0));
    s.point_on_line.push(PointOnLine { point: p, line: l, nid: 0, cid: 0, hb: CrossBlock::new() });
    let p1 = s.lines[l].p1.value;
    s.lines[l].p2.value = p1;
    assert!(s.dof().is_err(), "rank path must error");
    assert!(s.compute_dof(true).is_err(), "SVD/analyze path must error");
    assert!(s.compute_dof_eigenvalues(false).is_err(), "eigen path must error");
}

// The rank cache is keyed to the structure generation like the DOF
// cache, and dof() fills it: after any DOF query the probe basis is
// warm, so a drag start pays nothing. Structural mutation invalidates
// both with no explicit clear; value-only access preserves them.
#[test]
fn rank_cache_is_keyed_and_filled_by_dof() {
    let mut cell = arael_sketch_solver::SketchCell::new(Sketch::new());
    cell.get_mut().add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 0.0));
    cell.ensure_rank().unwrap();
    assert!(cell.cached_rank().is_some());
    assert_eq!(cell.cached_dof(), Some(4), "ensure_rank fills the DOF cache");

    cell.get_mut().add_line(vect2d::new(0.0, 1.0), vect2d::new(2.0, 1.0));
    assert!(cell.cached_rank().is_none(), "structural change must invalidate");

    let d = cell.dof().unwrap();
    assert_eq!(d, 8);
    assert!(cell.cached_rank().is_some(), "dof() must leave the basis warm");

    cell.mutate_values(|s| { let _ = s; });
    assert!(cell.cached_rank().is_some(), "value-only access must preserve");
}

// The DOF cache is keyed to the structure generation: a mutation
// through the cell's mutable door invalidates it with no explicit
// clear anywhere -- the historical bug was a forgotten clear serving
// a stale DOF forever.
#[test]
fn cached_dof_is_keyed_to_the_structure_generation() {
    let mut cell = arael_sketch_solver::SketchCell::new(Sketch::new());
    cell.get_mut().add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 0.0));
    let d = cell.dof().unwrap();
    assert_eq!(cell.cached_dof(), Some(d));

    // Raw structural change, deliberately without any cache clear.
    cell.get_mut().add_line(vect2d::new(0.0, 1.0), vect2d::new(2.0, 1.0));
    assert_eq!(cell.cached_dof(), None, "stale cache served across a structural change");
    assert_eq!(cell.dof().unwrap(), d + 4);

    // Value-only access does not invalidate.
    let d2 = cell.dof().unwrap();
    cell.mutate_values(|s| { let _ = s; });
    assert_eq!(cell.cached_dof(), Some(d2));
}

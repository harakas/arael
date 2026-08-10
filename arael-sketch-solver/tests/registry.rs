// Constraint-registry regressions: the bug class where a constraint
// collection was missing from one of the hand-maintained walks
// (delete cascades, helper cleanup, consolidation, naming). The
// registry makes the walks total; these tests pin the previously
// broken instances.

use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch_solver::{
    ArcLineParallel, AxisDistanceLP1, CoincidentLP1, MidpointArcPoint, Sketch,
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

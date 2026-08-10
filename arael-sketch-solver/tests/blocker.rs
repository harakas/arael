// Integration tests for blocker analysis against real sketch
// jacobians. Builds a pre-apply and post-apply Sketch, runs
// calc_jacobian, and exercises analyze_blockers over the real
// macro-generated row layout.

use arael::simple_lm::RootProblem;
use arael::model::{CrossBlock, JacobianModel};
use arael::vect::vect2d;
use arael_sketch_solver::*;

/// Build post-apply jacobian and nid-diff candidate cids, then call
/// analyze. Returns the BlockerReport or None if analysis declined.
fn run_analysis(
    pre: &mut Sketch,
    post: &mut Sketch,
) -> Option<BlockerReport> {
    // Pre-apply: populate cids / nids by running calc_jacobian once.
    let mut pre_params = Vec::new();
    pre.serialize(&mut pre_params);
    let _ = pre.calc_jacobian(&pre_params);
    let pre_nids: std::collections::HashSet<u32> = pre
        .constraint_nid_cid_pairs()
        .into_iter()
        .map(|(nid, _)| nid)
        .collect();

    // Post-apply: drift zeroed so the weak regulariser does not
    // dominate row-span (matches the production call site).
    let saved = post.drift_isigma;
    post.drift_isigma = 0.0;
    let mut post_params = Vec::new();
    post.serialize(&mut post_params);
    let jac = post.calc_jacobian(&post_params);
    post.drift_isigma = saved;

    let candidate_cids: std::collections::HashSet<u32> = post
        .constraint_nid_cid_pairs()
        .into_iter()
        .filter(|(nid, _)| !pre_nids.contains(nid))
        .map(|(_, cid)| cid)
        .collect();

    arael_sketch_solver::blocker::analyze(&jac, &candidate_cids)
}

#[test]
fn coincident_pp_blocked_by_hdist_and_vdist_singly() {
    // Two points stacked via hdist=0 + vdist=0. A coincident_pp
    // added on top is 2-row and fully implied: either single
    // removal unblocks since each existing row pins only one axis.
    let mut base = Sketch::new();
    let a = base.add_point(vect2d::new(2.0, 2.0));
    let b = base.add_point(vect2d::new(2.0, 2.0));
    base.hdistance_pp.push(HorizontalDistancePP {
        a, b, distance: 0.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    base.vdistance_pp.push(VerticalDistancePP {
        a, b, distance: 0.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    base.assign_constraint_names();
    // Snapshot pre and candidate-added post.
    let snap = bincode::serialize(&base).expect("serialize");
    let mut pre: Sketch = bincode::deserialize(&snap).expect("deserialize pre");
    let mut post: Sketch = bincode::deserialize(&snap).expect("deserialize post");
    post.coincident_pp.push(CoincidentPP {
        a, b, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    post.assign_constraint_names();

    let report = run_analysis(&mut pre, &mut post)
        .expect("analysis runs on real sketch");
    assert_eq!(report.minimum_size, 1);
    // Two size-1 sets, one for each of hdist/vdist.
    assert_eq!(report.sets.len(), 2);

    // Map cids back to "C{nid}" names via the same nid->cid pairing
    // commands.rs uses for its display map.
    let nid_by_cid: std::collections::HashMap<u32, u32> = post
        .constraint_nid_cid_pairs()
        .into_iter()
        .map(|(nid, cid)| (cid, nid))
        .collect();
    let named: std::collections::BTreeSet<u32> = report.sets.iter()
        .map(|s| *nid_by_cid.get(&s[0]).expect("cid has nid"))
        .collect();
    // HDist was added first (nid=1), VDist second (nid=2). Both
    // individually block the CoincidentPP candidate.
    assert_eq!(named, [1u32, 2u32].into_iter().collect());
}

#[test]
fn distance_pp_blocked_by_existing_distance() {
    // Two points + a DistancePP. A second identical DistancePP is a
    // pure 1-row duplicate. Minimum blocker set: the first
    // DistancePP.
    let mut base = Sketch::new();
    let a = base.add_point(vect2d::new(0.0, 0.0));
    let b = base.add_point(vect2d::new(5.0, 0.0));
    base.distance_pp.push(DistancePP {
        a, b, distance: 5.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    base.assign_constraint_names();
    base.solve();

    let snap = bincode::serialize(&base).expect("serialize");
    let mut pre: Sketch = bincode::deserialize(&snap).expect("deserialize pre");
    let mut post: Sketch = bincode::deserialize(&snap).expect("deserialize post");
    post.distance_pp.push(DistancePP {
        a, b, distance: 5.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    post.assign_constraint_names();

    let report = run_analysis(&mut pre, &mut post)
        .expect("analysis runs on real sketch");
    assert_eq!(report.minimum_size, 1);
    assert_eq!(report.sets.len(), 1);
    // The single pre-existing DistancePP was nid=1.
    let nid_by_cid: std::collections::HashMap<u32, u32> = post
        .constraint_nid_cid_pairs()
        .into_iter()
        .map(|(nid, cid)| (cid, nid))
        .collect();
    assert_eq!(*nid_by_cid.get(&report.sets[0][0]).unwrap(), 1);
}

#[test]
fn expression_backed_dimension_in_dim_map() {
    // Length dimensions feed expr_constraints, not a collection.
    // dimension_cid_name_map must still resolve the expression cid
    // to the d<N> handle so the blocker hint can name it correctly.
    use arael_sketch_solver::dimensions::{Dimension, DimensionKind};
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    sketch.lines[l].constraints.has_length = true;
    sketch.lines[l].constraints.length = 5.0;
    sketch.user_params.push(arael_sketch_solver::UserParam {
        name: "base_length".into(),
        expr_str: "5".into(),
        value: 5.0,
        broken: false,
    });
    sketch.dimensions.push(Dimension {
        did: 0,
        kind: DimensionKind::LineLength(l),
        value: 5.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        // expr_str populated -> this dimension is backed by an
        // ExpressionConstraint rather than the entity-level
        // has_length attribute.
        expr_str: Some("base_length".into()),
        broken: false,
        derived: false,
        range: None,
    });
    sketch.prepare_expr_constraints();
    let mut params = Vec::new();
    sketch.serialize(&mut params);
    // Running calc_jacobian populates cids on expr_constraints.
    let _ = sketch.calc_jacobian(&params);

    let map = sketch.dimension_cid_name_map();
    // Expect one expr cid -> "d0".
    let mut found = false;
    for ec in &sketch.expr_constraints {
        if let Some(name) = map.get(&ec.cid) {
            assert_eq!(name, "d0");
            found = true;
        }
    }
    assert!(found, "expected an expr cid mapped to d0");
}

#[test]
fn independent_candidate_no_blocker() {
    // Existing DistancePP between P0/P1. A new DistancePP between
    // unrelated P2/P3 adds a fresh row; analysis should decline
    // (not a rejection scenario).
    let mut base = Sketch::new();
    let a = base.add_point(vect2d::new(0.0, 0.0));
    let b = base.add_point(vect2d::new(5.0, 0.0));
    let c = base.add_point(vect2d::new(1.0, 1.0));
    let d = base.add_point(vect2d::new(4.0, 1.0));
    base.distance_pp.push(DistancePP {
        a, b, distance: 5.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    base.assign_constraint_names();
    base.solve();

    let snap = bincode::serialize(&base).expect("serialize");
    let mut pre: Sketch = bincode::deserialize(&snap).expect("deserialize pre");
    let mut post: Sketch = bincode::deserialize(&snap).expect("deserialize post");
    post.distance_pp.push(DistancePP {
        a: c, b: d, distance: 3.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    post.assign_constraint_names();

    assert!(run_analysis(&mut pre, &mut post).is_none(),
        "independent candidate should not trigger blocker analysis");
}

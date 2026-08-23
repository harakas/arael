// Probe-row equivalence: the hand-built candidate rows in probe.rs
// must match the macro-generated Jacobian rows in direction, and the
// span-test answers must match the old push-constraint-and-recompute
// DOF comparison.

use arael::model::JacobianModel;
use arael::simple_lm::RootProblem;
use arael::model::CrossBlock;
use arael_sketch_solver::{probe, Collinear, CoincidentArcStart, CoincidentLL11, MidpointLP1, Perpendicular, PointOnArc, PointOnLine, Sketch};

// A generic two-line sketch: nothing axis-aligned, unequal lengths.
fn two_lines() -> (Sketch, arael::refs::Ref<arael_sketch_solver::Line>, arael::refs::Ref<arael_sketch_solver::Line>) {
    let mut s = Sketch::new();
    let a = s.add_line(
        arael::vect::vect2d::new(0.3, -0.2),
        arael::vect::vect2d::new(4.1, 2.7),
    );
    let b = s.add_line(
        arael::vect::vect2d::new(1.0, 5.0),
        arael::vect::vect2d::new(-2.2, 6.9),
    );
    (s, a, b)
}

// Macro row(s) with the given label, as sparse (index, value) entries.
fn macro_rows(s: &mut Sketch, label_prefix: &str) -> Vec<Vec<(u32, f64)>> {
    let mut params = Vec::new();
    s.serialize(&mut params);
    let jac = s.calc_jacobian(&params);
    jac.rows
        .iter()
        .filter(|r| r.label.starts_with(label_prefix))
        .map(|r| r.entries.clone())
        .collect()
}

// |cos angle| between two sparse rows; 1.0 = same direction up to sign/scale.
fn alignment(a: &[(u32, f64)], b: &[(u32, f64)]) -> f64 {
    let mut map = std::collections::HashMap::new();
    for &(i, v) in a {
        map.insert(i, v);
    }
    let dot: f64 = b.iter().map(|&(i, v)| v * map.get(&i).copied().unwrap_or(0.0)).sum();
    let na: f64 = a.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
    (dot / (na * nb)).abs()
}

#[test]
fn horizontal_and_vertical_rows_match_the_macro() {
    let (mut s, a, _) = two_lines();
    s.lines[a].constraints.horizontal = true;
    s.lines[a].constraints.h_dir_sign = 1.0;
    let rows = macro_rows(&mut s, "horizontal");
    // First row is the equality; the dir barrier row is inactive
    // (all-zero) at capture-consistent geometry and is filtered by the
    // macro's guard only at solve level, so compare against row 0.
    let probe_row = probe::horizontal_row(&s.lines[a]);
    assert!(alignment(&rows[0], &probe_row) > 1.0 - 1e-12);

    s.lines[a].constraints.horizontal = false;
    s.lines[a].constraints.vertical = true;
    s.lines[a].constraints.v_dir_sign = 1.0;
    let rows = macro_rows(&mut s, "vertical");
    let probe_row = probe::vertical_row(&s.lines[a]);
    assert!(alignment(&rows[0], &probe_row) > 1.0 - 1e-12);
}

#[test]
fn perpendicular_row_matches_the_macro() {
    let (mut s, a, b) = two_lines();
    s.perpendicular.push(Perpendicular {
        a,
        b,
        dir_sign: 1.0,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "Perpendicular");
    assert!(!rows.is_empty());
    let probe_row = probe::perpendicular_row(&s.lines[a], &s.lines[b]);
    let al = alignment(&rows[0], &probe_row);
    assert!(al > 1.0 - 1e-12, "alignment {}", al);
}

#[test]
fn collinear_rows_match_the_macro() {
    let (mut s, a, b) = two_lines();
    s.collinear.push(Collinear {
        a,
        b,
        nid: 0,
        cid: 0,
        hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "Collinear");
    assert_eq!(rows.len(), 2);
    let probe_rows = probe::collinear_rows(&s.lines[a], &s.lines[b]);
    for (i, pr) in probe_rows.iter().enumerate() {
        let al = alignment(&rows[i], pr);
        assert!(al > 1.0 - 1e-12, "row {} alignment {}", i, al);
    }
}

// Old-style probe: push the trial mutation, recompute DOF, roll back.
fn dof_of(s: &mut Sketch) -> usize {
    s.clear_cached_dof();
    let d = s.compute_dof(false).unwrap().dof;
    s.clear_cached_dof();
    d
}

#[test]
fn span_test_matches_push_and_recompute() {
    let (mut s, a, b) = two_lines();

    // Case 1: free line -- horizontal reduces DOF.
    let before = dof_of(&mut s);
    s.lines[a].constraints.horizontal = true;
    s.lines[a].constraints.h_dir_sign = 1.0;
    let after = dof_of(&mut s);
    s.lines[a].constraints.horizontal = false;
    assert!(after < before);
    let rank = s.rank_analysis().unwrap();
    assert!(rank.reduces_rank(&probe::horizontal_row(&s.lines[a])));

    // Case 2: already-horizontal line -- a second horizontal is inert.
    s.lines[a].constraints.horizontal = true;
    s.lines[a].constraints.h_dir_sign = 1.0;
    let rank = s.rank_analysis().unwrap();
    assert!(!rank.reduces_rank(&probe::horizontal_row(&s.lines[a])));
    // But vertical on the same line still reduces.
    assert!(rank.reduces_rank(&probe::vertical_row(&s.lines[a])));
    s.lines[a].constraints.horizontal = false;

    // Case 3: perpendicular -- reduces on a free pair, inert once present.
    let before = dof_of(&mut s);
    s.perpendicular.push(Perpendicular {
        a, b, dir_sign: 1.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let after = dof_of(&mut s);
    assert!(after < before);
    let rank_with = s.rank_analysis().unwrap();
    assert!(!rank_with.reduces_rank(&probe::perpendicular_row(&s.lines[a], &s.lines[b])));
    s.perpendicular.pop();
    let rank_without = s.rank_analysis().unwrap();
    assert!(rank_without.reduces_rank(&probe::perpendicular_row(&s.lines[a], &s.lines[b])));

    // Case 4: collinear rows on a free pair reduce; with Collinear
    // already present they are inert.
    assert!(probe::any_reduces_rank(
        &rank_without,
        &probe::collinear_rows(&s.lines[a], &s.lines[b]),
    ));
    s.collinear.push(Collinear { a, b, nid: 0, cid: 0, hb: CrossBlock::new() });
    let rank_col = s.rank_analysis().unwrap();
    assert!(!probe::any_reduces_rank(
        &rank_col,
        &probe::collinear_rows(&s.lines[a], &s.lines[b]),
    ));
}

// The GUI's background DOF worker computes the rank analysis on a
// bincode round-tripped copy of the sketch; the returned basis must
// answer probe rows built from the live sketch's Params. That holds
// because serialization order is deterministic over identical
// structure, so both assign identical parameter indices.
#[test]
fn rank_from_serialized_copy_answers_live_probes() {
    let (mut s, a, b) = two_lines();
    s.lines[a].constraints.horizontal = true;
    s.lines[a].constraints.h_dir_sign = 1.0;

    // Freshen the live sketch's parameter indices, as a solve would.
    let mut params = Vec::new();
    s.serialize(&mut params);

    let bytes = bincode::serialize(&s).unwrap();
    let mut copy: Sketch = bincode::deserialize(&bytes).unwrap();
    let from_copy = copy.rank_analysis().unwrap();
    let live = s.rank_analysis().unwrap();

    assert_eq!(from_copy.nullity, live.nullity);
    for rows in [
        vec![probe::horizontal_row(&s.lines[a])],
        vec![probe::vertical_row(&s.lines[a])],
        vec![probe::perpendicular_row(&s.lines[a], &s.lines[b])],
        probe::collinear_rows(&s.lines[a], &s.lines[b]).to_vec(),
    ] {
        for row in &rows {
            assert_eq!(from_copy.reduces_rank(row), live.reduces_rank(row));
        }
    }
    // And the expected verdicts themselves.
    assert!(!from_copy.reduces_rank(&probe::horizontal_row(&s.lines[a])));
    assert!(from_copy.reduces_rank(&probe::vertical_row(&s.lines[a])));
}

#[test]
fn fixed_endpoints_suppress_the_hint() {
    let mut s = Sketch::new();
    let a = s.add_line(
        arael::vect::vect2d::new(0.0, 0.0),
        arael::vect::vect2d::new(3.0, 1.0),
    );
    s.lines[a].p1 = arael::model::Param::fixed(s.lines[a].p1.value);
    s.lines[a].p2 = arael::model::Param::fixed(s.lines[a].p2.value);
    let rank = s.rank_analysis().unwrap();
    // Both endpoints pinned: horizontal cannot remove a free direction.
    assert!(!rank.reduces_rank(&probe::horizontal_row(&s.lines[a])));
}

#[test]
fn coincident_rows_match_the_macro() {
    let (mut s, a, b) = two_lines();
    s.coincident_ll11.push(CoincidentLL11 {
        a, b, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "CoincidentLL11");
    assert_eq!(rows.len(), 2);
    let pa = probe::point_pos(s.lines[a].p1.index(), s.lines[a].p1.value);
    let pb = probe::point_pos(s.lines[b].p1.index(), s.lines[b].p1.value);
    let probe_rows = probe::coincident_rows(&pa, &pb);
    for (i, pr) in probe_rows.iter().enumerate() {
        let al = alignment(&rows[i], pr);
        assert!(al > 1.0 - 1e-12, "row {} alignment {}", i, al);
    }
}

#[test]
fn midpoint_rows_match_the_macro() {
    let (mut s, a, b) = two_lines();
    s.midpoint_lp1.push(MidpointLP1 {
        line: a, target: b, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "MidpointLP1");
    assert_eq!(rows.len(), 2);
    let pa = probe::point_pos(s.lines[a].p1.index(), s.lines[a].p1.value);
    let mid = probe::line_midpoint_pos(&s.lines[b]);
    let probe_rows = probe::coincident_rows(&pa, &mid);
    for (i, pr) in probe_rows.iter().enumerate() {
        let al = alignment(&rows[i], pr);
        assert!(al > 1.0 - 1e-12, "row {} alignment {}", i, al);
    }
}

#[test]
fn on_line_row_matches_the_macro() {
    let (mut s, a, _) = two_lines();
    let p = s.add_point(arael::vect::vect2d::new(2.0, 1.3));
    s.point_on_line.push(PointOnLine {
        point: p, line: a, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "PointOnLine");
    assert_eq!(rows.len(), 1);
    let q = probe::point_pos(s.points[p].pos.index(), s.points[p].pos.value);
    let pr = probe::on_line_row(&s.lines[a], &q);
    let al = alignment(&rows[0], &pr);
    assert!(al > 1.0 - 1e-12, "alignment {}", al);
}

#[test]
fn on_arc_row_matches_the_macro() {
    let mut s = Sketch::new();
    let e = s.add_ellipse(arael::vect::vect2d::new(1.0, -0.5), 2.0, 1.2, 0.7, true);
    let p = s.add_point(arael::vect::vect2d::new(3.4, 0.9));
    s.point_on_arc.push(PointOnArc {
        point: p, arc: e, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "PointOnArc");
    assert_eq!(rows.len(), 1);
    let q = probe::point_pos(s.points[p].pos.index(), s.points[p].pos.value);
    let pr = probe::on_arc_row(&s.arcs[e], &q);
    let al = alignment(&rows[0], &pr);
    assert!(al > 1.0 - 1e-12, "alignment {}", al);
}

#[test]
fn arc_anchor_rows_match_the_macro() {
    // Open elliptic arc: the start-anchor derivative exercises center,
    // radius, radius_b, rotation and the start angle at once.
    let mut s = Sketch::new();
    let e = s.add_ellipse(arael::vect::vect2d::new(0.4, 1.1), 1.7, 0.9, 0.5, false);
    let p = s.add_point(arael::vect::vect2d::new(2.0, 2.0));
    s.coincident_arc_start.push(CoincidentArcStart {
        point: p, arc: e, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let rows = macro_rows(&mut s, "CoincidentArcStart");
    assert_eq!(rows.len(), 2);
    let q = probe::point_pos(s.points[p].pos.index(), s.points[p].pos.value);
    let anchor = probe::arc_anchor_pos(&s.arcs[e], probe::ArcAnchor::Start);
    let probe_rows = probe::coincident_rows(&q, &anchor);
    for (i, pr) in probe_rows.iter().enumerate() {
        let al = alignment(&rows[i], pr);
        assert!(al > 1.0 - 1e-12, "row {} alignment {}", i, al);
    }
}

// Constraint conflict detection for the sketch editor.
// Checks proposed actions against the current sketch state and returns
// an error message if a conflict is detected.

use arael::refs::Ref;
use arael_sketch_solver::*;
use crate::actions::Action;

fn line_name(sketch: &Sketch, r: Ref<Line>) -> String {
    sketch.lines[r].name.clone()
}

fn arc_name(sketch: &Sketch, r: Ref<Arc>) -> String {
    sketch.arcs[r].name.clone()
}

// Check if a pair (a, b) or (b, a) exists in a list of pairs.
fn pair_exists<T, F: Fn(&T) -> (Ref<Line>, Ref<Line>)>(
    list: &[T], a: Ref<Line>, b: Ref<Line>, get: F,
) -> bool {
    list.iter().any(|item| {
        let (x, y) = get(item);
        (x == a && y == b) || (x == b && y == a)
    })
}

fn arc_pair_exists<T, F: Fn(&T) -> (Ref<Arc>, Ref<Arc>)>(
    list: &[T], a: Ref<Arc>, b: Ref<Arc>, get: F,
) -> bool {
    list.iter().any(|item| {
        let (x, y) = get(item);
        (x == a && y == b) || (x == b && y == a)
    })
}

// ---------------------------------------------------------------------------
// Union-Find for parallel groups
// ---------------------------------------------------------------------------

fn uf_find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut Vec<usize>, a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Orientation {
    Unknown,
    Horizontal,
    Vertical,
}

// Build parallel groups and determine orientation per group.
// Returns (parent, group_orientation, group_source, num_lines).
// group_orientation[find(i)] = orientation of line i's group.
// group_source[find(i)] = line ref that caused the orientation.
fn build_orientation_info(sketch: &Sketch) -> (
    Vec<usize>,
    Vec<Orientation>,
    Vec<Option<Ref<Line>>>,
    usize,
) {
    let num_lines = sketch.lines.slot_count();
    let mut parent: Vec<usize> = (0..num_lines).collect();

    // Union lines connected by Parallel constraints
    for p in &sketch.parallel {
        let a = p.a.index() as usize;
        let b = p.b.index() as usize;
        uf_union(&mut parent, a, b);
    }

    // Determine orientation per group from H/V flags
    let mut group_orient: Vec<Orientation> = vec![Orientation::Unknown; num_lines];
    let mut group_source: Vec<Option<Ref<Line>>> = vec![None; num_lines];

    for r in sketch.lines.refs() {
        let l = &sketch.lines[r];
        let root = uf_find(&mut parent, r.index() as usize);
        if l.constraints.horizontal {
            group_orient[root] = Orientation::Horizontal;
            if group_source[root].is_none() {
                group_source[root] = Some(r);
            }
        }
        if l.constraints.vertical {
            group_orient[root] = Orientation::Vertical;
            if group_source[root].is_none() {
                group_source[root] = Some(r);
            }
        }
    }

    // Propagate orientation through Perpendicular links
    // We may need multiple passes until stable.
    let mut changed = true;
    while changed {
        changed = false;
        for perp in &sketch.perpendicular {
            let ra = uf_find(&mut parent, perp.a.index() as usize);
            let rb = uf_find(&mut parent, perp.b.index() as usize);
            match (group_orient[ra], group_orient[rb]) {
                (Orientation::Horizontal, Orientation::Unknown) => {
                    group_orient[rb] = Orientation::Vertical;
                    if group_source[rb].is_none() {
                        group_source[rb] = group_source[ra];
                    }
                    changed = true;
                }
                (Orientation::Vertical, Orientation::Unknown) => {
                    group_orient[rb] = Orientation::Horizontal;
                    if group_source[rb].is_none() {
                        group_source[rb] = group_source[ra];
                    }
                    changed = true;
                }
                (Orientation::Unknown, Orientation::Horizontal) => {
                    group_orient[ra] = Orientation::Vertical;
                    if group_source[ra].is_none() {
                        group_source[ra] = group_source[rb];
                    }
                    changed = true;
                }
                (Orientation::Unknown, Orientation::Vertical) => {
                    group_orient[ra] = Orientation::Horizontal;
                    if group_source[ra].is_none() {
                        group_source[ra] = group_source[rb];
                    }
                    changed = true;
                }
                _ => {}
            }
        }
    }

    (parent, group_orient, group_source, num_lines)
}

// Check if two groups are linked by a perpendicular constraint.
fn groups_are_perp_linked(
    sketch: &Sketch,
    parent: &mut Vec<usize>,
    group_a: usize,
    group_b: usize,
) -> bool {
    for perp in &sketch.perpendicular {
        let ra = uf_find(parent, perp.a.index() as usize);
        let rb = uf_find(parent, perp.b.index() as usize);
        if (ra == group_a && rb == group_b) || (ra == group_b && rb == group_a) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Main conflict checker
// ---------------------------------------------------------------------------

pub fn check_constraint_conflict(sketch: &Sketch, action: &Action) -> Option<String> {
    match action {
        // ----- Self-referential checks -----
        Action::ApplyParallel { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyPerpendicular { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyEqualLength { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyCoincidentLL11 { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyCoincidentLL12 { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyCoincidentLL21 { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyCoincidentLL22 { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplyConcentric { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", arc_name(sketch, *a)));
        }
        Action::ApplyEqualRadius { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", arc_name(sketch, *a)));
        }
        Action::ApplyTangentAA { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", arc_name(sketch, *a)));
        }
        Action::ApplyCollinear { a, b } if a == b => {
            return Some(format!("Cannot constrain {} to itself", line_name(sketch, *a)));
        }
        Action::ApplySymmetryLL { a, b, c } if a == b || b == c || a == c => {
            return Some("Symmetry requires 3 distinct lines".into());
        }

        // ----- Direct H/V conflict -----
        Action::ApplyHorizontal { lines } => {
            for r in lines {
                if sketch.lines[*r].constraints.vertical {
                    return Some(format!("{} is already vertical", line_name(sketch, *r)));
                }
                if sketch.lines[*r].constraints.horizontal {
                    return Some(format!("{} is already horizontal", line_name(sketch, *r)));
                }
            }
            // Transitive orientation check
            let (mut parent, group_orient, group_source, _) = build_orientation_info(sketch);
            for r in lines {
                let root = uf_find(&mut parent, r.index() as usize);
                if group_orient[root] == Orientation::Vertical {
                    let cause = group_source[root]
                        .map(|s| line_name(sketch, s))
                        .unwrap_or_else(|| "a line in the same group".to_string());
                    return Some(format!(
                        "Cannot make {} horizontal: {} in the same group is vertical",
                        line_name(sketch, *r), cause
                    ));
                }
                // Check perpendicular propagation: if this line's group is perp-linked
                // to an already-horizontal group, making it horizontal would conflict
                for perp in &sketch.perpendicular {
                    let ra = uf_find(&mut parent, perp.a.index() as usize);
                    let rb = uf_find(&mut parent, perp.b.index() as usize);
                    if ra == root && group_orient[rb] == Orientation::Horizontal {
                        let cause = group_source[rb]
                            .map(|s| line_name(sketch, s))
                            .unwrap_or_else(|| "a perpendicular line".to_string());
                        return Some(format!(
                            "Cannot make {} horizontal: perpendicular to {} which is also horizontal",
                            line_name(sketch, *r), cause
                        ));
                    }
                    if rb == root && group_orient[ra] == Orientation::Horizontal {
                        let cause = group_source[ra]
                            .map(|s| line_name(sketch, s))
                            .unwrap_or_else(|| "a perpendicular line".to_string());
                        return Some(format!(
                            "Cannot make {} horizontal: perpendicular to {} which is also horizontal",
                            line_name(sketch, *r), cause
                        ));
                    }
                }
            }
        }

        Action::ApplyVertical { lines } => {
            for r in lines {
                if sketch.lines[*r].constraints.horizontal {
                    return Some(format!("{} is already horizontal", line_name(sketch, *r)));
                }
                if sketch.lines[*r].constraints.vertical {
                    return Some(format!("{} is already vertical", line_name(sketch, *r)));
                }
            }
            // Transitive orientation check
            let (mut parent, group_orient, group_source, _) = build_orientation_info(sketch);
            for r in lines {
                let root = uf_find(&mut parent, r.index() as usize);
                if group_orient[root] == Orientation::Horizontal {
                    let cause = group_source[root]
                        .map(|s| line_name(sketch, s))
                        .unwrap_or_else(|| "a line in the same group".to_string());
                    return Some(format!(
                        "Cannot make {} vertical: {} in the same group is horizontal",
                        line_name(sketch, *r), cause
                    ));
                }
                // Check perpendicular propagation
                for perp in &sketch.perpendicular {
                    let ra = uf_find(&mut parent, perp.a.index() as usize);
                    let rb = uf_find(&mut parent, perp.b.index() as usize);
                    if ra == root && group_orient[rb] == Orientation::Vertical {
                        let cause = group_source[rb]
                            .map(|s| line_name(sketch, s))
                            .unwrap_or_else(|| "a perpendicular line".to_string());
                        return Some(format!(
                            "Cannot make {} vertical: perpendicular to {} which is also vertical",
                            line_name(sketch, *r), cause
                        ));
                    }
                    if rb == root && group_orient[ra] == Orientation::Vertical {
                        let cause = group_source[ra]
                            .map(|s| line_name(sketch, s))
                            .unwrap_or_else(|| "a perpendicular line".to_string());
                        return Some(format!(
                            "Cannot make {} vertical: perpendicular to {} which is also vertical",
                            line_name(sketch, *r), cause
                        ));
                    }
                }
            }
        }

        // ----- Duplicate + conflict checks for pair constraints -----
        Action::ApplyParallel { a, b } => {
            let a_name = line_name(sketch, *a);
            let b_name = line_name(sketch, *b);

            // Duplicate
            if pair_exists(&sketch.parallel, *a, *b, |p| (p.a, p.b)) {
                return Some(format!("{} and {} are already parallel", a_name, b_name));
            }
            // Conflict: already perpendicular
            if pair_exists(&sketch.perpendicular, *a, *b, |p| (p.a, p.b)) {
                return Some(format!("{} and {} are already perpendicular", a_name, b_name));
            }

            // Transitive: if groups have conflicting orientations
            let (mut parent, group_orient, _, _) = build_orientation_info(sketch);
            let ra = uf_find(&mut parent, a.index() as usize);
            let rb = uf_find(&mut parent, b.index() as usize);

            // If they are in the same group, they are already parallel (transitively)
            if ra == rb {
                return Some(format!("{} and {} are already parallel (transitively)", a_name, b_name));
            }

            // One H and one V
            let (oa, ob) = (group_orient[ra], group_orient[rb]);
            if (oa == Orientation::Horizontal && ob == Orientation::Vertical)
                || (oa == Orientation::Vertical && ob == Orientation::Horizontal)
            {
                return Some(format!(
                    "Cannot make {} and {} parallel: one is horizontal and the other is vertical",
                    a_name, b_name
                ));
            }

            // Merging groups linked by perpendicular would mean a group is perp to itself
            if groups_are_perp_linked(sketch, &mut parent, ra, rb) {
                return Some(format!(
                    "{} and {} are already perpendicular (transitively)",
                    a_name, b_name
                ));
            }
        }

        Action::ApplyPerpendicular { a, b } => {
            let a_name = line_name(sketch, *a);
            let b_name = line_name(sketch, *b);

            // Duplicate
            if pair_exists(&sketch.perpendicular, *a, *b, |p| (p.a, p.b)) {
                return Some(format!("{} and {} are already perpendicular", a_name, b_name));
            }
            // Conflict: already parallel
            if pair_exists(&sketch.parallel, *a, *b, |p| (p.a, p.b)) {
                return Some(format!("{} and {} are already parallel", a_name, b_name));
            }

            // Transitive: if in the same parallel group, they are parallel
            let (mut parent, group_orient, _, _) = build_orientation_info(sketch);
            let ra = uf_find(&mut parent, a.index() as usize);
            let rb = uf_find(&mut parent, b.index() as usize);

            if ra == rb {
                return Some(format!(
                    "Cannot make {} and {} perpendicular: they are parallel",
                    a_name, b_name
                ));
            }

            // Both groups have the same non-Unknown orientation
            let (oa, ob) = (group_orient[ra], group_orient[rb]);
            if oa != Orientation::Unknown && oa == ob {
                let orient_str = if oa == Orientation::Horizontal { "horizontal" } else { "vertical" };
                return Some(format!(
                    "Cannot make {} and {} perpendicular: both are {}",
                    a_name, b_name, orient_str
                ));
            }
        }

        Action::ApplyEqualLength { a, b } => {
            if pair_exists(&sketch.equal_length, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} already have equal length",
                    line_name(sketch, *a), line_name(sketch, *b)
                ));
            }
        }

        Action::ApplyConcentric { a, b } => {
            if arc_pair_exists(&sketch.concentric, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} are already concentric",
                    arc_name(sketch, *a), arc_name(sketch, *b)
                ));
            }
        }

        Action::ApplyEqualRadius { a, b } => {
            if arc_pair_exists(&sketch.equal_radius, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} already have equal radius",
                    arc_name(sketch, *a), arc_name(sketch, *b)
                ));
            }
        }

        Action::ApplyTangentAA { a, b } => {
            if arc_pair_exists(&sketch.tangent_aa, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} are already tangent",
                    arc_name(sketch, *a), arc_name(sketch, *b)
                ));
            }
        }

        Action::ApplyCollinear { a, b } => {
            if pair_exists(&sketch.collinear, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} are already collinear",
                    line_name(sketch, *a), line_name(sketch, *b)
                ));
            }
            // Conflict: perpendicular lines cannot be collinear
            if pair_exists(&sketch.perpendicular, *a, *b, |p| (p.a, p.b)) {
                return Some(format!(
                    "{} and {} are perpendicular, cannot be collinear",
                    line_name(sketch, *a), line_name(sketch, *b)
                ));
            }
        }

        _ => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use arael::model::CrossBlock;
    use arael::vect::vect2d;

    fn two_lines() -> (Sketch, Ref<Line>, Ref<Line>) {
        let mut s = Sketch::new();
        let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0));
        let l1 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(0.0, 1.0));
        (s, l0, l1)
    }

    fn three_lines() -> (Sketch, Ref<Line>, Ref<Line>, Ref<Line>) {
        let mut s = Sketch::new();
        let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0));
        let l1 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(0.0, 1.0));
        let l2 = s.add_line(vect2d::new(1.0, 1.0), vect2d::new(2.0, 1.0));
        (s, l0, l1, l2)
    }

    fn two_arcs() -> (Sketch, Ref<Arc>, Ref<Arc>) {
        let mut s = Sketch::new();
        let a0 = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, std::f64::consts::TAU, true);
        let a1 = s.add_arc(vect2d::new(2.0, 0.0), 1.5, 0.0, std::f64::consts::TAU, true);
        (s, a0, a1)
    }

    // ---- Self-referential checks ----

    #[test]
    fn parallel_self() {
        let (s, l0, _) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn perpendicular_self() {
        let (s, l0, _) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn equal_length_self() {
        let (s, l0, _) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyEqualLength { a: l0, b: l0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn concentric_self() {
        let (s, a0, _) = two_arcs();
        let r = check_constraint_conflict(&s, &Action::ApplyConcentric { a: a0, b: a0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn equal_radius_self() {
        let (s, a0, _) = two_arcs();
        let r = check_constraint_conflict(&s, &Action::ApplyEqualRadius { a: a0, b: a0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn tangent_aa_self() {
        let (s, a0, _) = two_arcs();
        let r = check_constraint_conflict(&s, &Action::ApplyTangentAA { a: a0, b: a0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    #[test]
    fn coincident_ll21_self() {
        let (s, l0, _) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyCoincidentLL21 { a: l0, b: l0 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("to itself"));
    }

    // ---- Direct H/V conflict ----

    #[test]
    fn horizontal_on_vertical_line() {
        let (mut s, l0, _) = two_lines();
        s.lines[l0].constraints.vertical = true;
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l0] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already vertical"));
    }

    #[test]
    fn vertical_on_horizontal_line() {
        let (mut s, l0, _) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        let r = check_constraint_conflict(&s, &Action::ApplyVertical { lines: vec![l0] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already horizontal"));
    }

    #[test]
    fn horizontal_on_already_horizontal() {
        let (mut s, l0, _) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l0] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already horizontal"));
    }

    #[test]
    fn vertical_on_already_vertical() {
        let (mut s, l0, _) = two_lines();
        s.lines[l0].constraints.vertical = true;
        let r = check_constraint_conflict(&s, &Action::ApplyVertical { lines: vec![l0] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already vertical"));
    }

    // ---- Duplicate constraint checks ----

    #[test]
    fn duplicate_parallel() {
        let (mut s, l0, l1) = two_lines();
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already parallel"));
    }

    #[test]
    fn duplicate_parallel_reversed() {
        let (mut s, l0, l1) = two_lines();
        s.parallel.push(Parallel { a: l1, b: l0, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already parallel"));
    }

    #[test]
    fn duplicate_perpendicular() {
        let (mut s, l0, l1) = two_lines();
        s.perpendicular.push(Perpendicular { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already perpendicular"));
    }

    #[test]
    fn duplicate_equal_length() {
        let (mut s, l0, l1) = two_lines();
        s.equal_length.push(EqualLength { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyEqualLength { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already have equal length"));
    }

    #[test]
    fn duplicate_concentric() {
        let (mut s, a0, a1) = two_arcs();
        s.concentric.push(Concentric { a: a0, b: a1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyConcentric { a: a0, b: a1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already concentric"));
    }

    #[test]
    fn duplicate_equal_radius() {
        let (mut s, a0, a1) = two_arcs();
        s.equal_radius.push(EqualRadius { a: a0, b: a1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyEqualRadius { a: a0, b: a1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already have equal radius"));
    }

    #[test]
    fn duplicate_tangent_aa() {
        let (mut s, a0, a1) = two_arcs();
        s.tangent_aa.push(TangentAA { a: a0, b: a1, shared: SharedEndpoint::None, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyTangentAA { a: a0, b: a1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already tangent"));
    }

    // ---- Direct pair conflicts (parallel vs perpendicular) ----

    #[test]
    fn parallel_on_perpendicular_pair() {
        let (mut s, l0, l1) = two_lines();
        s.perpendicular.push(Perpendicular { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already perpendicular"));
    }

    #[test]
    fn perpendicular_on_parallel_pair() {
        let (mut s, l0, l1) = two_lines();
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already parallel"));
    }

    // ---- Transitive orientation conflicts ----

    #[test]
    fn horizontal_on_line_in_vertical_parallel_group() {
        // L0 is vertical, L1 is parallel to L0.
        // Making L1 horizontal should conflict.
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.vertical = true;
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l1] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("vertical"));
    }

    #[test]
    fn vertical_on_line_in_horizontal_parallel_group() {
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyVertical { lines: vec![l1] });
        assert!(r.is_some());
        assert!(r.unwrap().contains("horizontal"));
    }

    #[test]
    fn horizontal_via_perpendicular_propagation() {
        // L0 is horizontal, L1 is perpendicular to L0 (so L1 is implicitly vertical).
        // L2 is in the same group as L1 (parallel). Making L2 horizontal should conflict.
        let (mut s, l0, l1, l2) = three_lines();
        s.lines[l0].constraints.horizontal = true;
        s.perpendicular.push(Perpendicular { a: l0, b: l1, hb: CrossBlock::new() });
        // L1 should be implicitly vertical now. Make L2 parallel to L1.
        s.parallel.push(Parallel { a: l1, b: l2, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l2] });
        assert!(r.is_some());
    }

    #[test]
    fn parallel_merging_conflicting_orientations() {
        // L0 is horizontal, L1 is vertical.
        // Making them parallel should conflict.
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        s.lines[l1].constraints.vertical = true;
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_some());
        let msg = r.unwrap();
        assert!(msg.contains("horizontal") || msg.contains("vertical"));
    }

    #[test]
    fn transitive_parallel_detected() {
        // L0 parallel to L1, L1 parallel to L2 (via two constraints).
        // Adding L0 parallel to L2 is redundant (transitively already parallel).
        let (mut s, l0, l1, l2) = three_lines();
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        s.parallel.push(Parallel { a: l1, b: l2, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l2 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("already parallel"));
    }

    #[test]
    fn perpendicular_in_same_parallel_group() {
        // L0 parallel to L1. Making them perpendicular should conflict.
        let (mut s, l0, l1) = two_lines();
        s.parallel.push(Parallel { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("parallel"));
    }

    #[test]
    fn perpendicular_both_horizontal() {
        // L0 and L1 both horizontal. Making them perpendicular should conflict.
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        s.lines[l1].constraints.horizontal = true;
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("horizontal"));
    }

    #[test]
    fn perpendicular_both_vertical() {
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.vertical = true;
        s.lines[l1].constraints.vertical = true;
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("vertical"));
    }

    // ---- No conflict cases ----

    #[test]
    fn valid_horizontal() {
        let (s, l0, _) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l0] });
        assert!(r.is_none());
    }

    #[test]
    fn valid_parallel() {
        let (s, l0, l1) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_none());
    }

    #[test]
    fn valid_perpendicular() {
        let (s, l0, l1) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_none());
    }

    #[test]
    fn valid_perpendicular_h_and_v() {
        // H line perpendicular to V line is fine.
        let (mut s, l0, l1) = two_lines();
        s.lines[l0].constraints.horizontal = true;
        s.lines[l1].constraints.vertical = true;
        let r = check_constraint_conflict(&s, &Action::ApplyPerpendicular { a: l0, b: l1 });
        assert!(r.is_none());
    }

    #[test]
    fn valid_equal_length() {
        let (s, l0, l1) = two_lines();
        let r = check_constraint_conflict(&s, &Action::ApplyEqualLength { a: l0, b: l1 });
        assert!(r.is_none());
    }

    #[test]
    fn valid_concentric() {
        let (s, a0, a1) = two_arcs();
        let r = check_constraint_conflict(&s, &Action::ApplyConcentric { a: a0, b: a1 });
        assert!(r.is_none());
    }

    #[test]
    fn non_constraint_action_no_conflict() {
        let s = Sketch::new();
        let r = check_constraint_conflict(&s, &Action::AddLine {
            p1: vect2d::new(0.0, 0.0),
            p2: vect2d::new(1.0, 1.0),
        });
        assert!(r.is_none());
    }

    #[test]
    fn horizontal_perp_propagation_no_false_positive() {
        // L0 is horizontal, L1 is perpendicular to L0 (implicitly vertical).
        // Making L2 horizontal is fine since L2 is not in any related group.
        let (mut s, l0, l1, l2) = three_lines();
        s.lines[l0].constraints.horizontal = true;
        s.perpendicular.push(Perpendicular { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyHorizontal { lines: vec![l2] });
        assert!(r.is_none());
    }

    #[test]
    fn parallel_perp_linked_conflict() {
        // L0 perp L1. Making them parallel should conflict.
        let (mut s, l0, l1) = two_lines();
        s.perpendicular.push(Perpendicular { a: l0, b: l1, hb: CrossBlock::new() });
        let r = check_constraint_conflict(&s, &Action::ApplyParallel { a: l0, b: l1 });
        assert!(r.is_some());
        assert!(r.unwrap().contains("perpendicular"));
    }
}

//! Sequences of lines and arcs connected end to end: the walk from one
//! entity until an end or a branch (the offset tool's double-click), and
//! the ordering of an explicit set into one sequence.
//!
//! Connection is coincidence of endpoints -- direct line-line, line-arc
//! and arc-arc endpoint coincidences and any chain of them through shared
//! points. Arc centers, points on bodies and construction entities do not
//! connect, and standalone points never count as branches.

use std::collections::HashMap;

use arael::refs::Ref;
use arael_sketch_solver::*;

use crate::coincide::CoincidenceGroups;

/// An ordered sequence: each segment with the direction the chain
/// traverses it, `closed` when the last joins back to the first.
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    pub segs: Vec<OffsetSource>,
    pub closed: bool,
}

impl Sequence {
    pub fn entities(&self) -> impl Iterator<Item = OffsetEntity> + '_ {
        self.segs.iter().map(|s| s.entity)
    }
}

/// Name of a line or arc.
pub fn entity_name(sketch: &Sketch, e: OffsetEntity) -> String {
    match e {
        OffsetEntity::Line(l) => sketch.lines[l].name.clone(),
        OffsetEntity::Arc(a) => sketch.arcs[a].name.clone(),
    }
}

/// Whether the entity is construction geometry.
pub fn is_construction(sketch: &Sketch, e: OffsetEntity) -> bool {
    match e {
        OffsetEntity::Line(l) => sketch.lines[l].construction,
        OffsetEntity::Arc(a) => sketch.arcs[a].construction,
    }
}

/// A full circle or ellipse: a sequence on its own.
fn is_closed_entity(sketch: &Sketch, e: OffsetEntity) -> bool {
    match e {
        OffsetEntity::Line(_) => false,
        OffsetEntity::Arc(a) => sketch.arcs[a].closed,
    }
}

/// Endpoint connectivity over the entities `allow` admits.
struct Ends {
    groups: CoincidenceGroups,
    /// group root -> the (entity, is_end) endpoints in it
    members: HashMap<usize, Vec<(OffsetEntity, bool)>>,
}

impl Ends {
    fn build(sketch: &Sketch, allow: &dyn Fn(OffsetEntity) -> bool) -> Self {
        let mut groups = CoincidenceGroups::build(sketch);
        let mut members: HashMap<usize, Vec<(OffsetEntity, bool)>> = HashMap::new();
        let mut add = |groups: &mut CoincidenceGroups, e: OffsetEntity, is_end: bool, slot: usize| {
            let root = groups.find(slot);
            members.entry(root).or_default().push((e, is_end));
        };
        for r in sketch.lines.refs() {
            let e = OffsetEntity::Line(r);
            if !allow(e) { continue; }
            let (s1, s2) = (groups.lp1(r), groups.lp2(r));
            add(&mut groups, e, false, s1);
            add(&mut groups, e, true, s2);
        }
        for r in sketch.arcs.refs() {
            let e = OffsetEntity::Arc(r);
            if !allow(e) || sketch.arcs[r].closed { continue; }
            let (s1, s2) = (groups.arc_start(r), groups.arc_end(r));
            add(&mut groups, e, false, s1);
            add(&mut groups, e, true, s2);
        }
        Ends { groups, members }
    }

    fn slot(&self, e: OffsetEntity, is_end: bool) -> usize {
        match (e, is_end) {
            (OffsetEntity::Line(l), false) => self.groups.lp1(l),
            (OffsetEntity::Line(l), true) => self.groups.lp2(l),
            (OffsetEntity::Arc(a), false) => self.groups.arc_start(a),
            (OffsetEntity::Arc(a), true) => self.groups.arc_end(a),
        }
    }

    /// The other entities' endpoints connected to this one.
    fn neighbours(&mut self, e: OffsetEntity, is_end: bool) -> Vec<(OffsetEntity, bool)> {
        let slot = self.slot(e, is_end);
        let root = self.groups.find(slot);
        self.members
            .get(&root)
            .map(|v| v.iter().copied().filter(|(o, _)| *o != e).collect())
            .unwrap_or_default()
    }
}

/// Where a walk step ends.
enum Step {
    /// Nothing connected: a free end.
    End,
    /// Two or more connected: stop before the branch.
    Branch,
    /// Exactly one: the next entity, entered at its end (`true`) or start.
    Next(OffsetEntity, bool),
}

fn step(ends: &mut Ends, e: OffsetEntity, is_end: bool) -> Step {
    let n = ends.neighbours(e, is_end);
    match n.len() {
        0 => Step::End,
        1 => Step::Next(n[0].0, n[0].1),
        _ => Step::Branch,
    }
}

/// Walk both ways from `seed` over the entities `allow` admits, until an
/// end, a branch, or back to the seed. Chain direction is the seed's own
/// (p1 -> p2, start -> end).
fn walk_within(sketch: &Sketch, seed: OffsetEntity, allow: &dyn Fn(OffsetEntity) -> bool) -> Sequence {
    if is_closed_entity(sketch, seed) {
        return Sequence { segs: vec![OffsetSource { entity: seed, reversed: false }], closed: true };
    }
    let mut ends = Ends::build(sketch, allow);
    let mut segs = vec![OffsetSource { entity: seed, reversed: false }];
    let mut closed = false;

    // Forward: leave the last segment at its exit end.
    let (mut cur, mut cur_rev) = (seed, false);
    loop {
        let exit_is_end = !cur_rev;
        match step(&mut ends, cur, exit_is_end) {
            Step::End | Step::Branch => break,
            Step::Next(next, entered_at_end) => {
                if next == seed {
                    closed = true;
                    break;
                }
                if segs.iter().any(|s| s.entity == next) {
                    break;
                }
                // Entered at its end: traversed end -> start.
                let rev = entered_at_end;
                segs.push(OffsetSource { entity: next, reversed: rev });
                cur = next;
                cur_rev = rev;
            }
        }
    }
    if closed {
        return Sequence { segs, closed };
    }

    // Backward: leave the first segment at its entry end.
    let (mut cur, mut cur_rev) = (seed, false);
    loop {
        let entry_is_end = cur_rev;
        match step(&mut ends, cur, entry_is_end) {
            Step::End | Step::Branch => break,
            Step::Next(prev, met_at_end) => {
                if prev == seed || segs.iter().any(|s| s.entity == prev) {
                    break;
                }
                // Met at its end going backwards: forwards it exits there,
                // so it is traversed start -> end.
                let rev = !met_at_end;
                segs.insert(0, OffsetSource { entity: prev, reversed: rev });
                cur = prev;
                cur_rev = rev;
            }
        }
    }
    Sequence { segs, closed: false }
}

/// The sequence through `seed`: both ways until an end or a branch,
/// over the non-construction entities (the seed itself always counts).
pub fn walk(sketch: &Sketch, seed: OffsetEntity) -> Sequence {
    walk_within(sketch, seed, &|e| e == seed || !is_construction(sketch, e))
}

/// Order an explicit set into one sequence, or say why it is not one.
pub fn order(sketch: &Sketch, set: &[OffsetEntity]) -> Result<Sequence, String> {
    if set.is_empty() {
        return Err("nothing to offset".into());
    }
    let in_set = |e: OffsetEntity| set.contains(&e);
    if set.len() == 1 {
        return Ok(walk_within(sketch, set[0], &in_set));
    }
    let mut ends = Ends::build(sketch, &in_set);
    // Degree per endpoint within the set; a branch is a degree above one.
    let mut open_start: Option<OffsetEntity> = None;
    for &e in set {
        if is_closed_entity(sketch, e) {
            return Err(format!("{} is a closed curve and cannot join a sequence", entity_name(sketch, e)));
        }
        for is_end in [false, true] {
            let n = ends.neighbours(e, is_end).len();
            if n > 1 {
                return Err(format!("the selection branches at {}", entity_name(sketch, e)));
            }
            if n == 0 && open_start.is_none() {
                open_start = Some(e);
            }
        }
    }
    // Start at an open end when there is one (so the walk covers the
    // whole path), else anywhere on the cycle.
    let seed = open_start.unwrap_or(set[0]);
    let seq = walk_within(sketch, seed, &in_set);
    if seq.segs.len() != set.len() {
        let missing: Vec<String> = set
            .iter()
            .filter(|e| !seq.segs.iter().any(|s| s.entity == **e))
            .map(|e| entity_name(sketch, *e))
            .collect();
        return Err(format!(
            "the selection is not one connected sequence: {} not connected",
            missing.join(", ")
        ));
    }
    // A seed with a free end walked from that end; make the chain start
    // there in the forward direction (the walk may have prepended).
    Ok(seq)
}

/// Resolve a line or arc reference.
pub fn entity_of_line(r: Ref<Line>) -> OffsetEntity {
    OffsetEntity::Line(r)
}
pub fn entity_of_arc(r: Ref<Arc>) -> OffsetEntity {
    OffsetEntity::Arc(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::CommandContext;

    fn ctx(script: &str) -> CommandContext {
        let mut ctx = CommandContext::new();
        for r in crate::commands::execute(&mut ctx, script) {
            assert!(!r.is_error, "{}", r.output);
        }
        ctx
    }

    fn names(s: &Sketch, seq: &Sequence) -> Vec<String> {
        seq.segs
            .iter()
            .map(|x| format!("{}{}", entity_name(s, x.entity), if x.reversed { "'" } else { "" }))
            .collect()
    }

    fn line(s: &Sketch, name: &str) -> OffsetEntity {
        OffsetEntity::Line(crate::commands::resolve_line(s, name).unwrap())
    }

    #[test]
    fn walk_follows_a_polyline_both_ways() {
        // L0: (0,0)-(2,0), L1: (2,0)-(2,2), L2: (2,2)-(0,2): open path.
        let c = ctx("add_line 0,0 2,0 2,2 0,2");
        let seq = walk(&c.sketch, line(&c.sketch, "L1"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1", "L2"]);
        assert!(!seq.closed);
        // Seeded from the last line, same chain, same direction.
        let seq = walk(&c.sketch, line(&c.sketch, "L2"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1", "L2"]);
    }

    #[test]
    fn walk_marks_reversed_segments() {
        // L1 drawn backwards: (2,2)->(2,0) meets L0's end at (2,0).
        let c = ctx("add_line 0,0 2,0; add_line 2,2 2,0");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1'"]);
        // Seeded from L1 the chain runs L1 then L0 reversed.
        let seq = walk(&c.sketch, line(&c.sketch, "L1"));
        assert_eq!(names(&c.sketch, &seq), ["L1", "L0'"]);
    }

    #[test]
    fn walk_stops_at_a_branch_and_closes_a_loop() {
        // A rectangle with a line hanging off one corner.
        // Both walks stop at the branch corner: all four sides, not a loop.
        let c = ctx("add_rect 0,0 4,3; add_line 4,3 6,5");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert!(!seq.closed, "the hanging line branches the loop");
        assert_eq!(names(&c.sketch, &seq), ["L2", "L3", "L0", "L1"]);
        // A branch in the middle: the walk stops before it on both sides.
        let c = ctx("add_line 0,0 2,0 4,0 6,0; add_line 4,0 4,3");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1"]);
        let c = ctx("add_rect 0,0 4,3");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert!(seq.closed);
        assert_eq!(seq.segs.len(), 4);
    }

    #[test]
    fn walk_ignores_construction_and_body_points() {
        let c = ctx("add_line 0,0 2,0 4,0; add_line 2,0 2,3; constr L2 on");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1"], "a construction line is not a branch");
        let c = ctx("add_line 0,0 2,0 4,0; add_point 2,0; point_on P0 L0");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "L1"]);
    }

    #[test]
    fn walk_crosses_a_shared_point_and_an_arc() {
        // Line to a point, arc from that point: connected through the
        // point the auto-coincident ties both to (or directly; either
        // way one group).
        let c = ctx("add_line 0,0 2,0; add_point 2,0; add_arc 2,0 4,2 3.4142,0.5858");
        let seq = walk(&c.sketch, line(&c.sketch, "L0"));
        assert_eq!(names(&c.sketch, &seq), ["L0", "A0"]);
        let c = ctx("add_circle 0,0 1");
        let seq = walk(&c.sketch, OffsetEntity::Arc(crate::commands::resolve_arc(&c.sketch, "A0").unwrap()));
        assert!(seq.closed && seq.segs.len() == 1);
    }

    #[test]
    fn order_validates_a_set() {
        let c = ctx("add_line 0,0 2,0 2,2 0,2; add_line 5,5 6,6");
        let s = &c.sketch;
        let seq = order(s, &[line(s, "L2"), line(s, "L0"), line(s, "L1")]).unwrap();
        assert_eq!(names(s, &seq), ["L0", "L1", "L2"]);
        let e = order(s, &[line(s, "L0"), line(s, "L3")]).unwrap_err();
        assert!(e.contains("not one connected sequence") && e.contains("L3"), "{}", e);
        let e = order(s, &[line(s, "L0"), line(s, "L2")]).unwrap_err();
        assert!(e.contains("not connected"), "{}", e);
        let c = ctx("add_rect 0,0 4,3; add_line 4,3 6,5");
        let s = &c.sketch;
        let e = order(s, &[line(s, "L0"), line(s, "L1"), line(s, "L2"), line(s, "L4")]).unwrap_err();
        assert!(e.contains("branches"), "{}", e);
        let seq = order(s, &[line(s, "L0"), line(s, "L1"), line(s, "L2"), line(s, "L3")]).unwrap();
        assert!(seq.closed);
    }
}

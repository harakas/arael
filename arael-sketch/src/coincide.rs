// Transitive-coincidence groups over the sketch's endpoint slots.
// One union-find covers points, line endpoints, and arc
// center/start/end; every coincident_* collection plus concentric
// contributes a union. Shared by locked-set computation, drag snap
// filtering, and group unlock.

use arael::refs::Ref;
use arael_sketch_solver::*;
use arael_sketch_backend::Selection;

pub struct CoincidenceGroups {
    np: usize,
    nl: usize,
    na: usize,
    parent: Vec<usize>,
}

impl CoincidenceGroups {
    /// Build the groups from every coincidence collection in the
    /// sketch (including concentric, which couples arc centers).
    pub fn build(sketch: &Sketch) -> Self {
        let np = sketch.points.slot_count();
        let nl = sketch.lines.slot_count();
        let na = sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut g = CoincidenceGroups { np, nl, na, parent: (0..total).collect() };

        // Point-Point, Line-Point, Line-Line
        for c in &sketch.coincident_pp { g.union(g.pt(c.a), g.pt(c.b)); }
        for c in &sketch.coincident_lp1 { g.union(g.lp1(c.line), g.pt(c.point)); }
        for c in &sketch.coincident_lp2 { g.union(g.lp2(c.line), g.pt(c.point)); }
        for c in &sketch.coincident_ll11 { g.union(g.lp1(c.a), g.lp1(c.b)); }
        for c in &sketch.coincident_ll12 { g.union(g.lp1(c.a), g.lp2(c.b)); }
        for c in &sketch.coincident_ll21 { g.union(g.lp2(c.a), g.lp1(c.b)); }
        for c in &sketch.coincident_ll22 { g.union(g.lp2(c.a), g.lp2(c.b)); }
        // Point-Arc
        for c in &sketch.coincident_arc_center { g.union(g.pt(c.point), g.arc_center(c.arc)); }
        for c in &sketch.coincident_arc_start { g.union(g.pt(c.point), g.arc_start(c.arc)); }
        for c in &sketch.coincident_arc_end { g.union(g.pt(c.point), g.arc_end(c.arc)); }
        // Line-Arc
        for c in &sketch.coincident_lp1_arc_center { g.union(g.lp1(c.line), g.arc_center(c.arc)); }
        for c in &sketch.coincident_lp2_arc_center { g.union(g.lp2(c.line), g.arc_center(c.arc)); }
        for c in &sketch.coincident_lp1_arc_start { g.union(g.lp1(c.line), g.arc_start(c.arc)); }
        for c in &sketch.coincident_lp2_arc_start { g.union(g.lp2(c.line), g.arc_start(c.arc)); }
        for c in &sketch.coincident_lp1_arc_end { g.union(g.lp1(c.line), g.arc_end(c.arc)); }
        for c in &sketch.coincident_lp2_arc_end { g.union(g.lp2(c.line), g.arc_end(c.arc)); }
        // Arc-Arc
        for c in &sketch.concentric { g.union(g.arc_center(c.a), g.arc_center(c.b)); }
        for c in &sketch.coincident_arc_center_start { g.union(g.arc_center(c.a), g.arc_start(c.b)); }
        for c in &sketch.coincident_arc_center_end { g.union(g.arc_center(c.a), g.arc_end(c.b)); }
        for c in &sketch.coincident_arc_start_center { g.union(g.arc_start(c.a), g.arc_center(c.b)); }
        for c in &sketch.coincident_arc_end_center { g.union(g.arc_end(c.a), g.arc_center(c.b)); }
        for c in &sketch.coincident_arc_start_start { g.union(g.arc_start(c.a), g.arc_start(c.b)); }
        for c in &sketch.coincident_arc_start_end { g.union(g.arc_start(c.a), g.arc_end(c.b)); }
        for c in &sketch.coincident_arc_end_start { g.union(g.arc_end(c.a), g.arc_start(c.b)); }
        for c in &sketch.coincident_arc_end_end { g.union(g.arc_end(c.a), g.arc_end(c.b)); }
        g
    }

    pub fn pt(&self, r: Ref<Point>) -> usize { r.index() as usize }
    pub fn lp1(&self, r: Ref<Line>) -> usize { self.np + r.index() as usize }
    pub fn lp2(&self, r: Ref<Line>) -> usize { self.np + self.nl + r.index() as usize }
    pub fn arc_center(&self, r: Ref<Arc>) -> usize { self.np + 2 * self.nl + r.index() as usize }
    pub fn arc_start(&self, r: Ref<Arc>) -> usize { self.np + 2 * self.nl + self.na + r.index() as usize }
    pub fn arc_end(&self, r: Ref<Arc>) -> usize { self.np + 2 * self.nl + 2 * self.na + r.index() as usize }

    /// Slot id for a point-like selection; None for whole entities.
    pub fn selection_id(&self, s: Selection) -> Option<usize> {
        match s {
            Selection::Point(r) => Some(self.pt(r)),
            Selection::LineP1(r) => Some(self.lp1(r)),
            Selection::LineP2(r) => Some(self.lp2(r)),
            Selection::ArcCenter(r) => Some(self.arc_center(r)),
            Selection::ArcStart(r) => Some(self.arc_start(r)),
            Selection::ArcEnd(r) => Some(self.arc_end(r)),
            _ => None,
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb { self.parent[ra] = rb; }
    }

    pub fn same_group(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }
}

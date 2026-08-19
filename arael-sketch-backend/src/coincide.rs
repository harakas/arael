//! Transitive-coincidence groups over the sketch's endpoint slots.
//! One union-find covers points, line endpoints, and arc
//! center/start/end; every coincident_* collection plus concentric
//! contributes a union. Shared by the GUI's locked-set computation,
//! drag snap filtering and group unlock, and by the offset engine's
//! sequence walk.

use arael::refs::Ref;
use arael_sketch_solver::*;
use crate::Selection;

pub struct CoincidenceGroups {
    np: usize,
    nl: usize,
    na: usize,
    parent: Vec<usize>,
}

impl CoincidenceGroups {
    /// Build the groups from every coincidence collection in the
    /// sketch, via the registry's canonical endpoint pairs -- a new
    /// coincidence collection participates without this file
    /// changing. Concentric couples arc centers without being a
    /// coincidence collection; it is the one explicit addition.
    pub fn build(sketch: &Sketch) -> Self {
        let np = sketch.points.slot_count();
        let nl = sketch.lines.slot_count();
        let na = sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut g = CoincidenceGroups { np, nl, na, parent: (0..total).collect() };

        sketch.for_each_coincidence_pair(|a, b| {
            let sa = g.slot_of(a);
            let sb = g.slot_of(b);
            g.union(sa, sb);
        });
        for c in &sketch.concentric { g.union(g.arc_center(c.a), g.arc_center(c.b)); }
        g
    }

    /// Union-find slot for a canonical endpoint id.
    fn slot_of(&self, enc: u64) -> usize {
        let (role, idx) = decode_endpoint(enc);
        let idx = idx as usize;
        match role {
            EndpointRole::Point => idx,
            EndpointRole::LineP1 => self.np + idx,
            EndpointRole::LineP2 => self.np + self.nl + idx,
            EndpointRole::ArcCenter => self.np + 2 * self.nl + idx,
            EndpointRole::ArcStart => self.np + 2 * self.nl + self.na + idx,
            EndpointRole::ArcEnd => self.np + 2 * self.nl + 2 * self.na + idx,
        }
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

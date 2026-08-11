//! The constraint registry: one interface over the 112 constraint
//! collections, one enumeration point.
//!
//! Every constraint struct implements [`SketchConstraint`]; every
//! `Vec<C>` of them is usable as a `&mut dyn ConstraintCollection`;
//! [`Sketch::for_each_constraint_collection`] hands each collection to
//! a plain closure together with the entity arenas and the collection's
//! metadata. The walker destructures `Sketch` exhaustively -- adding a
//! collection field without registering it here is a compile error, so
//! the historical failure mode (a new collection missing from some of
//! the hand-maintained delete/cleanup/naming/listing walks) cannot
//! recur.

use crate::{Arc, Line, Point, Sketch};
use arael::refs::{Arena, Ref};

/// Per-collection metadata, defined at the registration site.
pub struct CollectionMeta {
    /// Field name of the collection on `Sketch`.
    pub name: &'static str,
    /// Coincidence constraints act as helper-point bridges; every
    /// other constraint referencing a helper point gives it a purpose.
    pub coincidence: bool,
    /// Backed by a dimension (distance / axis-distance / angle
    /// families): deleted through the dimension, so the constraint is
    /// not addressable by its C<n> name.
    pub dimension_backed: bool,
}

/// Identity key for duplicate removal. Values (distances, angles) are
/// deliberately not part of any key: two constraints on the same
/// referents are duplicates even when their targets disagree.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DedupKey {
    /// Order-free coincidence of two canonical endpoints. Deduped in
    /// one set shared across every coincidence collection, so the
    /// same endpoint pair expressed through different collections
    /// (e.g. arc start-center vs center-start) still collides.
    Coincidence(u64, u64),
    /// Collection-local identity: packed refs, normalized per type
    /// (sorted for symmetric pairs), padded with u64::MAX.
    Local(u64, u64, u64),
}

// Canonical endpoint ids for DedupKey::Coincidence: arena index in the
// high bits, endpoint role in the low bits. Roles are disjoint across
// entity kinds, so the encoding is collision-free.
pub fn ep_point(r: Ref<Point>) -> u64 { (r.index() as u64) << 3 }
pub fn ep_line_p1(r: Ref<Line>) -> u64 { (r.index() as u64) << 3 | 1 }
pub fn ep_line_p2(r: Ref<Line>) -> u64 { (r.index() as u64) << 3 | 2 }
pub fn ep_arc_center(r: Ref<Arc>) -> u64 { (r.index() as u64) << 3 | 3 }
pub fn ep_arc_start(r: Ref<Arc>) -> u64 { (r.index() as u64) << 3 | 4 }
pub fn ep_arc_end(r: Ref<Arc>) -> u64 { (r.index() as u64) << 3 | 5 }

pub fn coincidence_key(a: u64, b: u64) -> DedupKey {
    DedupKey::Coincidence(a.min(b), a.max(b))
}

/// The interface every constraint struct implements.
pub trait SketchConstraint {
    fn nid(&self) -> u32;
    fn set_nid(&mut self, nid: u32);
    fn cid(&self) -> u32;
    fn references_point(&self, r: Ref<Point>) -> bool;
    fn references_line(&self, r: Ref<Line>) -> bool;
    fn references_arc(&self, r: Ref<Arc>) -> bool;
    /// Visit every point reference.
    fn each_point_ref(&self, f: &mut dyn FnMut(Ref<Point>));
    /// Visit every line reference.
    fn each_line_ref(&self, f: &mut dyn FnMut(Ref<Line>));
    /// Visit every arc reference.
    fn each_arc_ref(&self, f: &mut dyn FnMut(Ref<Arc>));
    /// Rewrite point references `old -> new` (helper consolidation).
    fn remap_point(&mut self, old: Ref<Point>, new: Ref<Point>);
    /// Identity for duplicate removal (see [`DedupKey`]).
    fn dedup_key(&self) -> DedupKey;
    /// Human-readable description, as shown by `list constraints`
    /// (without the `C<n>: ` prefix -- the walker adds it).
    fn describe(&self, s: &Sketch) -> String;
}

/// A constraint collection, type-erased for the registry walkers.
pub trait ConstraintCollection {
    fn len(&self) -> usize;
    fn item(&self, i: usize) -> &dyn SketchConstraint;
    fn item_mut(&mut self, i: usize) -> &mut dyn SketchConstraint;
    fn retain_constraints(&mut self, f: &mut dyn FnMut(&dyn SketchConstraint) -> bool);
}

impl<C: SketchConstraint> ConstraintCollection for Vec<C> {
    fn len(&self) -> usize {
        self.len()
    }
    fn item(&self, i: usize) -> &dyn SketchConstraint {
        &self[i]
    }
    fn item_mut(&mut self, i: usize) -> &mut dyn SketchConstraint {
        &mut self[i]
    }
    fn retain_constraints(&mut self, f: &mut dyn FnMut(&dyn SketchConstraint) -> bool) {
        self.retain(|c| f(c));
    }
}

/// Read-only view of the entity arenas, available inside the walkers.
pub struct ConstraintArenas<'a> {
    pub points: &'a Arena<Point>,
    pub lines: &'a Arena<Line>,
    pub arcs: &'a Arena<Arc>,
}

// Key builder for the `dedup(...)` spec in sketch_constraint!.
macro_rules! dedup_key_expr {
    // Exact identity: the named ref fields, in order.
    ($c:ident, exact($f0:ident)) => {
        DedupKey::Local($c.$f0.index() as u64, u64::MAX, u64::MAX)
    };
    ($c:ident, exact($f0:ident, $f1:ident)) => {
        DedupKey::Local($c.$f0.index() as u64, $c.$f1.index() as u64, u64::MAX)
    };
    ($c:ident, exact($f0:ident, $f1:ident, $f2:ident)) => {
        DedupKey::Local($c.$f0.index() as u64, $c.$f1.index() as u64, $c.$f2.index() as u64)
    };
    // Axis distance: the horizontal flag distinguishes hdistance from
    // vdistance on the same referents -- it is identity, not a value.
    ($c:ident, axis($f0:ident, $f1:ident; $h:ident)) => {
        DedupKey::Local($c.$f0.index() as u64, $c.$f1.index() as u64, $c.$h as u64)
    };
    // Symmetric pair within the collection (parallel, equal, ...).
    ($c:ident, sorted($fa:ident, $fb:ident)) => {{
        let a = $c.$fa.index() as u64;
        let b = $c.$fb.index() as u64;
        DedupKey::Local(a.min(b), a.max(b), u64::MAX)
    }};
    // Symmetry: the two sides are swappable, the mirror is not.
    ($c:ident, mirror($fa:ident, $fb:ident; $m:ident)) => {{
        let a = $c.$fa.index() as u64;
        let b = $c.$fb.index() as u64;
        DedupKey::Local(a.min(b), a.max(b), $c.$m.index() as u64)
    }};
    // Coincidence of two canonical endpoints; the role token is the
    // ep_* encoder function.
    ($c:ident, coincide($ra:ident($fa:ident), $rb:ident($fb:ident))) => {
        coincidence_key($ra($c.$fa), $rb($c.$fb))
    };
}

// Argument forms for the `describe(...)` spec in sketch_constraint!.
macro_rules! describe_arg {
    ($s:ident, $c:ident, line($f:ident)) => { &$s.lines[$c.$f].name };
    ($s:ident, $c:ident, arc($f:ident)) => { &$s.arcs[$c.$f].name };
    // Points resolve helpers to the endpoint they are bridged to.
    ($s:ident, $c:ident, point($f:ident)) => { $s.point_display_name($c.$f) };
    ($s:ident, $c:ident, num($f:ident)) => { $c.$f.abs() };
    ($s:ident, $c:ident, deg($f:ident)) => { $c.$f.to_degrees() };
    ($s:ident, $c:ident, axis($f:ident)) => {
        if $c.$f { "hdistance" } else { "vdistance" }
    };
}

macro_rules! sketch_constraint {
    ($ty:ident, points($($p:ident),*), lines($($l:ident),*), arcs($($a:ident),*),
     dedup($($dk:tt)*), describe($fmt:literal $(, $ar:ident($af:ident))*)) => {
        impl SketchConstraint for crate::$ty {
            fn nid(&self) -> u32 { self.nid }
            fn set_nid(&mut self, nid: u32) { self.nid = nid; }
            fn cid(&self) -> u32 { self.cid }
            fn references_point(&self, r: Ref<Point>) -> bool {
                let _ = &r;
                false $(|| self.$p == r)*
            }
            fn references_line(&self, r: Ref<Line>) -> bool {
                let _ = &r;
                false $(|| self.$l == r)*
            }
            fn references_arc(&self, r: Ref<Arc>) -> bool {
                let _ = &r;
                false $(|| self.$a == r)*
            }
            fn each_point_ref(&self, f: &mut dyn FnMut(Ref<Point>)) {
                let _ = &f;
                $(f(self.$p);)*
            }
            fn each_line_ref(&self, f: &mut dyn FnMut(Ref<Line>)) {
                let _ = &f;
                $(f(self.$l);)*
            }
            fn each_arc_ref(&self, f: &mut dyn FnMut(Ref<Arc>)) {
                let _ = &f;
                $(f(self.$a);)*
            }
            fn remap_point(&mut self, old: Ref<Point>, new: Ref<Point>) {
                let _ = (&old, &new);
                $(if self.$p == old { self.$p = new; })*
            }
            fn dedup_key(&self) -> DedupKey {
                let c = self;
                dedup_key_expr!(c, $($dk)*)
            }
            fn describe(&self, s: &Sketch) -> String {
                let _ = &s;
                let c = self;
                format!($fmt $(, describe_arg!(s, c, $ar($af)))*)
            }
        }
    };
}

sketch_constraint!(CoincidentPP, points(a, b), lines(), arcs(),
    dedup(coincide(ep_point(a), ep_point(b))),
    describe("coincident {} {}", point(a), point(b)));
sketch_constraint!(CoincidentLP1, points(point), lines(line), arcs(),
    dedup(coincide(ep_line_p1(line), ep_point(point))),
    describe("coincident {}.p1 {}", line(line), point(point)));
sketch_constraint!(CoincidentLP2, points(point), lines(line), arcs(),
    dedup(coincide(ep_line_p2(line), ep_point(point))),
    describe("coincident {}.p2 {}", line(line), point(point)));
sketch_constraint!(CoincidentLL11, points(), lines(a, b), arcs(),
    dedup(coincide(ep_line_p1(a), ep_line_p1(b))),
    describe("coincident {}.p1 {}.p1", line(a), line(b)));
sketch_constraint!(CoincidentLL12, points(), lines(a, b), arcs(),
    dedup(coincide(ep_line_p1(a), ep_line_p2(b))),
    describe("coincident {}.p1 {}.p2", line(a), line(b)));
sketch_constraint!(CoincidentLL21, points(), lines(a, b), arcs(),
    dedup(coincide(ep_line_p2(a), ep_line_p1(b))),
    describe("coincident {}.p2 {}.p1", line(a), line(b)));
sketch_constraint!(CoincidentLL22, points(), lines(a, b), arcs(),
    dedup(coincide(ep_line_p2(a), ep_line_p2(b))),
    describe("coincident {}.p2 {}.p2", line(a), line(b)));
sketch_constraint!(DistancePP, points(a, b), lines(), arcs(),
    dedup(exact(a, b)),
    describe("distance {} {} = {}", point(a), point(b), num(distance)));
sketch_constraint!(HorizontalDistancePP, points(a, b), lines(), arcs(),
    dedup(exact(a, b)),
    describe("hdistance {} {} = {}", point(a), point(b), num(distance)));
sketch_constraint!(VerticalDistancePP, points(a, b), lines(), arcs(),
    dedup(exact(a, b)),
    describe("vdistance {} {} = {}", point(a), point(b), num(distance)));
sketch_constraint!(PointOnLine, points(point), lines(line), arcs(),
    dedup(exact(point, line)),
    describe("point_on {} {}", point(point), line(line)));
sketch_constraint!(MidpointConstraint, points(point), lines(line), arcs(),
    dedup(exact(point, line)),
    describe("midpoint {} {}", point(point), line(line)));
sketch_constraint!(MidpointLP1, points(), lines(line, target), arcs(),
    dedup(exact(line, target)),
    describe("midpoint {}.p1 {}", line(line), line(target)));
sketch_constraint!(MidpointLP2, points(), lines(line, target), arcs(),
    dedup(exact(line, target)),
    describe("midpoint {}.p2 {}", line(line), line(target)));
sketch_constraint!(MidpointArcStart, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("midpoint {}.start {}", arc(arc), line(line)));
sketch_constraint!(MidpointArcEnd, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("midpoint {}.end {}", arc(arc), line(line)));
sketch_constraint!(MidpointArcPoint, points(point), lines(), arcs(arc),
    dedup(exact(point, arc)),
    describe("midpoint {} {}", point(point), arc(arc)));
sketch_constraint!(MidpointLP1Arc, points(), lines(line), arcs(arc),
    dedup(exact(line, arc)),
    describe("midpoint {}.p1 {}", line(line), arc(arc)));
sketch_constraint!(MidpointLP2Arc, points(), lines(line), arcs(arc),
    dedup(exact(line, arc)),
    describe("midpoint {}.p2 {}", line(line), arc(arc)));
sketch_constraint!(MidpointArcStartArc, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("midpoint {}.start {}", arc(a), arc(b)));
sketch_constraint!(MidpointArcEndArc, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("midpoint {}.end {}", arc(a), arc(b)));
sketch_constraint!(PointOnArc, points(point), lines(), arcs(arc),
    dedup(exact(point, arc)),
    describe("point_on {} {}", point(point), arc(arc)));
sketch_constraint!(Parallel, points(), lines(a, b), arcs(),
    dedup(sorted(a, b)),
    describe("parallel {} {}", line(a), line(b)));
sketch_constraint!(Perpendicular, points(), lines(a, b), arcs(),
    dedup(sorted(a, b)),
    describe("perpendicular {} {}", line(a), line(b)));
sketch_constraint!(ArcLineParallel, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("parallel {} {}", arc(arc), line(line)));
sketch_constraint!(ArcArcParallel, points(), lines(), arcs(a, b),
    dedup(sorted(a, b)),
    describe("parallel {} {}", arc(a), arc(b)));
sketch_constraint!(Collinear, points(), lines(a, b), arcs(),
    dedup(sorted(a, b)),
    describe("collinear {} {}", line(a), line(b)));
sketch_constraint!(EqualLength, points(), lines(a, b), arcs(),
    dedup(sorted(a, b)),
    describe("equal {} {}", line(a), line(b)));
sketch_constraint!(AngleConstraint, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("angle {} {} = {:.1}deg", line(a), line(b), deg(angle)));
sketch_constraint!(TangentLA, points(), lines(line), arcs(arc),
    dedup(exact(line, arc)),
    describe("tangent {} {}", line(line), arc(arc)));
sketch_constraint!(Concentric, points(), lines(), arcs(a, b),
    dedup(sorted(a, b)),
    describe("concentric {} {}", arc(a), arc(b)));
sketch_constraint!(EqualRadius, points(), lines(), arcs(a, b),
    dedup(sorted(a, b)),
    describe("equal {} {}", arc(a), arc(b)));
sketch_constraint!(TangentAA, points(), lines(), arcs(a, b),
    dedup(sorted(a, b)),
    describe("tangent {} {}", arc(a), arc(b)));
sketch_constraint!(SymmetryLL, points(), lines(a, b, c), arcs(),
    dedup(mirror(a, c; b)),
    describe("symmetry {} {} {}", line(a), line(b), line(c)));
sketch_constraint!(SymmetryPP, points(a, c), lines(line), arcs(),
    dedup(mirror(a, c; line)),
    describe("symmetry {} {} {}", point(a), line(line), point(c)));
sketch_constraint!(SymmetryAA, points(), lines(line), arcs(a, c),
    dedup(mirror(a, c; line)),
    describe("symmetry {} {} {}", arc(a), line(line), arc(c)));
sketch_constraint!(DistancePL, points(point), lines(line), arcs(),
    dedup(exact(point, line)),
    describe("distance {} {} = {}", point(point), line(line), num(distance)));
sketch_constraint!(DistanceLP1L, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {} = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLP2L, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {} = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceArcCenterL, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.center {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.start {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.end {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(LineP1OnLine, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("point_on {}.p1 {}", line(a), line(b)));
sketch_constraint!(LineP2OnLine, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("point_on {}.p2 {}", line(a), line(b)));
sketch_constraint!(CoincidentArcCenter, points(point), lines(), arcs(arc),
    dedup(coincide(ep_point(point), ep_arc_center(arc))),
    describe("coincident {} {}.center", point(point), arc(arc)));
sketch_constraint!(CoincidentArcStart, points(point), lines(), arcs(arc),
    dedup(coincide(ep_point(point), ep_arc_start(arc))),
    describe("coincident {} {}.start", point(point), arc(arc)));
sketch_constraint!(CoincidentArcEnd, points(point), lines(), arcs(arc),
    dedup(coincide(ep_point(point), ep_arc_end(arc))),
    describe("coincident {} {}.end", point(point), arc(arc)));
sketch_constraint!(CoincidentLP1ArcCenter, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p1(line), ep_arc_center(arc))),
    describe("coincident {}.p1 {}.center", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcCenter, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p2(line), ep_arc_center(arc))),
    describe("coincident {}.p2 {}.center", line(line), arc(arc)));
sketch_constraint!(CoincidentLP1ArcStart, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p1(line), ep_arc_start(arc))),
    describe("coincident {}.p1 {}.start", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcStart, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p2(line), ep_arc_start(arc))),
    describe("coincident {}.p2 {}.start", line(line), arc(arc)));
sketch_constraint!(CoincidentLP1ArcEnd, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p1(line), ep_arc_end(arc))),
    describe("coincident {}.p1 {}.end", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcEnd, points(), lines(line), arcs(arc),
    dedup(coincide(ep_line_p2(line), ep_arc_end(arc))),
    describe("coincident {}.p2 {}.end", line(line), arc(arc)));
sketch_constraint!(CoincidentArcCenterStart, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_center(a), ep_arc_start(b))),
    describe("coincident {}.center {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcCenterEnd, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_center(a), ep_arc_end(b))),
    describe("coincident {}.center {}.end", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartCenter, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_start(a), ep_arc_center(b))),
    describe("coincident {}.start {}.center", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndCenter, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_end(a), ep_arc_center(b))),
    describe("coincident {}.end {}.center", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartStart, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_start(a), ep_arc_start(b))),
    describe("coincident {}.start {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartEnd, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_start(a), ep_arc_end(b))),
    describe("coincident {}.start {}.end", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndStart, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_end(a), ep_arc_start(b))),
    describe("coincident {}.end {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndEnd, points(), lines(), arcs(a, b),
    dedup(coincide(ep_arc_end(a), ep_arc_end(b))),
    describe("coincident {}.end {}.end", arc(a), arc(b)));
sketch_constraint!(LineP1OnArc, points(), lines(line), arcs(arc),
    dedup(exact(line, arc)),
    describe("point_on {}.p1 {}", line(line), arc(arc)));
sketch_constraint!(LineP2OnArc, points(), lines(line), arcs(arc),
    dedup(exact(line, arc)),
    describe("point_on {}.p2 {}", line(line), arc(arc)));
sketch_constraint!(DistanceLL11, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {}.p1 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL12, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {}.p2 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL21, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {}.p1 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL22, points(), lines(a, b), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {}.p2 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLP1, points(point), lines(line), arcs(),
    dedup(exact(line, point)),
    describe("distance {}.p1 {} = {}", line(line), point(point), num(distance)));
sketch_constraint!(DistanceLP2, points(point), lines(line), arcs(),
    dedup(exact(line, point)),
    describe("distance {}.p2 {} = {}", line(line), point(point), num(distance)));
sketch_constraint!(DistanceArcCenterP, points(point), lines(), arcs(arc),
    dedup(exact(arc, point)),
    describe("distance {}.center {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcStartP, points(point), lines(), arcs(arc),
    dedup(exact(arc, point)),
    describe("distance {}.start {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcEndP, points(point), lines(), arcs(arc),
    dedup(exact(arc, point)),
    describe("distance {}.end {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcCenterL1, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.center {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcCenterL2, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.center {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL1, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.start {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL2, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.start {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL1, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.end {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL2, points(), lines(line), arcs(arc),
    dedup(exact(arc, line)),
    describe("distance {}.end {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceAACeCe, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.center {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAACeS, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.center {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAACeE, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.center {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASCe, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.start {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASS, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.start {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASE, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.start {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAECe, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.end {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAES, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.end {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAEE, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {}.end {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceConcentric, points(), lines(), arcs(a, b),
    dedup(exact(a, b)),
    describe("distance {} {} = {} (concentric)", arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceLL11, points(), lines(a, b), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p1 {}.p1 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL12, points(), lines(a, b), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p1 {}.p2 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL21, points(), lines(a, b), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p2 {}.p1 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL22, points(), lines(a, b), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p2 {}.p2 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLP1, points(point), lines(line), arcs(),
    dedup(axis(line, point; horizontal)),
    describe("{} {}.p1 {} = {}", axis(horizontal), line(line), point(point), num(distance)));
sketch_constraint!(AxisDistanceLP2, points(point), lines(line), arcs(),
    dedup(axis(line, point; horizontal)),
    describe("{} {}.p2 {} = {}", axis(horizontal), line(line), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcCenterP, points(point), lines(), arcs(arc),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.center {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcStartP, points(point), lines(), arcs(arc),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.start {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcEndP, points(point), lines(), arcs(arc),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.end {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcCenterL1, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.center {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcCenterL2, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.center {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcStartL1, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.start {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcStartL2, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.start {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcEndL1, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.end {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcEndL2, points(), lines(line), arcs(arc),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.end {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceAACeCe, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAACeS, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAACeE, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASCe, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASS, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASE, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAECe, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAES, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAEE, points(), lines(), arcs(a, b),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));

/// Number of registered constraint collections; a tripwire for tests.
pub const CONSTRAINT_COLLECTION_COUNT: usize = 112;

impl Sketch {
    /// Hand every constraint collection to `f`, mutably, with the
    /// entity arenas readable alongside. Exhaustive on purpose: a new
    /// collection field fails to compile until registered here and in
    /// the read-only walker below.
    pub fn for_each_constraint_collection(
        &mut self,
        mut f: impl FnMut(&ConstraintArenas, &CollectionMeta, &mut dyn ConstraintCollection),
    ) {
        let Sketch {
            points,
            lines,
            arcs,
            drift_isigma: _,
            constraint_isigma: _,
            min_length: _,
            next_point_id: _,
            next_line_id: _,
            next_arc_id: _,
            next_dimension_id: _,
            next_constraint_id: _,
            symbol_bag: _,
            expr_hb: _,
            cached_dof: _,
            cached_rank: _,
            structure_gen: _,
            dimensions: _,
            user_params: _,
            expr_constraints: _,
            coincident_pp,
            coincident_lp1,
            coincident_lp2,
            coincident_ll11,
            coincident_ll12,
            coincident_ll21,
            coincident_ll22,
            distance_pp,
            hdistance_pp,
            vdistance_pp,
            point_on_line,
            midpoint,
            midpoint_lp1,
            midpoint_lp2,
            midpoint_arc_start,
            midpoint_arc_end,
            midpoint_arc_point,
            midpoint_lp1_arc,
            midpoint_lp2_arc,
            midpoint_arc_start_arc,
            midpoint_arc_end_arc,
            point_on_arc,
            parallel,
            perpendicular,
            arc_line_parallel,
            arc_arc_parallel,
            collinear,
            equal_length,
            angle,
            tangent_la,
            concentric,
            equal_radius,
            tangent_aa,
            symmetry_ll,
            symmetry_pp,
            symmetry_aa,
            distance_pl,
            distance_lp1l,
            distance_lp2l,
            distance_arc_center_l,
            distance_arc_start_l,
            distance_arc_end_l,
            line_p1_on_line,
            line_p2_on_line,
            coincident_arc_center,
            coincident_arc_start,
            coincident_arc_end,
            coincident_lp1_arc_center,
            coincident_lp2_arc_center,
            coincident_lp1_arc_start,
            coincident_lp2_arc_start,
            coincident_lp1_arc_end,
            coincident_lp2_arc_end,
            coincident_arc_center_start,
            coincident_arc_center_end,
            coincident_arc_start_center,
            coincident_arc_end_center,
            coincident_arc_start_start,
            coincident_arc_start_end,
            coincident_arc_end_start,
            coincident_arc_end_end,
            line_p1_on_arc,
            line_p2_on_arc,
            distance_ll11,
            distance_ll12,
            distance_ll21,
            distance_ll22,
            distance_lp1,
            distance_lp2,
            distance_arc_center_p,
            distance_arc_start_p,
            distance_arc_end_p,
            distance_arc_center_l1,
            distance_arc_center_l2,
            distance_arc_start_l1,
            distance_arc_start_l2,
            distance_arc_end_l1,
            distance_arc_end_l2,
            distance_aa_ce_ce,
            distance_aa_ce_s,
            distance_aa_ce_e,
            distance_aa_s_ce,
            distance_aa_s_s,
            distance_aa_s_e,
            distance_aa_e_ce,
            distance_aa_e_s,
            distance_aa_e_e,
            distance_concentric,
            axis_distance_ll11,
            axis_distance_ll12,
            axis_distance_ll21,
            axis_distance_ll22,
            axis_distance_lp1,
            axis_distance_lp2,
            axis_distance_arc_center_p,
            axis_distance_arc_start_p,
            axis_distance_arc_end_p,
            axis_distance_arc_center_l1,
            axis_distance_arc_center_l2,
            axis_distance_arc_start_l1,
            axis_distance_arc_start_l2,
            axis_distance_arc_end_l1,
            axis_distance_arc_end_l2,
            axis_distance_aa_ce_ce,
            axis_distance_aa_ce_s,
            axis_distance_aa_ce_e,
            axis_distance_aa_s_ce,
            axis_distance_aa_s_s,
            axis_distance_aa_s_e,
            axis_distance_aa_e_ce,
            axis_distance_aa_e_s,
            axis_distance_aa_e_e,
        } = self;
        let arenas = ConstraintArenas { points, lines, arcs };
        f(&arenas, &CollectionMeta { name: "coincident_pp", coincidence: true, dimension_backed: false }, coincident_pp);
        f(&arenas, &CollectionMeta { name: "coincident_lp1", coincidence: true, dimension_backed: false }, coincident_lp1);
        f(&arenas, &CollectionMeta { name: "coincident_lp2", coincidence: true, dimension_backed: false }, coincident_lp2);
        f(&arenas, &CollectionMeta { name: "coincident_ll11", coincidence: true, dimension_backed: false }, coincident_ll11);
        f(&arenas, &CollectionMeta { name: "coincident_ll12", coincidence: true, dimension_backed: false }, coincident_ll12);
        f(&arenas, &CollectionMeta { name: "coincident_ll21", coincidence: true, dimension_backed: false }, coincident_ll21);
        f(&arenas, &CollectionMeta { name: "coincident_ll22", coincidence: true, dimension_backed: false }, coincident_ll22);
        f(&arenas, &CollectionMeta { name: "distance_pp", coincidence: false, dimension_backed: true }, distance_pp);
        f(&arenas, &CollectionMeta { name: "hdistance_pp", coincidence: false, dimension_backed: true }, hdistance_pp);
        f(&arenas, &CollectionMeta { name: "vdistance_pp", coincidence: false, dimension_backed: true }, vdistance_pp);
        f(&arenas, &CollectionMeta { name: "point_on_line", coincidence: false, dimension_backed: false }, point_on_line);
        f(&arenas, &CollectionMeta { name: "midpoint", coincidence: false, dimension_backed: false }, midpoint);
        f(&arenas, &CollectionMeta { name: "midpoint_lp1", coincidence: false, dimension_backed: false }, midpoint_lp1);
        f(&arenas, &CollectionMeta { name: "midpoint_lp2", coincidence: false, dimension_backed: false }, midpoint_lp2);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_start", coincidence: false, dimension_backed: false }, midpoint_arc_start);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_end", coincidence: false, dimension_backed: false }, midpoint_arc_end);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_point", coincidence: false, dimension_backed: false }, midpoint_arc_point);
        f(&arenas, &CollectionMeta { name: "midpoint_lp1_arc", coincidence: false, dimension_backed: false }, midpoint_lp1_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_lp2_arc", coincidence: false, dimension_backed: false }, midpoint_lp2_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_start_arc", coincidence: false, dimension_backed: false }, midpoint_arc_start_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_end_arc", coincidence: false, dimension_backed: false }, midpoint_arc_end_arc);
        f(&arenas, &CollectionMeta { name: "point_on_arc", coincidence: false, dimension_backed: false }, point_on_arc);
        f(&arenas, &CollectionMeta { name: "parallel", coincidence: false, dimension_backed: false }, parallel);
        f(&arenas, &CollectionMeta { name: "perpendicular", coincidence: false, dimension_backed: false }, perpendicular);
        f(&arenas, &CollectionMeta { name: "arc_line_parallel", coincidence: false, dimension_backed: false }, arc_line_parallel);
        f(&arenas, &CollectionMeta { name: "arc_arc_parallel", coincidence: false, dimension_backed: false }, arc_arc_parallel);
        f(&arenas, &CollectionMeta { name: "collinear", coincidence: false, dimension_backed: false }, collinear);
        f(&arenas, &CollectionMeta { name: "equal_length", coincidence: false, dimension_backed: false }, equal_length);
        f(&arenas, &CollectionMeta { name: "angle", coincidence: false, dimension_backed: true }, angle);
        f(&arenas, &CollectionMeta { name: "tangent_la", coincidence: false, dimension_backed: false }, tangent_la);
        f(&arenas, &CollectionMeta { name: "concentric", coincidence: false, dimension_backed: false }, concentric);
        f(&arenas, &CollectionMeta { name: "equal_radius", coincidence: false, dimension_backed: false }, equal_radius);
        f(&arenas, &CollectionMeta { name: "tangent_aa", coincidence: false, dimension_backed: false }, tangent_aa);
        f(&arenas, &CollectionMeta { name: "symmetry_ll", coincidence: false, dimension_backed: false }, symmetry_ll);
        f(&arenas, &CollectionMeta { name: "symmetry_pp", coincidence: false, dimension_backed: false }, symmetry_pp);
        f(&arenas, &CollectionMeta { name: "symmetry_aa", coincidence: false, dimension_backed: false }, symmetry_aa);
        f(&arenas, &CollectionMeta { name: "distance_pl", coincidence: false, dimension_backed: true }, distance_pl);
        f(&arenas, &CollectionMeta { name: "distance_lp1l", coincidence: false, dimension_backed: true }, distance_lp1l);
        f(&arenas, &CollectionMeta { name: "distance_lp2l", coincidence: false, dimension_backed: true }, distance_lp2l);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l", coincidence: false, dimension_backed: true }, distance_arc_center_l);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l", coincidence: false, dimension_backed: true }, distance_arc_start_l);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l", coincidence: false, dimension_backed: true }, distance_arc_end_l);
        f(&arenas, &CollectionMeta { name: "line_p1_on_line", coincidence: false, dimension_backed: false }, line_p1_on_line);
        f(&arenas, &CollectionMeta { name: "line_p2_on_line", coincidence: false, dimension_backed: false }, line_p2_on_line);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center", coincidence: true, dimension_backed: false }, coincident_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start", coincidence: true, dimension_backed: false }, coincident_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end", coincidence: true, dimension_backed: false }, coincident_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_center", coincidence: true, dimension_backed: false }, coincident_lp1_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_center", coincidence: true, dimension_backed: false }, coincident_lp2_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_start", coincidence: true, dimension_backed: false }, coincident_lp1_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_start", coincidence: true, dimension_backed: false }, coincident_lp2_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_end", coincidence: true, dimension_backed: false }, coincident_lp1_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_end", coincidence: true, dimension_backed: false }, coincident_lp2_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center_start", coincidence: true, dimension_backed: false }, coincident_arc_center_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center_end", coincidence: true, dimension_backed: false }, coincident_arc_center_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_center", coincidence: true, dimension_backed: false }, coincident_arc_start_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_center", coincidence: true, dimension_backed: false }, coincident_arc_end_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_start", coincidence: true, dimension_backed: false }, coincident_arc_start_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_end", coincidence: true, dimension_backed: false }, coincident_arc_start_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_start", coincidence: true, dimension_backed: false }, coincident_arc_end_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_end", coincidence: true, dimension_backed: false }, coincident_arc_end_end);
        f(&arenas, &CollectionMeta { name: "line_p1_on_arc", coincidence: false, dimension_backed: false }, line_p1_on_arc);
        f(&arenas, &CollectionMeta { name: "line_p2_on_arc", coincidence: false, dimension_backed: false }, line_p2_on_arc);
        f(&arenas, &CollectionMeta { name: "distance_ll11", coincidence: false, dimension_backed: true }, distance_ll11);
        f(&arenas, &CollectionMeta { name: "distance_ll12", coincidence: false, dimension_backed: true }, distance_ll12);
        f(&arenas, &CollectionMeta { name: "distance_ll21", coincidence: false, dimension_backed: true }, distance_ll21);
        f(&arenas, &CollectionMeta { name: "distance_ll22", coincidence: false, dimension_backed: true }, distance_ll22);
        f(&arenas, &CollectionMeta { name: "distance_lp1", coincidence: false, dimension_backed: true }, distance_lp1);
        f(&arenas, &CollectionMeta { name: "distance_lp2", coincidence: false, dimension_backed: true }, distance_lp2);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_p", coincidence: false, dimension_backed: true }, distance_arc_center_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_p", coincidence: false, dimension_backed: true }, distance_arc_start_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_p", coincidence: false, dimension_backed: true }, distance_arc_end_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l1", coincidence: false, dimension_backed: true }, distance_arc_center_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l2", coincidence: false, dimension_backed: true }, distance_arc_center_l2);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l1", coincidence: false, dimension_backed: true }, distance_arc_start_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l2", coincidence: false, dimension_backed: true }, distance_arc_start_l2);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l1", coincidence: false, dimension_backed: true }, distance_arc_end_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l2", coincidence: false, dimension_backed: true }, distance_arc_end_l2);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_ce", coincidence: false, dimension_backed: true }, distance_aa_ce_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_s", coincidence: false, dimension_backed: true }, distance_aa_ce_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_e", coincidence: false, dimension_backed: true }, distance_aa_ce_e);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_ce", coincidence: false, dimension_backed: true }, distance_aa_s_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_s", coincidence: false, dimension_backed: true }, distance_aa_s_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_e", coincidence: false, dimension_backed: true }, distance_aa_s_e);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_ce", coincidence: false, dimension_backed: true }, distance_aa_e_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_s", coincidence: false, dimension_backed: true }, distance_aa_e_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_e", coincidence: false, dimension_backed: true }, distance_aa_e_e);
        f(&arenas, &CollectionMeta { name: "distance_concentric", coincidence: false, dimension_backed: true }, distance_concentric);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll11", coincidence: false, dimension_backed: true }, axis_distance_ll11);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll12", coincidence: false, dimension_backed: true }, axis_distance_ll12);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll21", coincidence: false, dimension_backed: true }, axis_distance_ll21);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll22", coincidence: false, dimension_backed: true }, axis_distance_ll22);
        f(&arenas, &CollectionMeta { name: "axis_distance_lp1", coincidence: false, dimension_backed: true }, axis_distance_lp1);
        f(&arenas, &CollectionMeta { name: "axis_distance_lp2", coincidence: false, dimension_backed: true }, axis_distance_lp2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_p", coincidence: false, dimension_backed: true }, axis_distance_arc_center_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_p", coincidence: false, dimension_backed: true }, axis_distance_arc_start_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_p", coincidence: false, dimension_backed: true }, axis_distance_arc_end_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_l1", coincidence: false, dimension_backed: true }, axis_distance_arc_center_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_l2", coincidence: false, dimension_backed: true }, axis_distance_arc_center_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_l1", coincidence: false, dimension_backed: true }, axis_distance_arc_start_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_l2", coincidence: false, dimension_backed: true }, axis_distance_arc_start_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_l1", coincidence: false, dimension_backed: true }, axis_distance_arc_end_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_l2", coincidence: false, dimension_backed: true }, axis_distance_arc_end_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_ce", coincidence: false, dimension_backed: true }, axis_distance_aa_ce_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_s", coincidence: false, dimension_backed: true }, axis_distance_aa_ce_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_e", coincidence: false, dimension_backed: true }, axis_distance_aa_ce_e);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_ce", coincidence: false, dimension_backed: true }, axis_distance_aa_s_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_s", coincidence: false, dimension_backed: true }, axis_distance_aa_s_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_e", coincidence: false, dimension_backed: true }, axis_distance_aa_s_e);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_ce", coincidence: false, dimension_backed: true }, axis_distance_aa_e_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_s", coincidence: false, dimension_backed: true }, axis_distance_aa_e_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_e", coincidence: false, dimension_backed: true }, axis_distance_aa_e_e);
    }

    /// Read-only twin of [`Self::for_each_constraint_collection`].
    pub fn for_each_constraint_collection_ref(
        &self,
        mut f: impl FnMut(&ConstraintArenas, &CollectionMeta, &dyn ConstraintCollection),
    ) {
        let Sketch {
            points,
            lines,
            arcs,
            drift_isigma: _,
            constraint_isigma: _,
            min_length: _,
            next_point_id: _,
            next_line_id: _,
            next_arc_id: _,
            next_dimension_id: _,
            next_constraint_id: _,
            symbol_bag: _,
            expr_hb: _,
            cached_dof: _,
            cached_rank: _,
            structure_gen: _,
            dimensions: _,
            user_params: _,
            expr_constraints: _,
            coincident_pp,
            coincident_lp1,
            coincident_lp2,
            coincident_ll11,
            coincident_ll12,
            coincident_ll21,
            coincident_ll22,
            distance_pp,
            hdistance_pp,
            vdistance_pp,
            point_on_line,
            midpoint,
            midpoint_lp1,
            midpoint_lp2,
            midpoint_arc_start,
            midpoint_arc_end,
            midpoint_arc_point,
            midpoint_lp1_arc,
            midpoint_lp2_arc,
            midpoint_arc_start_arc,
            midpoint_arc_end_arc,
            point_on_arc,
            parallel,
            perpendicular,
            arc_line_parallel,
            arc_arc_parallel,
            collinear,
            equal_length,
            angle,
            tangent_la,
            concentric,
            equal_radius,
            tangent_aa,
            symmetry_ll,
            symmetry_pp,
            symmetry_aa,
            distance_pl,
            distance_lp1l,
            distance_lp2l,
            distance_arc_center_l,
            distance_arc_start_l,
            distance_arc_end_l,
            line_p1_on_line,
            line_p2_on_line,
            coincident_arc_center,
            coincident_arc_start,
            coincident_arc_end,
            coincident_lp1_arc_center,
            coincident_lp2_arc_center,
            coincident_lp1_arc_start,
            coincident_lp2_arc_start,
            coincident_lp1_arc_end,
            coincident_lp2_arc_end,
            coincident_arc_center_start,
            coincident_arc_center_end,
            coincident_arc_start_center,
            coincident_arc_end_center,
            coincident_arc_start_start,
            coincident_arc_start_end,
            coincident_arc_end_start,
            coincident_arc_end_end,
            line_p1_on_arc,
            line_p2_on_arc,
            distance_ll11,
            distance_ll12,
            distance_ll21,
            distance_ll22,
            distance_lp1,
            distance_lp2,
            distance_arc_center_p,
            distance_arc_start_p,
            distance_arc_end_p,
            distance_arc_center_l1,
            distance_arc_center_l2,
            distance_arc_start_l1,
            distance_arc_start_l2,
            distance_arc_end_l1,
            distance_arc_end_l2,
            distance_aa_ce_ce,
            distance_aa_ce_s,
            distance_aa_ce_e,
            distance_aa_s_ce,
            distance_aa_s_s,
            distance_aa_s_e,
            distance_aa_e_ce,
            distance_aa_e_s,
            distance_aa_e_e,
            distance_concentric,
            axis_distance_ll11,
            axis_distance_ll12,
            axis_distance_ll21,
            axis_distance_ll22,
            axis_distance_lp1,
            axis_distance_lp2,
            axis_distance_arc_center_p,
            axis_distance_arc_start_p,
            axis_distance_arc_end_p,
            axis_distance_arc_center_l1,
            axis_distance_arc_center_l2,
            axis_distance_arc_start_l1,
            axis_distance_arc_start_l2,
            axis_distance_arc_end_l1,
            axis_distance_arc_end_l2,
            axis_distance_aa_ce_ce,
            axis_distance_aa_ce_s,
            axis_distance_aa_ce_e,
            axis_distance_aa_s_ce,
            axis_distance_aa_s_s,
            axis_distance_aa_s_e,
            axis_distance_aa_e_ce,
            axis_distance_aa_e_s,
            axis_distance_aa_e_e,
        } = self;
        let arenas = ConstraintArenas { points, lines, arcs };
        f(&arenas, &CollectionMeta { name: "coincident_pp", coincidence: true, dimension_backed: false }, &*coincident_pp);
        f(&arenas, &CollectionMeta { name: "coincident_lp1", coincidence: true, dimension_backed: false }, &*coincident_lp1);
        f(&arenas, &CollectionMeta { name: "coincident_lp2", coincidence: true, dimension_backed: false }, &*coincident_lp2);
        f(&arenas, &CollectionMeta { name: "coincident_ll11", coincidence: true, dimension_backed: false }, &*coincident_ll11);
        f(&arenas, &CollectionMeta { name: "coincident_ll12", coincidence: true, dimension_backed: false }, &*coincident_ll12);
        f(&arenas, &CollectionMeta { name: "coincident_ll21", coincidence: true, dimension_backed: false }, &*coincident_ll21);
        f(&arenas, &CollectionMeta { name: "coincident_ll22", coincidence: true, dimension_backed: false }, &*coincident_ll22);
        f(&arenas, &CollectionMeta { name: "distance_pp", coincidence: false, dimension_backed: true }, &*distance_pp);
        f(&arenas, &CollectionMeta { name: "hdistance_pp", coincidence: false, dimension_backed: true }, &*hdistance_pp);
        f(&arenas, &CollectionMeta { name: "vdistance_pp", coincidence: false, dimension_backed: true }, &*vdistance_pp);
        f(&arenas, &CollectionMeta { name: "point_on_line", coincidence: false, dimension_backed: false }, &*point_on_line);
        f(&arenas, &CollectionMeta { name: "midpoint", coincidence: false, dimension_backed: false }, &*midpoint);
        f(&arenas, &CollectionMeta { name: "midpoint_lp1", coincidence: false, dimension_backed: false }, &*midpoint_lp1);
        f(&arenas, &CollectionMeta { name: "midpoint_lp2", coincidence: false, dimension_backed: false }, &*midpoint_lp2);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_start", coincidence: false, dimension_backed: false }, &*midpoint_arc_start);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_end", coincidence: false, dimension_backed: false }, &*midpoint_arc_end);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_point", coincidence: false, dimension_backed: false }, &*midpoint_arc_point);
        f(&arenas, &CollectionMeta { name: "midpoint_lp1_arc", coincidence: false, dimension_backed: false }, &*midpoint_lp1_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_lp2_arc", coincidence: false, dimension_backed: false }, &*midpoint_lp2_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_start_arc", coincidence: false, dimension_backed: false }, &*midpoint_arc_start_arc);
        f(&arenas, &CollectionMeta { name: "midpoint_arc_end_arc", coincidence: false, dimension_backed: false }, &*midpoint_arc_end_arc);
        f(&arenas, &CollectionMeta { name: "point_on_arc", coincidence: false, dimension_backed: false }, &*point_on_arc);
        f(&arenas, &CollectionMeta { name: "parallel", coincidence: false, dimension_backed: false }, &*parallel);
        f(&arenas, &CollectionMeta { name: "perpendicular", coincidence: false, dimension_backed: false }, &*perpendicular);
        f(&arenas, &CollectionMeta { name: "arc_line_parallel", coincidence: false, dimension_backed: false }, &*arc_line_parallel);
        f(&arenas, &CollectionMeta { name: "arc_arc_parallel", coincidence: false, dimension_backed: false }, &*arc_arc_parallel);
        f(&arenas, &CollectionMeta { name: "collinear", coincidence: false, dimension_backed: false }, &*collinear);
        f(&arenas, &CollectionMeta { name: "equal_length", coincidence: false, dimension_backed: false }, &*equal_length);
        f(&arenas, &CollectionMeta { name: "angle", coincidence: false, dimension_backed: true }, &*angle);
        f(&arenas, &CollectionMeta { name: "tangent_la", coincidence: false, dimension_backed: false }, &*tangent_la);
        f(&arenas, &CollectionMeta { name: "concentric", coincidence: false, dimension_backed: false }, &*concentric);
        f(&arenas, &CollectionMeta { name: "equal_radius", coincidence: false, dimension_backed: false }, &*equal_radius);
        f(&arenas, &CollectionMeta { name: "tangent_aa", coincidence: false, dimension_backed: false }, &*tangent_aa);
        f(&arenas, &CollectionMeta { name: "symmetry_ll", coincidence: false, dimension_backed: false }, &*symmetry_ll);
        f(&arenas, &CollectionMeta { name: "symmetry_pp", coincidence: false, dimension_backed: false }, &*symmetry_pp);
        f(&arenas, &CollectionMeta { name: "symmetry_aa", coincidence: false, dimension_backed: false }, &*symmetry_aa);
        f(&arenas, &CollectionMeta { name: "distance_pl", coincidence: false, dimension_backed: true }, &*distance_pl);
        f(&arenas, &CollectionMeta { name: "distance_lp1l", coincidence: false, dimension_backed: true }, &*distance_lp1l);
        f(&arenas, &CollectionMeta { name: "distance_lp2l", coincidence: false, dimension_backed: true }, &*distance_lp2l);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l", coincidence: false, dimension_backed: true }, &*distance_arc_center_l);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l", coincidence: false, dimension_backed: true }, &*distance_arc_start_l);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l", coincidence: false, dimension_backed: true }, &*distance_arc_end_l);
        f(&arenas, &CollectionMeta { name: "line_p1_on_line", coincidence: false, dimension_backed: false }, &*line_p1_on_line);
        f(&arenas, &CollectionMeta { name: "line_p2_on_line", coincidence: false, dimension_backed: false }, &*line_p2_on_line);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center", coincidence: true, dimension_backed: false }, &*coincident_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start", coincidence: true, dimension_backed: false }, &*coincident_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end", coincidence: true, dimension_backed: false }, &*coincident_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_center", coincidence: true, dimension_backed: false }, &*coincident_lp1_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_center", coincidence: true, dimension_backed: false }, &*coincident_lp2_arc_center);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_start", coincidence: true, dimension_backed: false }, &*coincident_lp1_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_start", coincidence: true, dimension_backed: false }, &*coincident_lp2_arc_start);
        f(&arenas, &CollectionMeta { name: "coincident_lp1_arc_end", coincidence: true, dimension_backed: false }, &*coincident_lp1_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_lp2_arc_end", coincidence: true, dimension_backed: false }, &*coincident_lp2_arc_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center_start", coincidence: true, dimension_backed: false }, &*coincident_arc_center_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_center_end", coincidence: true, dimension_backed: false }, &*coincident_arc_center_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_center", coincidence: true, dimension_backed: false }, &*coincident_arc_start_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_center", coincidence: true, dimension_backed: false }, &*coincident_arc_end_center);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_start", coincidence: true, dimension_backed: false }, &*coincident_arc_start_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_start_end", coincidence: true, dimension_backed: false }, &*coincident_arc_start_end);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_start", coincidence: true, dimension_backed: false }, &*coincident_arc_end_start);
        f(&arenas, &CollectionMeta { name: "coincident_arc_end_end", coincidence: true, dimension_backed: false }, &*coincident_arc_end_end);
        f(&arenas, &CollectionMeta { name: "line_p1_on_arc", coincidence: false, dimension_backed: false }, &*line_p1_on_arc);
        f(&arenas, &CollectionMeta { name: "line_p2_on_arc", coincidence: false, dimension_backed: false }, &*line_p2_on_arc);
        f(&arenas, &CollectionMeta { name: "distance_ll11", coincidence: false, dimension_backed: true }, &*distance_ll11);
        f(&arenas, &CollectionMeta { name: "distance_ll12", coincidence: false, dimension_backed: true }, &*distance_ll12);
        f(&arenas, &CollectionMeta { name: "distance_ll21", coincidence: false, dimension_backed: true }, &*distance_ll21);
        f(&arenas, &CollectionMeta { name: "distance_ll22", coincidence: false, dimension_backed: true }, &*distance_ll22);
        f(&arenas, &CollectionMeta { name: "distance_lp1", coincidence: false, dimension_backed: true }, &*distance_lp1);
        f(&arenas, &CollectionMeta { name: "distance_lp2", coincidence: false, dimension_backed: true }, &*distance_lp2);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_p", coincidence: false, dimension_backed: true }, &*distance_arc_center_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_p", coincidence: false, dimension_backed: true }, &*distance_arc_start_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_p", coincidence: false, dimension_backed: true }, &*distance_arc_end_p);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l1", coincidence: false, dimension_backed: true }, &*distance_arc_center_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_center_l2", coincidence: false, dimension_backed: true }, &*distance_arc_center_l2);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l1", coincidence: false, dimension_backed: true }, &*distance_arc_start_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_start_l2", coincidence: false, dimension_backed: true }, &*distance_arc_start_l2);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l1", coincidence: false, dimension_backed: true }, &*distance_arc_end_l1);
        f(&arenas, &CollectionMeta { name: "distance_arc_end_l2", coincidence: false, dimension_backed: true }, &*distance_arc_end_l2);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_ce", coincidence: false, dimension_backed: true }, &*distance_aa_ce_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_s", coincidence: false, dimension_backed: true }, &*distance_aa_ce_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_ce_e", coincidence: false, dimension_backed: true }, &*distance_aa_ce_e);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_ce", coincidence: false, dimension_backed: true }, &*distance_aa_s_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_s", coincidence: false, dimension_backed: true }, &*distance_aa_s_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_s_e", coincidence: false, dimension_backed: true }, &*distance_aa_s_e);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_ce", coincidence: false, dimension_backed: true }, &*distance_aa_e_ce);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_s", coincidence: false, dimension_backed: true }, &*distance_aa_e_s);
        f(&arenas, &CollectionMeta { name: "distance_aa_e_e", coincidence: false, dimension_backed: true }, &*distance_aa_e_e);
        f(&arenas, &CollectionMeta { name: "distance_concentric", coincidence: false, dimension_backed: true }, &*distance_concentric);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll11", coincidence: false, dimension_backed: true }, &*axis_distance_ll11);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll12", coincidence: false, dimension_backed: true }, &*axis_distance_ll12);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll21", coincidence: false, dimension_backed: true }, &*axis_distance_ll21);
        f(&arenas, &CollectionMeta { name: "axis_distance_ll22", coincidence: false, dimension_backed: true }, &*axis_distance_ll22);
        f(&arenas, &CollectionMeta { name: "axis_distance_lp1", coincidence: false, dimension_backed: true }, &*axis_distance_lp1);
        f(&arenas, &CollectionMeta { name: "axis_distance_lp2", coincidence: false, dimension_backed: true }, &*axis_distance_lp2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_p", coincidence: false, dimension_backed: true }, &*axis_distance_arc_center_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_p", coincidence: false, dimension_backed: true }, &*axis_distance_arc_start_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_p", coincidence: false, dimension_backed: true }, &*axis_distance_arc_end_p);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_l1", coincidence: false, dimension_backed: true }, &*axis_distance_arc_center_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_center_l2", coincidence: false, dimension_backed: true }, &*axis_distance_arc_center_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_l1", coincidence: false, dimension_backed: true }, &*axis_distance_arc_start_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_start_l2", coincidence: false, dimension_backed: true }, &*axis_distance_arc_start_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_l1", coincidence: false, dimension_backed: true }, &*axis_distance_arc_end_l1);
        f(&arenas, &CollectionMeta { name: "axis_distance_arc_end_l2", coincidence: false, dimension_backed: true }, &*axis_distance_arc_end_l2);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_ce", coincidence: false, dimension_backed: true }, &*axis_distance_aa_ce_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_s", coincidence: false, dimension_backed: true }, &*axis_distance_aa_ce_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_ce_e", coincidence: false, dimension_backed: true }, &*axis_distance_aa_ce_e);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_ce", coincidence: false, dimension_backed: true }, &*axis_distance_aa_s_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_s", coincidence: false, dimension_backed: true }, &*axis_distance_aa_s_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_s_e", coincidence: false, dimension_backed: true }, &*axis_distance_aa_s_e);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_ce", coincidence: false, dimension_backed: true }, &*axis_distance_aa_e_ce);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_s", coincidence: false, dimension_backed: true }, &*axis_distance_aa_e_s);
        f(&arenas, &CollectionMeta { name: "axis_distance_aa_e_e", coincidence: false, dimension_backed: true }, &*axis_distance_aa_e_e);
    }
}

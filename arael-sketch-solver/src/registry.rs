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

/// Endpoint role carried in the low bits of a canonical endpoint id;
/// the inverse of the `ep_*` encoders above.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EndpointRole {
    Point,
    LineP1,
    LineP2,
    ArcCenter,
    ArcStart,
    ArcEnd,
}

/// Decode a canonical endpoint id into its role and arena index.
pub fn decode_endpoint(enc: u64) -> (EndpointRole, u32) {
    let role = match enc & 7 {
        0 => EndpointRole::Point,
        1 => EndpointRole::LineP1,
        2 => EndpointRole::LineP2,
        3 => EndpointRole::ArcCenter,
        4 => EndpointRole::ArcStart,
        5 => EndpointRole::ArcEnd,
        r => panic!("decode_endpoint: unknown role {}", r),
    };
    (role, (enc >> 3) as u32)
}

/// Role a line/arc reference plays in its constraint, declared per
/// field at the registration site. The split engine's reference
/// transfer is driven entirely by these (see docs/dev/TRIMSPLIT.md):
/// the role decides which piece of a split entity inherits the
/// reference, whether the constraint is replicated, or whether it is
/// dropped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefRole {
    /// Line p1 / arc start: follows the piece owning that endpoint.
    Start,
    /// Line p2 / arc end: follows the piece owning that endpoint.
    End,
    /// Arc center: shared by every piece; follows the first survivor.
    Center,
    /// The unbounded curve (point-on, distance-to-line anchor,
    /// symmetry mirror): follows the piece nearest the constraint's
    /// other referents.
    Host,
    /// Direction or shape of the whole entity: replicated onto every
    /// surviving piece.
    Whole,
    /// Tangency: follows the piece containing the contact point.
    Contact,
    /// A measure of the whole span (equal length, midpoint-of,
    /// symmetry operand): dropped on split.
    Extent,
}

// Lowercase role tokens for the sketch_constraint! field lists.
macro_rules! ref_role {
    (start) => { RefRole::Start };
    (end) => { RefRole::End };
    (center) => { RefRole::Center };
    (host) => { RefRole::Host };
    (whole) => { RefRole::Whole };
    (contact) => { RefRole::Contact };
    (extent) => { RefRole::Extent };
}

/// The interface every constraint struct implements.
pub trait SketchConstraint {
    fn nid(&self) -> u32;
    fn set_nid(&mut self, nid: u32);
    fn cid(&self) -> u32;
    fn set_cid(&mut self, cid: u32);
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
    /// Visit every line-ref field: (slot among line fields, ref, role).
    fn each_line_field(&self, f: &mut dyn FnMut(usize, Ref<Line>, RefRole));
    /// Rewrite the line-ref field at `slot` (numbering of each_line_field).
    fn set_line_field(&mut self, slot: usize, new: Ref<Line>);
    /// Visit every arc-ref field: (slot among arc fields, ref, role).
    fn each_arc_field(&self, f: &mut dyn FnMut(usize, Ref<Arc>, RefRole));
    /// Rewrite the arc-ref field at `slot` (numbering of each_arc_field).
    fn set_arc_field(&mut self, slot: usize, new: Ref<Arc>);
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
    /// Clone item `i` and push the copy with nid/cid zeroed (fresh ids
    /// minted by `assign_constraint_names`). Returns the copy's index.
    /// The split engine's Whole-role replication runs through this.
    fn clone_push_blank(&mut self, i: usize) -> usize;
}

impl<C: SketchConstraint + Clone> ConstraintCollection for Vec<C> {
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
    fn clone_push_blank(&mut self, i: usize) -> usize {
        let mut c = self[i].clone();
        c.set_nid(0);
        c.set_cid(0);
        self.push(c);
        self.len() - 1
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
    ($ty:ident, points($($p:ident),*), lines($($l:ident: $lr:tt),*), arcs($($a:ident: $arr:tt),*),
     dedup($($dk:tt)*), describe($fmt:literal $(, $ar:ident($af:ident))*)) => {
        impl SketchConstraint for crate::$ty {
            fn nid(&self) -> u32 { self.nid }
            fn set_nid(&mut self, nid: u32) { self.nid = nid; }
            fn cid(&self) -> u32 { self.cid }
            fn set_cid(&mut self, cid: u32) { self.cid = cid; }
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
            fn each_line_field(&self, f: &mut dyn FnMut(usize, Ref<Line>, RefRole)) {
                let _ = &f;
                let mut _slot = 0usize;
                $(f(_slot, self.$l, ref_role!($lr)); _slot += 1;)*
            }
            fn set_line_field(&mut self, slot: usize, new: Ref<Line>) {
                let _ = &new;
                let mut _slot = 0usize;
                $(if _slot == slot { self.$l = new; return; } _slot += 1;)*
                panic!("set_line_field: {} has no line field at slot {}",
                    stringify!($ty), slot);
            }
            fn each_arc_field(&self, f: &mut dyn FnMut(usize, Ref<Arc>, RefRole)) {
                let _ = &f;
                let mut _slot = 0usize;
                $(f(_slot, self.$a, ref_role!($arr)); _slot += 1;)*
            }
            fn set_arc_field(&mut self, slot: usize, new: Ref<Arc>) {
                let _ = &new;
                let mut _slot = 0usize;
                $(if _slot == slot { self.$a = new; return; } _slot += 1;)*
                panic!("set_arc_field: {} has no arc field at slot {}",
                    stringify!($ty), slot);
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
sketch_constraint!(CoincidentLP1, points(point), lines(line: start), arcs(),
    dedup(coincide(ep_line_p1(line), ep_point(point))),
    describe("coincident {}.p1 {}", line(line), point(point)));
sketch_constraint!(CoincidentLP2, points(point), lines(line: end), arcs(),
    dedup(coincide(ep_line_p2(line), ep_point(point))),
    describe("coincident {}.p2 {}", line(line), point(point)));
sketch_constraint!(CoincidentLL11, points(), lines(a: start, b: start), arcs(),
    dedup(coincide(ep_line_p1(a), ep_line_p1(b))),
    describe("coincident {}.p1 {}.p1", line(a), line(b)));
sketch_constraint!(CoincidentLL12, points(), lines(a: start, b: end), arcs(),
    dedup(coincide(ep_line_p1(a), ep_line_p2(b))),
    describe("coincident {}.p1 {}.p2", line(a), line(b)));
sketch_constraint!(CoincidentLL21, points(), lines(a: end, b: start), arcs(),
    dedup(coincide(ep_line_p2(a), ep_line_p1(b))),
    describe("coincident {}.p2 {}.p1", line(a), line(b)));
sketch_constraint!(CoincidentLL22, points(), lines(a: end, b: end), arcs(),
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
sketch_constraint!(PointOnLine, points(point), lines(line: host), arcs(),
    dedup(exact(point, line)),
    describe("point_on {} {}", point(point), line(line)));
sketch_constraint!(MidpointConstraint, points(point), lines(line: extent), arcs(),
    dedup(exact(point, line)),
    describe("midpoint {} {}", point(point), line(line)));
sketch_constraint!(MidpointLP1, points(), lines(line: start, target: extent), arcs(),
    dedup(exact(line, target)),
    describe("midpoint {}.p1 {}", line(line), line(target)));
sketch_constraint!(MidpointLP2, points(), lines(line: end, target: extent), arcs(),
    dedup(exact(line, target)),
    describe("midpoint {}.p2 {}", line(line), line(target)));
sketch_constraint!(MidpointArcStart, points(), lines(line: extent), arcs(arc: start),
    dedup(exact(arc, line)),
    describe("midpoint {}.start {}", arc(arc), line(line)));
sketch_constraint!(MidpointArcEnd, points(), lines(line: extent), arcs(arc: end),
    dedup(exact(arc, line)),
    describe("midpoint {}.end {}", arc(arc), line(line)));
sketch_constraint!(MidpointArcPoint, points(point), lines(), arcs(arc: extent),
    dedup(exact(point, arc)),
    describe("midpoint {} {}", point(point), arc(arc)));
sketch_constraint!(MidpointLP1Arc, points(), lines(line: start), arcs(arc: extent),
    dedup(exact(line, arc)),
    describe("midpoint {}.p1 {}", line(line), arc(arc)));
sketch_constraint!(MidpointLP2Arc, points(), lines(line: end), arcs(arc: extent),
    dedup(exact(line, arc)),
    describe("midpoint {}.p2 {}", line(line), arc(arc)));
sketch_constraint!(MidpointArcStartArc, points(), lines(), arcs(a: start, b: extent),
    dedup(exact(a, b)),
    describe("midpoint {}.start {}", arc(a), arc(b)));
sketch_constraint!(MidpointArcEndArc, points(), lines(), arcs(a: end, b: extent),
    dedup(exact(a, b)),
    describe("midpoint {}.end {}", arc(a), arc(b)));
sketch_constraint!(PointOnArc, points(point), lines(), arcs(arc: host),
    dedup(exact(point, arc)),
    describe("point_on {} {}", point(point), arc(arc)));
sketch_constraint!(Parallel, points(), lines(a: whole, b: whole), arcs(),
    dedup(sorted(a, b)),
    describe("parallel {} {}", line(a), line(b)));
sketch_constraint!(Perpendicular, points(), lines(a: whole, b: whole), arcs(),
    dedup(sorted(a, b)),
    describe("perpendicular {} {}", line(a), line(b)));
sketch_constraint!(ArcLineParallel, points(), lines(line: whole), arcs(arc: whole),
    dedup(exact(arc, line)),
    describe("parallel {} {}", arc(arc), line(line)));
sketch_constraint!(ArcArcParallel, points(), lines(), arcs(a: whole, b: whole),
    dedup(sorted(a, b)),
    describe("parallel {} {}", arc(a), arc(b)));
sketch_constraint!(Collinear, points(), lines(a: whole, b: whole), arcs(),
    dedup(sorted(a, b)),
    describe("collinear {} {}", line(a), line(b)));
sketch_constraint!(EqualLength, points(), lines(a: extent, b: extent), arcs(),
    dedup(sorted(a, b)),
    describe("equal {} {}", line(a), line(b)));
sketch_constraint!(AngleConstraint, points(), lines(a: whole, b: whole), arcs(),
    dedup(exact(a, b)),
    describe("angle {} {} = {:.1}deg", line(a), line(b), deg(angle)));
sketch_constraint!(TangentLA, points(), lines(line: contact), arcs(arc: contact),
    dedup(exact(line, arc)),
    describe("tangent {} {}", line(line), arc(arc)));
sketch_constraint!(Concentric, points(), lines(), arcs(a: whole, b: whole),
    dedup(sorted(a, b)),
    describe("concentric {} {}", arc(a), arc(b)));
sketch_constraint!(EqualRadius, points(), lines(), arcs(a: whole, b: whole),
    dedup(sorted(a, b)),
    describe("equal {} {}", arc(a), arc(b)));
sketch_constraint!(TangentAA, points(), lines(), arcs(a: contact, b: contact),
    dedup(sorted(a, b)),
    describe("tangent {} {}", arc(a), arc(b)));
sketch_constraint!(SymmetryLL, points(), lines(a: extent, b: host, c: extent), arcs(),
    dedup(mirror(a, c; b)),
    describe("symmetry {} {} {}", line(a), line(b), line(c)));
sketch_constraint!(SymmetryPP, points(a, c), lines(line: host), arcs(),
    dedup(mirror(a, c; line)),
    describe("symmetry {} {} {}", point(a), line(line), point(c)));
sketch_constraint!(SymmetryAA, points(), lines(line: host), arcs(a: whole, c: whole),
    dedup(mirror(a, c; line)),
    describe("symmetry {} {} {}", arc(a), line(line), arc(c)));
sketch_constraint!(DistancePL, points(point), lines(line: host), arcs(),
    dedup(exact(point, line)),
    describe("distance {} {} = {}", point(point), line(line), num(distance)));
sketch_constraint!(DistanceLP1L, points(), lines(a: start, b: host), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {} = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLP2L, points(), lines(a: end, b: host), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {} = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceArcCenterL, points(), lines(line: host), arcs(arc: center),
    dedup(exact(arc, line)),
    describe("distance {}.center {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL, points(), lines(line: host), arcs(arc: start),
    dedup(exact(arc, line)),
    describe("distance {}.start {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL, points(), lines(line: host), arcs(arc: end),
    dedup(exact(arc, line)),
    describe("distance {}.end {} = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(LineP1OnLine, points(), lines(a: start, b: host), arcs(),
    dedup(exact(a, b)),
    describe("point_on {}.p1 {}", line(a), line(b)));
sketch_constraint!(LineP2OnLine, points(), lines(a: end, b: host), arcs(),
    dedup(exact(a, b)),
    describe("point_on {}.p2 {}", line(a), line(b)));
sketch_constraint!(CoincidentArcCenter, points(point), lines(), arcs(arc: center),
    dedup(coincide(ep_point(point), ep_arc_center(arc))),
    describe("coincident {} {}.center", point(point), arc(arc)));
sketch_constraint!(CoincidentArcStart, points(point), lines(), arcs(arc: start),
    dedup(coincide(ep_point(point), ep_arc_start(arc))),
    describe("coincident {} {}.start", point(point), arc(arc)));
sketch_constraint!(CoincidentArcEnd, points(point), lines(), arcs(arc: end),
    dedup(coincide(ep_point(point), ep_arc_end(arc))),
    describe("coincident {} {}.end", point(point), arc(arc)));
sketch_constraint!(CoincidentLP1ArcCenter, points(), lines(line: start), arcs(arc: center),
    dedup(coincide(ep_line_p1(line), ep_arc_center(arc))),
    describe("coincident {}.p1 {}.center", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcCenter, points(), lines(line: end), arcs(arc: center),
    dedup(coincide(ep_line_p2(line), ep_arc_center(arc))),
    describe("coincident {}.p2 {}.center", line(line), arc(arc)));
sketch_constraint!(CoincidentLP1ArcStart, points(), lines(line: start), arcs(arc: start),
    dedup(coincide(ep_line_p1(line), ep_arc_start(arc))),
    describe("coincident {}.p1 {}.start", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcStart, points(), lines(line: end), arcs(arc: start),
    dedup(coincide(ep_line_p2(line), ep_arc_start(arc))),
    describe("coincident {}.p2 {}.start", line(line), arc(arc)));
sketch_constraint!(CoincidentLP1ArcEnd, points(), lines(line: start), arcs(arc: end),
    dedup(coincide(ep_line_p1(line), ep_arc_end(arc))),
    describe("coincident {}.p1 {}.end", line(line), arc(arc)));
sketch_constraint!(CoincidentLP2ArcEnd, points(), lines(line: end), arcs(arc: end),
    dedup(coincide(ep_line_p2(line), ep_arc_end(arc))),
    describe("coincident {}.p2 {}.end", line(line), arc(arc)));
sketch_constraint!(CoincidentArcCenterStart, points(), lines(), arcs(a: center, b: start),
    dedup(coincide(ep_arc_center(a), ep_arc_start(b))),
    describe("coincident {}.center {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcCenterEnd, points(), lines(), arcs(a: center, b: end),
    dedup(coincide(ep_arc_center(a), ep_arc_end(b))),
    describe("coincident {}.center {}.end", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartCenter, points(), lines(), arcs(a: start, b: center),
    dedup(coincide(ep_arc_start(a), ep_arc_center(b))),
    describe("coincident {}.start {}.center", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndCenter, points(), lines(), arcs(a: end, b: center),
    dedup(coincide(ep_arc_end(a), ep_arc_center(b))),
    describe("coincident {}.end {}.center", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartStart, points(), lines(), arcs(a: start, b: start),
    dedup(coincide(ep_arc_start(a), ep_arc_start(b))),
    describe("coincident {}.start {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcStartEnd, points(), lines(), arcs(a: start, b: end),
    dedup(coincide(ep_arc_start(a), ep_arc_end(b))),
    describe("coincident {}.start {}.end", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndStart, points(), lines(), arcs(a: end, b: start),
    dedup(coincide(ep_arc_end(a), ep_arc_start(b))),
    describe("coincident {}.end {}.start", arc(a), arc(b)));
sketch_constraint!(CoincidentArcEndEnd, points(), lines(), arcs(a: end, b: end),
    dedup(coincide(ep_arc_end(a), ep_arc_end(b))),
    describe("coincident {}.end {}.end", arc(a), arc(b)));
sketch_constraint!(LineP1OnArc, points(), lines(line: start), arcs(arc: host),
    dedup(exact(line, arc)),
    describe("point_on {}.p1 {}", line(line), arc(arc)));
sketch_constraint!(LineP2OnArc, points(), lines(line: end), arcs(arc: host),
    dedup(exact(line, arc)),
    describe("point_on {}.p2 {}", line(line), arc(arc)));
sketch_constraint!(DistanceLL11, points(), lines(a: start, b: start), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {}.p1 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL12, points(), lines(a: start, b: end), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p1 {}.p2 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL21, points(), lines(a: end, b: start), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {}.p1 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLL22, points(), lines(a: end, b: end), arcs(),
    dedup(exact(a, b)),
    describe("distance {}.p2 {}.p2 = {}", line(a), line(b), num(distance)));
sketch_constraint!(DistanceLP1, points(point), lines(line: start), arcs(),
    dedup(exact(line, point)),
    describe("distance {}.p1 {} = {}", line(line), point(point), num(distance)));
sketch_constraint!(DistanceLP2, points(point), lines(line: end), arcs(),
    dedup(exact(line, point)),
    describe("distance {}.p2 {} = {}", line(line), point(point), num(distance)));
sketch_constraint!(DistanceArcCenterP, points(point), lines(), arcs(arc: center),
    dedup(exact(arc, point)),
    describe("distance {}.center {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcStartP, points(point), lines(), arcs(arc: start),
    dedup(exact(arc, point)),
    describe("distance {}.start {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcEndP, points(point), lines(), arcs(arc: end),
    dedup(exact(arc, point)),
    describe("distance {}.end {} = {}", arc(arc), point(point), num(distance)));
sketch_constraint!(DistanceArcCenterL1, points(), lines(line: start), arcs(arc: center),
    dedup(exact(arc, line)),
    describe("distance {}.center {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcCenterL2, points(), lines(line: end), arcs(arc: center),
    dedup(exact(arc, line)),
    describe("distance {}.center {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL1, points(), lines(line: start), arcs(arc: start),
    dedup(exact(arc, line)),
    describe("distance {}.start {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcStartL2, points(), lines(line: end), arcs(arc: start),
    dedup(exact(arc, line)),
    describe("distance {}.start {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL1, points(), lines(line: start), arcs(arc: end),
    dedup(exact(arc, line)),
    describe("distance {}.end {}.p1 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceArcEndL2, points(), lines(line: end), arcs(arc: end),
    dedup(exact(arc, line)),
    describe("distance {}.end {}.p2 = {}", arc(arc), line(line), num(distance)));
sketch_constraint!(DistanceAACeCe, points(), lines(), arcs(a: center, b: center),
    dedup(exact(a, b)),
    describe("distance {}.center {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAACeS, points(), lines(), arcs(a: center, b: start),
    dedup(exact(a, b)),
    describe("distance {}.center {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAACeE, points(), lines(), arcs(a: center, b: end),
    dedup(exact(a, b)),
    describe("distance {}.center {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASCe, points(), lines(), arcs(a: start, b: center),
    dedup(exact(a, b)),
    describe("distance {}.start {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASS, points(), lines(), arcs(a: start, b: start),
    dedup(exact(a, b)),
    describe("distance {}.start {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAASE, points(), lines(), arcs(a: start, b: end),
    dedup(exact(a, b)),
    describe("distance {}.start {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAECe, points(), lines(), arcs(a: end, b: center),
    dedup(exact(a, b)),
    describe("distance {}.end {}.center = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAES, points(), lines(), arcs(a: end, b: start),
    dedup(exact(a, b)),
    describe("distance {}.end {}.start = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceAAEE, points(), lines(), arcs(a: end, b: end),
    dedup(exact(a, b)),
    describe("distance {}.end {}.end = {}", arc(a), arc(b), num(distance)));
sketch_constraint!(DistanceConcentric, points(), lines(), arcs(a: whole, b: whole),
    dedup(exact(a, b)),
    describe("distance {} {} = {} (concentric)", arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceLL11, points(), lines(a: start, b: start), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p1 {}.p1 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL12, points(), lines(a: start, b: end), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p1 {}.p2 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL21, points(), lines(a: end, b: start), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p2 {}.p1 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLL22, points(), lines(a: end, b: end), arcs(),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.p2 {}.p2 = {}", axis(horizontal), line(a), line(b), num(distance)));
sketch_constraint!(AxisDistanceLP1, points(point), lines(line: start), arcs(),
    dedup(axis(line, point; horizontal)),
    describe("{} {}.p1 {} = {}", axis(horizontal), line(line), point(point), num(distance)));
sketch_constraint!(AxisDistanceLP2, points(point), lines(line: end), arcs(),
    dedup(axis(line, point; horizontal)),
    describe("{} {}.p2 {} = {}", axis(horizontal), line(line), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcCenterP, points(point), lines(), arcs(arc: center),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.center {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcStartP, points(point), lines(), arcs(arc: start),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.start {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcEndP, points(point), lines(), arcs(arc: end),
    dedup(axis(arc, point; horizontal)),
    describe("{} {}.end {} = {}", axis(horizontal), arc(arc), point(point), num(distance)));
sketch_constraint!(AxisDistanceArcCenterL1, points(), lines(line: start), arcs(arc: center),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.center {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcCenterL2, points(), lines(line: end), arcs(arc: center),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.center {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcStartL1, points(), lines(line: start), arcs(arc: start),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.start {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcStartL2, points(), lines(line: end), arcs(arc: start),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.start {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcEndL1, points(), lines(line: start), arcs(arc: end),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.end {}.p1 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceArcEndL2, points(), lines(line: end), arcs(arc: end),
    dedup(axis(arc, line; horizontal)),
    describe("{} {}.end {}.p2 = {}", axis(horizontal), arc(arc), line(line), num(distance)));
sketch_constraint!(AxisDistanceAACeCe, points(), lines(), arcs(a: center, b: center),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAACeS, points(), lines(), arcs(a: center, b: start),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAACeE, points(), lines(), arcs(a: center, b: end),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.center {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASCe, points(), lines(), arcs(a: start, b: center),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASS, points(), lines(), arcs(a: start, b: start),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAASE, points(), lines(), arcs(a: start, b: end),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.start {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAECe, points(), lines(), arcs(a: end, b: center),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.center = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAES, points(), lines(), arcs(a: end, b: start),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.start = {}", axis(horizontal), arc(a), arc(b), num(distance)));
sketch_constraint!(AxisDistanceAAEE, points(), lines(), arcs(a: end, b: end),
    dedup(axis(a, b; horizontal)),
    describe("{} {}.end {}.end = {}", axis(horizontal), arc(a), arc(b), num(distance)));

/// Number of registered constraint collections; a tripwire for tests.
pub const CONSTRAINT_COLLECTION_COUNT: usize = 112;

impl Sketch {
    /// Hand every coincidence constraint's canonical endpoint pair to
    /// `f`, in the same encoding the dedup keys use (decode with
    /// [`decode_endpoint`]). Registry-driven, so a new coincidence
    /// collection participates without any caller changing -- this is
    /// what the GUI's transitive-coincidence union-finds build from.
    pub fn for_each_coincidence_pair(&self, mut f: impl FnMut(u64, u64)) {
        self.for_each_constraint_collection_ref(|_, meta, coll| {
            if !meta.coincidence {
                return;
            }
            for i in 0..coll.len() {
                match coll.item(i).dedup_key() {
                    DedupKey::Coincidence(a, b) => f(a, b),
                    // A collection flagged coincidence must dedup as
                    // one; a Local key here would silently drop its
                    // unions from every consumer.
                    k => panic!(
                        "collection {} is marked coincidence but deduped {:?}",
                        meta.name, k
                    ),
                }
            }
        });
    }
}

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

#[cfg(test)]
mod role_tests {
    use super::*;
    use arael::model::CrossBlock;
    use arael::vect::vect2d;

    fn two_line_sketch() -> (Sketch, Ref<Line>, Ref<Line>) {
        let mut s = Sketch::new();
        let a = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
        let b = s.add_line(vect2d::new(0.0, 1.0), vect2d::new(5.0, 1.0));
        (s, a, b)
    }

    #[test]
    fn test_roles_and_slots() {
        let (mut s, a, b) = two_line_sketch();
        s.coincident_ll21.push(crate::CoincidentLL21 {
            a, b, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        let c = &s.coincident_ll21[0];
        let mut fields = Vec::new();
        c.each_line_field(&mut |slot, r, role| fields.push((slot, r, role)));
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], (0, a, RefRole::End));
        assert_eq!(fields[1], (1, b, RefRole::Start));
        // No arc fields.
        let mut arc_fields = 0;
        c.each_arc_field(&mut |_, _, _| arc_fields += 1);
        assert_eq!(arc_fields, 0);
    }

    #[test]
    fn test_set_line_field_rewrites_one_slot() {
        let (mut s, a, b) = two_line_sketch();
        let c2 = s.add_line(vect2d::new(0.0, 2.0), vect2d::new(5.0, 2.0));
        s.parallel.push(crate::Parallel { a, b, nid: 0, cid: 0, hb: CrossBlock::new() });
        s.parallel[0].set_line_field(1, c2);
        assert_eq!(s.parallel[0].a, a);
        assert_eq!(s.parallel[0].b, c2);
    }

    #[test]
    fn test_clone_push_blank_zeroes_ids() {
        let (mut s, a, b) = two_line_sketch();
        s.perpendicular.push(crate::Perpendicular {
            a, b, dir_sign: 1.0, nid: 7, cid: 3, hb: CrossBlock::new(),
        });
        let coll: &mut dyn ConstraintCollection = &mut s.perpendicular;
        let idx = coll.clone_push_blank(0);
        assert_eq!(idx, 1);
        assert_eq!(s.perpendicular[1].nid, 0);
        assert_eq!(s.perpendicular[1].cid, 0);
        assert_eq!(s.perpendicular[1].a, a);
        assert_eq!(s.perpendicular[1].b, b);
        assert_eq!(s.perpendicular[1].dir_sign, 1.0);
        // Original untouched.
        assert_eq!(s.perpendicular[0].nid, 7);
    }

    #[test]
    fn test_field_walk_matches_ref_walk_everywhere() {
        // each_line_field / each_line_ref are generated from the same
        // macro list, but this pins the contract for hand-written
        // impls that might appear later. Exercise via a sketch with a
        // few constraint kinds present.
        let (mut s, a, b) = two_line_sketch();
        let arc = s.add_arc(vect2d::new(2.0, 2.0), 1.0, 0.0, 1.0, false);
        s.tangent_la.push(crate::TangentLA {
            line: a, arc, sign: 1.0,
            p1_arc_start: false, p1_arc_end: false,
            p2_arc_start: false, p2_arc_end: false,
            dir_sign: f64::NAN, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.coincident_ll12.push(crate::CoincidentLL12 {
            a, b, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.for_each_constraint_collection(|_, _, coll| {
            for i in 0..coll.len() {
                let c = coll.item(i);
                let mut refs = 0;
                c.each_line_ref(&mut |_| refs += 1);
                let mut fields = 0;
                c.each_line_field(&mut |_, _, _| fields += 1);
                assert_eq!(refs, fields);
                let mut arefs = 0;
                c.each_arc_ref(&mut |_| arefs += 1);
                let mut afields = 0;
                c.each_arc_field(&mut |_, _, _| afields += 1);
                assert_eq!(arefs, afields);
            }
        });
    }
}

#[cfg(test)]
mod coincidence_pair_tests {
    use super::*;
    use arael::model::CrossBlock;
    use arael::vect::vect2d;

    #[test]
    fn test_decode_endpoint_roundtrip() {
        let mut s = Sketch::new();
        let p = s.add_point(vect2d::new(0.0, 0.0));
        let l = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0));
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, 1.0, false);
        assert_eq!(decode_endpoint(ep_point(p)), (EndpointRole::Point, p.index()));
        assert_eq!(decode_endpoint(ep_line_p1(l)), (EndpointRole::LineP1, l.index()));
        assert_eq!(decode_endpoint(ep_line_p2(l)), (EndpointRole::LineP2, l.index()));
        assert_eq!(decode_endpoint(ep_arc_center(a)), (EndpointRole::ArcCenter, a.index()));
        assert_eq!(decode_endpoint(ep_arc_start(a)), (EndpointRole::ArcStart, a.index()));
        assert_eq!(decode_endpoint(ep_arc_end(a)), (EndpointRole::ArcEnd, a.index()));
    }

    #[test]
    fn test_for_each_coincidence_pair_yields_only_coincidences() {
        let mut s = Sketch::new();
        let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0));
        let l1 = s.add_line(vect2d::new(1.0, 0.0), vect2d::new(2.0, 0.0));
        let a0 = s.add_arc(vect2d::new(2.0, 1.0), 1.0, 0.0, 1.0, false);
        s.coincident_ll21.push(crate::CoincidentLL21 {
            a: l0, b: l1, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.coincident_lp2_arc_start.push(crate::CoincidentLP2ArcStart {
            line: l1, arc: a0, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        // A non-coincidence constraint must not appear.
        s.parallel.push(crate::Parallel { a: l0, b: l1, nid: 0, cid: 0, hb: CrossBlock::new() });
        let mut pairs = Vec::new();
        s.for_each_coincidence_pair(|a, b| pairs.push((a, b)));
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&coincidence_pair(ep_line_p2(l0), ep_line_p1(l1))));
        assert!(pairs.contains(&coincidence_pair(ep_line_p2(l1), ep_arc_start(a0))));
    }

    // The (min, max) normalisation dedup applies; pairs arrive that way.
    fn coincidence_pair(a: u64, b: u64) -> (u64, u64) {
        (a.min(b), a.max(b))
    }
}

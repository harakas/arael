//! The drag apparatus: a gesture-scoped attachment that pulls a grab
//! target toward the cursor.
//!
//! Deliberately outside actions and history -- a gesture, not an edit.
//! The caller installs it at gesture start, moves the helper point(s)
//! and solves per frame, and removes it at gesture end; a caller that
//! restores a pre-gesture snapshot instead simply drops the token.
//! Removal is by identity (the helpers' refs), not pop order, so
//! anything the solve added or removed in between cannot desync it.

use arael::model::{CrossBlock, Param};
use arael::refs::Ref;
use arael::vect::vect2d;

use crate::{
    Arc, CoincidentArcCenter, CoincidentArcEnd, CoincidentArcStart, CoincidentLP1, CoincidentLP2,
    CoincidentPP, DragAutoAnchorState, Line, Point, Sketch,
};

/// What a drag has hold of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragTarget {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    /// Whole line: two helpers, one per endpoint.
    LineBody(Ref<Line>),
    /// Whole arc: helper on the center, shape locked for the gesture.
    ArcBody(Ref<Arc>),
}

/// Arc state saved while an ArcBody drag locks the shape.
#[derive(Clone, Copy, Debug)]
struct ArcLocks {
    had_radius: bool,
    old_radius: f64,
    had_radius_b: bool,
    old_radius_b: f64,
    rotation_optimize: bool,
    had_sweep: bool,
    old_sweep: f64,
    old_sweep_sign: f64,
    start_optimize: bool,
    end_optimize: bool,
}

/// Token for an installed drag apparatus. Pass it back to
/// [`Sketch::remove_drag`] at gesture end; drop it without removing
/// when the whole sketch is restored from a pre-gesture snapshot.
pub struct DragApparatus {
    pub target: DragTarget,
    /// Tracks the cursor (for LineBody: the p1-side helper).
    pub helper: Ref<Point>,
    /// The p2-side helper of a LineBody drag.
    pub helper2: Option<Ref<Point>>,
    arc_locks: Option<(Ref<Arc>, ArcLocks)>,
    anchors: DragAutoAnchorState,
}

impl Sketch {
    fn add_drag_point(&mut self, pos: vect2d, pull: Option<f64>) -> Ref<Point> {
        match pull {
            Some(w) => {
                let r = self.add_helper_point(pos);
                self.points[r].drag_pull = w;
                r
            }
            None => self.add_point_fixed(pos),
        }
    }

    /// Install the drag apparatus for `target`: helper point(s) at
    /// `pos` (and `pos2` for LineBody), bridged to the target with a
    /// coincidence, shape locks for ArcBody, plus the chain
    /// auto-anchors. `pull` Some(w) is the soft drag attractor weight;
    /// None hard-pins the helpers (raw drag).
    pub fn install_drag(
        &mut self,
        target: DragTarget,
        pos: vect2d,
        pos2: Option<vect2d>,
        pull: Option<f64>,
    ) -> DragApparatus {
        let helper = self.add_drag_point(pos, pull);
        let mut helper2 = None;
        let mut arc_locks = None;
        match target {
            DragTarget::Point(r) => {
                self.coincident_pp.push(CoincidentPP {
                    a: helper, b: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::LineP1(r) => {
                self.coincident_lp1.push(CoincidentLP1 {
                    line: r, point: helper, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::LineP2(r) => {
                self.coincident_lp2.push(CoincidentLP2 {
                    line: r, point: helper, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::ArcCenter(r) => {
                self.coincident_arc_center.push(CoincidentArcCenter {
                    point: helper, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::ArcStart(r) => {
                self.coincident_arc_start.push(CoincidentArcStart {
                    point: helper, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::ArcEnd(r) => {
                self.coincident_arc_end.push(CoincidentArcEnd {
                    point: helper, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            DragTarget::LineBody(r) => {
                self.coincident_lp1.push(CoincidentLP1 {
                    line: r, point: helper, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
                let h2 = self.add_drag_point(pos2.unwrap_or(pos), pull);
                self.coincident_lp2.push(CoincidentLP2 {
                    line: r, point: h2, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
                helper2 = Some(h2);
            }
            DragTarget::ArcBody(r) => {
                self.coincident_arc_center.push(CoincidentArcCenter {
                    point: helper, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
                let a = &self.arcs[r];
                arc_locks = Some((r, ArcLocks {
                    had_radius: a.constraints.has_target_radius,
                    old_radius: a.constraints.target_radius,
                    had_radius_b: a.constraints.has_target_radius_b,
                    old_radius_b: a.constraints.target_radius_b,
                    rotation_optimize: a.rotation.optimize,
                    had_sweep: a.constraints.has_target_sweep,
                    old_sweep: a.constraints.target_sweep,
                    old_sweep_sign: a.constraints.sweep_sign,
                    start_optimize: a.start_angle.optimize,
                    end_optimize: a.end_angle.optimize,
                }));
                let a = &mut self.arcs[r];
                a.constraints.has_target_radius = true;
                a.constraints.target_radius = a.radius.value;
                if a.is_ellipse {
                    a.constraints.has_target_radius_b = true;
                    a.constraints.target_radius_b = a.radius_b.value;
                    a.rotation.optimize = false;
                }
                a.constraints.has_target_sweep = true;
                // target_sweep is the positive sweep magnitude; sweep_sign
                // carries the direction. A signed delta would mismatch
                // sweep_sign on CW arcs and force radius to 0 to zero the
                // residual.
                a.constraints.target_sweep = (a.end_angle.value - a.start_angle.value).abs();
                a.constraints.sweep_sign = if a.ccw { 1.0 } else { -1.0 };
                a.start_angle.optimize = false;
                a.end_angle.optimize = false;
            }
        }
        // Chain auto-anchors last; removal takes them down first.
        let anchors = self.add_drag_auto_anchors();
        DragApparatus { target, helper, helper2, arc_locks, anchors }
    }

    /// Remove the apparatus: anchors first, then every constraint
    /// referencing the helpers -- by identity, via the registry --
    /// then the helpers themselves and the arc shape locks. Borrows
    /// the token: the same apparatus can be removed from a
    /// bincode-identical clone of the sketch (refs survive the round
    /// trip), which the GUI's best-cost snapshot path relies on.
    pub fn remove_drag(&mut self, apparatus: &DragApparatus) {
        self.remove_drag_auto_anchors(&apparatus.anchors);
        let h1 = apparatus.helper;
        let h2 = apparatus.helper2;
        // The bridges live in the collection the target dictates (see
        // install_drag); retain there instead of sweeping the whole
        // registry -- nothing else references a drag helper.
        match apparatus.target {
            DragTarget::Point(_) => self.coincident_pp.retain(|c| c.a != h1 && c.b != h1),
            DragTarget::LineP1(_) => self.coincident_lp1.retain(|c| c.point != h1),
            DragTarget::LineP2(_) => self.coincident_lp2.retain(|c| c.point != h1),
            DragTarget::ArcCenter(_) => self.coincident_arc_center.retain(|c| c.point != h1),
            DragTarget::ArcStart(_) => self.coincident_arc_start.retain(|c| c.point != h1),
            DragTarget::ArcEnd(_) => self.coincident_arc_end.retain(|c| c.point != h1),
            DragTarget::LineBody(_) => {
                self.coincident_lp1.retain(|c| c.point != h1);
                if let Some(h) = h2 {
                    self.coincident_lp2.retain(|c| c.point != h);
                }
            }
            DragTarget::ArcBody(_) => self.coincident_arc_center.retain(|c| c.point != h1),
        }
        if self.points.get(h1).is_some() {
            self.points.remove(h1);
        }
        if let Some(h) = h2
            && self.points.get(h).is_some() {
                self.points.remove(h);
        }
        if let Some((r, locks)) = apparatus.arc_locks {
            if let Some(a) = self.arcs.get_mut(r) {
                a.constraints.has_target_radius = locks.had_radius;
                a.constraints.target_radius = locks.old_radius;
                a.constraints.has_target_radius_b = locks.had_radius_b;
                a.constraints.target_radius_b = locks.old_radius_b;
                a.rotation.optimize = locks.rotation_optimize;
                a.constraints.has_target_sweep = locks.had_sweep;
                a.constraints.target_sweep = locks.old_sweep;
                a.constraints.sweep_sign = locks.old_sweep_sign;
                a.start_angle.optimize = locks.start_optimize;
                a.end_angle.optimize = locks.end_optimize;
            }
        }
    }

    /// Move a drag helper to a new position, preserving its mode --
    /// the per-frame update. `Param` assignment is deliberate for raw
    /// helpers (fixed stays fixed); soft helpers keep their index and
    /// only the value moves.
    pub fn move_drag_helper(&mut self, helper: Ref<Point>, pos: vect2d) {
        if let Some(p) = self.points.get_mut(helper) {
            if p.pos.index() == u32::MAX && !p.pos.optimize {
                p.pos = Param::fixed(pos);
            } else {
                p.pos.value = pos;
            }
        }
    }
}

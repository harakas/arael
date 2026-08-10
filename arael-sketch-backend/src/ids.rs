// Constraint identification and selection types used by commands/actions
// and the GUI. Carved out of the old arael-sketch/src/tools.rs so the
// backend crate can reference them without pulling in egui.

use arael::refs::Ref;
use arael_sketch_solver::*;

// Selection -- what entity is selected for constraint application.
// Held by CommandContext so command output and the GUI can share state.
#[derive(Clone, Copy, PartialEq)]
pub enum Selection {
    Point(Ref<Point>),
    Line(Ref<Line>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    Arc(Ref<Arc>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    Constraint(ConstraintId),
    /// By permanent dimension id (Dimension::did), not Vec index.
    Dimension(u32),
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintId {
    /// Line H flag constraint (name CL0H).
    Horizontal(Ref<Line>),
    /// Line V flag constraint (name CL0V).
    Vertical(Ref<Line>),
    /// Any collection-stored constraint, by its permanent nid (C<n>).
    /// Nids are unique and survive every mutation; positional indices
    /// do not.
    Numbered(u32),
    /// Hidden helper-point bridge (internal, no user-visible name).
    HelperBridge(Ref<Point>),
}

/// Format a ConstraintId as its user-visible name: `CL0H` /
/// `CL0V` for Horizontal/Vertical line flags, `C<nid>` for every
/// numbered constraint collection. Mirrors the name shown by `list`,
/// accepted by `delete`, and returned from `find_constraint_by_name`.
/// Command handlers use this to surface the IDs of constraints they
/// create or delete in their success messages.
pub fn constraint_id_name(sketch: &Sketch, id: ConstraintId) -> Option<String> {
    use arael_sketch_solver::format_flag_name;
    match id {
        ConstraintId::Horizontal(r) => Some(format_flag_name(&sketch.lines.get(r)?.name, 'H')),
        ConstraintId::Vertical(r) => Some(format_flag_name(&sketch.lines.get(r)?.name, 'V')),
        ConstraintId::Numbered(nid) => sketch.has_constraint_nid(nid).then(|| format!("C{}", nid)),
        // Helper bridges are internal plumbing -- no user-visible name.
        ConstraintId::HelperBridge(_) => None,
    }
}

/// Look up a constraint by its auto-assigned name.
///
/// Numeric names `C<n>` scan every Vec-stored constraint that has a
/// ConstraintId variant and return the first match. Synthetic flag
/// names like "CL0H" / "CL3V" parse the entity reference and validate
/// the flag is currently set on that line.
pub fn find_constraint_by_name(sketch: &Sketch, name: &str) -> Option<ConstraintId> {
    if let Some(rest) = name.strip_prefix('C')
        && let Ok(nid) = rest.parse::<u32>() {
        if nid == 0 { return None; }
        // Addressable when the nid exists in a collection that is not
        // dimension-backed (those delete through their d<n>) and is
        // not a hidden helper-point bridge -- deleting a bridge would
        // cascade the user's visible constraint away with the helper.
        let mut addressable = None;
        sketch.for_each_constraint_collection_ref(|arenas, meta, coll| {
            if addressable.is_some() { return; }
            for i in 0..coll.len() {
                let c = coll.item(i);
                if c.nid() != nid { continue; }
                let mut bridge = false;
                if meta.coincidence {
                    c.each_point_ref(&mut |p| {
                        bridge |= arenas.points.get(p).is_some_and(|pt| pt.helper);
                    });
                }
                addressable = Some(!meta.dimension_backed && !bridge);
                return;
            }
        });
        return match addressable {
            Some(true) => Some(ConstraintId::Numbered(nid)),
            _ => None,
        };
    }

    if let Some((entity, flag)) = arael_sketch_solver::parse_flag_name(name) {
        for r in sketch.lines.refs() {
            if sketch.lines[r].name == entity {
                let l = &sketch.lines[r];
                return match flag {
                    'H' if l.constraints.horizontal => Some(ConstraintId::Horizontal(r)),
                    'V' if l.constraints.vertical => Some(ConstraintId::Vertical(r)),
                    _ => None,
                };
            }
        }
    }
    None
}

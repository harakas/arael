//! Meta-constraints, the part every kind shares: the reconcile that drops
//! a record when what it owns was edited behind its back, name lookup,
//! dissolving and deleting, and the listing line. Each kind's engine
//! (`crate::offset`, ...) creates and edits its own records.

use arael_sketch_solver::*;

use crate::actions::Action;
use crate::corner_ops::ActionRunner;

pub(crate) fn nid_exists(sketch: &Sketch, nid: u32) -> bool {
    let mut found = false;
    sketch.for_each_constraint_collection_ref(|_, _, coll| {
        if found {
            return;
        }
        for i in 0..coll.len() {
            if coll.item(i).nid() == nid {
                found = true;
                return;
            }
        }
    });
    found
}

pub fn entity_exists(sketch: &Sketch, e: impl Into<MetaEntity>) -> bool {
    match e.into() {
        MetaEntity::Line(l) => sketch.lines.get(l).is_some(),
        MetaEntity::Arc(a) => sketch.arcs.get(a).is_some(),
        MetaEntity::Point(p) => sketch.points.get(p).is_some(),
    }
}

/// The entity's name.
pub fn entity_name(sketch: &Sketch, e: MetaEntity) -> String {
    match e {
        MetaEntity::Line(l) => sketch.lines[l].name.clone(),
        MetaEntity::Arc(a) => sketch.arcs[a].name.clone(),
        MetaEntity::Point(p) => sketch.points[p].name.clone(),
    }
}

/// A point-like reference as the user names it (`L0.p1`, `A0.center`,
/// `P3`); a helper point shows as what it is bridged to.
pub fn endpoint_name(sketch: &Sketch, e: &DimensionEndpoint) -> String {
    match e {
        DimensionEndpoint::Point(p) => sketch.point_display_name(*p),
        DimensionEndpoint::LineP1(l) => format!("{}.p1", sketch.lines[*l].name),
        DimensionEndpoint::LineP2(l) => format!("{}.p2", sketch.lines[*l].name),
        DimensionEndpoint::ArcCenter(a) => format!("{}.center", sketch.arcs[*a].name),
        DimensionEndpoint::ArcStart(a) => format!("{}.start", sketch.arcs[*a].name),
        DimensionEndpoint::ArcEnd(a) => format!("{}.end", sketch.arcs[*a].name),
    }
}

/// The action deleting the entity; its relations cascade.
pub fn delete_action(e: MetaEntity) -> Action {
    match e {
        MetaEntity::Line(l) => Action::DeleteLine { line: l },
        MetaEntity::Arc(a) => Action::DeleteArc { arc: a },
        MetaEntity::Point(p) => Action::DeletePoint { point: p },
    }
}

/// Delete the entity; its relations cascade.
pub fn delete_entity(runner: &mut dyn ActionRunner, e: MetaEntity) {
    runner.run(delete_action(e));
}

/// Why a record no longer holds, if it does not: a source or result
/// entity gone, an owned constraint gone, an owned dimension gone, made
/// derived, or carrying a different value / expression than was written.
/// Soft-owned constraints are not checked.
fn broken_reason(sketch: &Sketch, m: &Meta) -> Option<String> {
    if m.source_entities().iter().any(|e| !entity_exists(sketch, *e)) {
        return Some("a source entity was deleted".into());
    }
    if m.owned_entities().iter().any(|e| !entity_exists(sketch, *e)) {
        return Some("a result entity was deleted".into());
    }
    for nid in m.owned_constraints() {
        if !nid_exists(sketch, nid) {
            return Some(format!("C{} was deleted", nid));
        }
    }
    for d in m.owned_dims() {
        let Some(i) = sketch.dimension_index_by_did(d.did) else {
            return Some("a dimension was deleted".into());
        };
        let dim = &sketch.dimensions[i];
        if dim.derived {
            return Some(format!("{} was made derived", dim.name));
        }
        let same = match (&dim.expr_str, &d.expect.expr) {
            (Some(a), Some(b)) => a == b,
            (None, None) => (dim.value - d.expect.value).abs() <= 1e-9 * (1.0 + d.expect.value.abs()),
            _ => false,
        };
        if !same {
            return Some(format!("{} was edited", dim.name));
        }
    }
    None
}

/// Drop every meta-constraint whose result, relations or dimensions were
/// changed behind its back, with a notice each. Runs after every action.
pub fn reconcile(sketch: &mut Sketch) {
    let mut dropped: Vec<(u32, String)> = Vec::new();
    for m in &sketch.metas {
        if let Some(why) = broken_reason(sketch, m) {
            dropped.push((m.mid, format!("{} {} dropped: {}", m.kind_name(), m.name, why)));
        }
    }
    for (mid, msg) in dropped {
        sketch.metas.retain(|m| m.mid != mid);
        sketch.push_notice(msg);
    }
}

/// The meta-constraint named `name` (`M<n>`).
pub fn resolve(sketch: &Sketch, name: &str) -> Result<usize, String> {
    sketch.find_meta(name).ok_or_else(|| format!("Unknown meta-constraint: {}", name))
}

/// The meta-constraint that owns `e` as a result, if any.
pub fn owner_of(sketch: &Sketch, e: impl Into<MetaEntity>) -> Option<&Meta> {
    let e = e.into();
    sketch.metas.iter().find(|m| m.owns_entity(e))
}

/// Forget the record; the result stays as plain geometry.
pub fn dissolve(runner: &mut dyn ActionRunner, mid: u32) -> Result<(), String> {
    if runner.sketch().meta_index(mid).is_none() {
        return Err(format!("no meta-constraint M{}", mid));
    }
    runner.begin_group();
    runner.run(Action::UnregisterMeta { mid });
    Ok(())
}

/// Forget the record and delete its result entities (their relations
/// cascade). Returns the names deleted.
pub fn delete_with_result(runner: &mut dyn ActionRunner, mid: u32) -> Result<Vec<String>, String> {
    let Some(i) = runner.sketch().meta_index(mid) else {
        return Err(format!("no meta-constraint M{}", mid));
    };
    let owned = runner.sketch().metas[i].owned_entities();
    runner.begin_group();
    runner.run(Action::UnregisterMeta { mid });
    let mut names = Vec::new();
    let mut deletes = Vec::new();
    for e in owned {
        if !entity_exists(runner.sketch(), e) {
            continue;
        }
        // Helper points are invisible: not named, deleted all the same.
        if !matches!(e, MetaEntity::Point(p) if runner.sketch().points[p].helper) {
            names.push(entity_name(runner.sketch(), e));
        }
        deletes.push(delete_action(e));
    }
    // One step for all of them: a pattern can own hundreds.
    if !deletes.is_empty() {
        runner.run(Action::Batch { label: "Delete meta-constraint result".into(), actions: deletes });
    }
    Ok(names)
}

/// One line describing a meta-constraint, for `list metas` / `info`.
pub fn describe(sketch: &Sketch, m: &Meta) -> String {
    match &m.kind {
        MetaKind::Offset(o) => format!("{}: {}", m.name, crate::offset::describe(sketch, o)),
        MetaKind::Pattern(p) => format!("{}: {}", m.name, crate::pattern::describe(sketch, p)),
    }
}

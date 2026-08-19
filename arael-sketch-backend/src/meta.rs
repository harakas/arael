//! Meta-constraints, the part every kind shares: the reconcile that drops
//! a record when what it owns was edited behind its back, name lookup,
//! dissolving and deleting, and the listing line. Each kind's engine
//! (`crate::offset`, ...) creates and edits its own records.

use arael_sketch_solver::*;

use crate::actions::Action;
use crate::chain;
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

pub fn entity_exists(sketch: &Sketch, e: OffsetEntity) -> bool {
    match e {
        OffsetEntity::Line(l) => sketch.lines.get(l).is_some(),
        OffsetEntity::Arc(a) => sketch.arcs.get(a).is_some(),
    }
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
pub fn owner_of(sketch: &Sketch, e: OffsetEntity) -> Option<&Meta> {
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
    for e in owned {
        if !entity_exists(runner.sketch(), e) {
            continue;
        }
        names.push(chain::entity_name(runner.sketch(), e));
        match e {
            OffsetEntity::Line(l) => { runner.run(Action::DeleteLine { line: l }); }
            OffsetEntity::Arc(a) => { runner.run(Action::DeleteArc { arc: a }); }
        }
    }
    Ok(names)
}

/// One line describing a meta-constraint, for `list metas` / `info`.
pub fn describe(sketch: &Sketch, m: &Meta) -> String {
    match &m.kind {
        MetaKind::Offset(o) => format!("{}: {}", m.name, crate::offset::describe(sketch, o)),
    }
}

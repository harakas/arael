//! Mirror engine: copy a set of lines, arcs and points across an axis
//! line, recreate the set's coincidences among the copies, and hold
//! every copy to its source with symmetry constraints. The command
//! (`mirror ... about L`) and the GUI Mirror tool both run through
//! [`plan`] and [`apply`].

use std::f64::consts::TAU;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use crate::actions::{Action, Created};
use crate::corner_ops::ActionRunner;
use crate::geometry::{arc_start_pos, arc_end_pos};

/// Reflect a point across a line defined by two points.
pub fn mirror_point_across(pt: vect2d, lp1: vect2d, lp2: vect2d) -> vect2d {
    let dx = lp2.x - lp1.x;
    let dy = lp2.y - lp1.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-24 { return pt; }
    let t = ((pt.x - lp1.x) * dx + (pt.y - lp1.y) * dy) / len2;
    let proj = vect2d::new(lp1.x + t * dx, lp1.y + t * dy);
    vect2d::new(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y)
}

#[derive(Clone, Debug)]
pub struct MirrorParams {
    pub axis: Ref<Line>,
    /// Skip the coincident and symmetry constraints: bare copies.
    pub noconstraint: bool,
    /// Error (and roll the whole mirror back) if the result cannot
    /// satisfy all constraints; default is a warning.
    pub strict: bool,
}

/// The mirrored geometry of one source, ready for a creation action.
#[derive(Clone, Copy, Debug)]
enum MirrorGeom {
    Line { p1: vect2d, p2: vect2d },
    Point { pos: vect2d },
    Circle { center: vect2d, edge: vect2d },
    Arc { start: vect2d, end: vect2d, mid: vect2d },
}

/// Everything [`apply`] needs, computed without touching the sketch.
/// Ties and symmetry entries are in source-slot space; apply maps them
/// onto the copies. The structural decision walks the set in order
/// over the source's coincidence graph: the first slot of every
/// coincidence group gets a symmetry constraint, every later slot a
/// recreated coincident to it.
#[derive(Clone, Debug)]
pub struct MirrorPlan {
    pub sources: Vec<MetaEntity>,
    pub params: MirrorParams,
    geoms: Vec<MirrorGeom>,
    /// (this slot, the group's already-placed slot), both source-side.
    ties: Vec<(DimensionEndpoint, DimensionEndpoint)>,
    /// One per coincidence group: the group's first source slot.
    syms: Vec<DimensionEndpoint>,
    /// Source indices of closed circles whose center group was not
    /// already placed: one arc symmetry holds center and radius.
    sym_aa: Vec<usize>,
    /// Source indices of closed arcs whose center ties to an already-
    /// placed slot: the copy keeps the radius with an equal-radius
    /// constraint (open arcs' endpoint symmetries imply it).
    eq_radius: Vec<usize>,
}

/// What a mirror created, for the command output and the status line.
#[derive(Clone, Debug, Default)]
pub struct MirrorOutcome {
    /// Per source, in order: (source name, copy name).
    pub mirrored: Vec<(String, String)>,
    /// The created copies, in source order.
    pub copies: Vec<MetaEntity>,
    /// Applied constraint descriptions; the last line lists the ids.
    pub applied: Vec<String>,
    pub warnings: Vec<String>,
}

/// Plan a mirror: the copies' geometry, the coincidences to recreate
/// and the deduped symmetry set. The axis is dropped from the set
/// silently (it mirrors onto its own position); an empty set refuses.
pub fn plan(sketch: &Sketch, sources: &[MetaEntity], params: &MirrorParams) -> Result<MirrorPlan, String> {
    let axis = sketch.lines.get(params.axis).ok_or("mirror axis line does not exist")?;
    let (mlp1, mlp2) = (axis.p1.value, axis.p2.value);
    let sources: Vec<MetaEntity> = sources.iter().copied()
        .filter(|e| !matches!(e, MetaEntity::Line(l) if *l == params.axis))
        .collect();
    if sources.is_empty() {
        return Err("No lines, arcs, or points to mirror".into());
    }

    let mut geoms = Vec::with_capacity(sources.len());
    for e in &sources {
        geoms.push(match *e {
            MetaEntity::Line(r) => {
                let l = sketch.lines.get(r).ok_or("mirror source line does not exist")?;
                MirrorGeom::Line {
                    p1: mirror_point_across(l.p1.value, mlp1, mlp2),
                    p2: mirror_point_across(l.p2.value, mlp1, mlp2),
                }
            }
            MetaEntity::Point(r) => {
                let p = sketch.points.get(r).ok_or("mirror source point does not exist")?;
                MirrorGeom::Point { pos: mirror_point_across(p.pos.value, mlp1, mlp2) }
            }
            MetaEntity::Arc(r) => {
                let a = sketch.arcs.get(r).ok_or("mirror source arc does not exist")?;
                let mc = mirror_point_across(a.center.value, mlp1, mlp2);
                let rad = a.radius.value;
                if a.closed {
                    MirrorGeom::Circle { center: mc, edge: vect2d::new(mc.x + rad, mc.y) }
                } else {
                    let mid_angle = (a.start_angle.value + a.end_angle.value) / 2.0;
                    let mid_pt = vect2d::new(
                        a.center.value.x + rad * mid_angle.cos(),
                        a.center.value.y + rad * mid_angle.sin(),
                    );
                    MirrorGeom::Arc {
                        start: mirror_point_across(arc_start_pos(a), mlp1, mlp2),
                        end: mirror_point_across(arc_end_pos(a), mlp1, mlp2),
                        mid: mirror_point_across(mid_pt, mlp1, mlp2),
                    }
                }
            }
        });
    }

    // The structural decision, walking the set in order over the
    // source's coincidence graph (every coincident flavor, concentric
    // included): the first slot of each group is pinned by symmetry,
    // every later slot ties to it with a recreated coincident.
    let mut ties: Vec<(DimensionEndpoint, DimensionEndpoint)> = Vec::new();
    let mut syms: Vec<DimensionEndpoint> = Vec::new();
    let mut sym_aa: Vec<usize> = Vec::new();
    let mut eq_radius: Vec<usize> = Vec::new();
    if !params.noconstraint {
        let mut groups = crate::coincide::CoincidenceGroups::build(sketch);
        let mut placed: std::collections::HashMap<usize, DimensionEndpoint> =
            std::collections::HashMap::new();
        let slot = |groups: &mut crate::coincide::CoincidenceGroups,
                        placed: &mut std::collections::HashMap<usize, DimensionEndpoint>,
                        ties: &mut Vec<(DimensionEndpoint, DimensionEndpoint)>,
                        syms: &mut Vec<DimensionEndpoint>,
                        ep: DimensionEndpoint| {
            let id = match ep {
                DimensionEndpoint::Point(p) => groups.pt(p),
                DimensionEndpoint::LineP1(l) => groups.lp1(l),
                DimensionEndpoint::LineP2(l) => groups.lp2(l),
                DimensionEndpoint::ArcCenter(a) => groups.arc_center(a),
                DimensionEndpoint::ArcStart(a) => groups.arc_start(a),
                DimensionEndpoint::ArcEnd(a) => groups.arc_end(a),
            };
            let root = groups.find(id);
            match placed.get(&root) {
                Some(&other) => ties.push((ep, other)),
                None => {
                    placed.insert(root, ep);
                    syms.push(ep);
                }
            }
        };
        for (i, e) in sources.iter().enumerate() {
            match *e {
                MetaEntity::Line(r) => {
                    slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::LineP1(r));
                    slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::LineP2(r));
                }
                MetaEntity::Point(r) => {
                    slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::Point(r));
                }
                MetaEntity::Arc(r) => {
                    let a = &sketch.arcs[r];
                    if a.closed && !a.is_ellipse {
                        // An untied circle takes one arc symmetry
                        // (center + radius); a tied one keeps the
                        // coincident and holds the radius separately.
                        let root = groups.find(groups.arc_center(r));
                        match placed.get(&root) {
                            Some(&other) => {
                                ties.push((DimensionEndpoint::ArcCenter(r), other));
                                eq_radius.push(i);
                            }
                            None => {
                                placed.insert(root, DimensionEndpoint::ArcCenter(r));
                                sym_aa.push(i);
                            }
                        }
                    } else {
                        slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::ArcCenter(r));
                        if a.closed {
                            // Closed ellipse: the copy is created
                            // circular, so only the radius is held.
                            eq_radius.push(i);
                        } else {
                            slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::ArcStart(r));
                            slot(&mut groups, &mut placed, &mut ties, &mut syms, DimensionEndpoint::ArcEnd(r));
                        }
                    }
                }
            }
        }
    }

    Ok(MirrorPlan { sources, params: params.clone(), geoms, ties, syms, sym_aa, eq_radius })
}

/// The planned copies as polylines for the preview: lines as their two
/// endpoints, points as one, arcs sampled with `n` segments (as the
/// circular arc creation makes).
pub fn preview_polylines(sketch: &Sketch, mplan: &MirrorPlan, n: usize) -> Vec<Vec<vect2d>> {
    let Some(axis) = sketch.lines.get(mplan.params.axis) else { return Vec::new() };
    let (mlp1, mlp2) = (axis.p1.value, axis.p2.value);
    let mut out = Vec::with_capacity(mplan.sources.len());
    for (e, g) in mplan.sources.iter().zip(&mplan.geoms) {
        out.push(match (*e, *g) {
            (_, MirrorGeom::Line { p1, p2 }) => vec![p1, p2],
            (_, MirrorGeom::Point { pos }) => vec![pos],
            (MetaEntity::Arc(r), _) => {
                let Some(a) = sketch.arcs.get(r) else { continue };
                let (s, e) = if a.closed {
                    (0.0, TAU)
                } else {
                    let (sa, ea) = (a.start_angle.value, a.end_angle.value);
                    let sweep = if a.ccw {
                        (ea - sa).rem_euclid(TAU)
                    } else {
                        -((sa - ea).rem_euclid(TAU))
                    };
                    (sa, sa + sweep)
                };
                let (c, rad) = (a.center.value, a.radius.value);
                (0..=n)
                    .map(|i| {
                        let t = s + (e - s) * i as f64 / n as f64;
                        mirror_point_across(
                            vect2d::new(c.x + rad * t.cos(), c.y + rad * t.sin()),
                            mlp1, mlp2)
                    })
                    .collect()
            }
            _ => continue,
        });
    }
    out
}

/// Create the planned mirror as one undo group: the entities as one
/// batch, then the coincidents and symmetry constraints as another --
/// the mirrored geometry satisfies them exactly, so they skip the
/// per-constraint gate like a pattern's image constraints. A failure
/// (or a strict cost-check failure) rolls the whole group back.
pub fn apply(runner: &mut dyn ActionRunner, mplan: &MirrorPlan) -> Result<MirrorOutcome, String> {
    let old_cost = runner.sketch_mut().current_cost();
    runner.begin_group();
    match apply_inner(runner, mplan, old_cost) {
        Ok(out) => {
            runner.end_group();
            Ok(out)
        }
        Err(e) => {
            runner.rollback_group();
            Err(e)
        }
    }
}

fn apply_inner(runner: &mut dyn ActionRunner, mplan: &MirrorPlan, old_cost: f64) -> Result<MirrorOutcome, String> {
    let mut out = MirrorOutcome::default();

    // Phase 1: the entities.
    let creates: Vec<Action> = mplan.geoms.iter().map(|g| match *g {
        MirrorGeom::Line { p1, p2 } => Action::AddLine { p1, p2 },
        MirrorGeom::Point { pos } => Action::AddPoint { pos },
        MirrorGeom::Circle { center, edge } => Action::AddCircle { center, edge },
        MirrorGeom::Arc { start, end, mid } => Action::AddArc { start, end, mid },
    }).collect();
    let created = runner.run(Action::Batch { label: "Mirror".into(), actions: creates });
    if let Some(e) = runner.take_error() {
        return Err(e);
    }
    let Created::Many(created) = created else {
        return Err("Internal: creation batch added nothing".into());
    };
    let mut copies: Vec<MetaEntity> = Vec::with_capacity(mplan.sources.len());
    for (src, c) in mplan.sources.iter().zip(created) {
        let sketch = runner.sketch();
        let (copy, src_name, dst_name) = match (*src, c) {
            (MetaEntity::Line(s), Created::Line(d)) =>
                (MetaEntity::Line(d), sketch.lines[s].name.clone(), sketch.lines[d].name.clone()),
            (MetaEntity::Point(s), Created::Point(d)) =>
                (MetaEntity::Point(d), sketch.points[s].name.clone(), sketch.points[d].name.clone()),
            (MetaEntity::Arc(s), Created::Arc(d)) =>
                (MetaEntity::Arc(d), sketch.arcs[s].name.clone(), sketch.arcs[d].name.clone()),
            (MetaEntity::Arc(_), _) =>
                return Err("Cannot mirror arc: degenerate mirrored geometry".into()),
            _ => return Err("Internal: creation action added no entity".into()),
        };
        copies.push(copy);
        out.mirrored.push((src_name, dst_name));
    }
    out.copies = copies.clone();

    // Phase 2: recreated coincidents first (they merge the free
    // copies), then one symmetry per coincidence group, then the
    // closed arcs' radii.
    let mut acts: Vec<Action> = Vec::new();
    let mut descs: Vec<String> = Vec::new();
    {
        let sketch = runner.sketch();
        let ep_name = |ep: DimensionEndpoint| -> String {
            match ep {
                DimensionEndpoint::LineP1(l) => format!("{}.p1", sketch.lines[l].name),
                DimensionEndpoint::LineP2(l) => format!("{}.p2", sketch.lines[l].name),
                DimensionEndpoint::Point(p) => sketch.points[p].name.clone(),
                DimensionEndpoint::ArcCenter(a) => format!("{}.center", sketch.arcs[a].name),
                DimensionEndpoint::ArcStart(a) => format!("{}.start", sketch.arcs[a].name),
                DimensionEndpoint::ArcEnd(a) => format!("{}.end", sketch.arcs[a].name),
            }
        };
        let map = |ep: DimensionEndpoint| -> Result<DimensionEndpoint, String> {
            crate::pattern::map_endpoint(ep, &mplan.sources, &copies)
                .ok_or_else(|| "Internal: mirror tie outside the source set".to_string())
        };
        for &(a, b) in &mplan.ties {
            let (ca, cb) = (map(a)?, map(b)?);
            let action = match Action::coincident(ca, cb) {
                Some(act) => act,
                // Two circle centers in one group (concentric source).
                None => match (ca, cb) {
                    (DimensionEndpoint::ArcCenter(x), DimensionEndpoint::ArcCenter(y)) =>
                        Action::ApplyConcentric { a: x, b: y },
                    _ => return Err("Internal: no coincident action for a mirror tie".into()),
                },
            };
            let joiner = if matches!(
                (ca, cb),
                (DimensionEndpoint::LineP1(_) | DimensionEndpoint::LineP2(_),
                 DimensionEndpoint::LineP1(_) | DimensionEndpoint::LineP2(_))
            ) { "=" } else { " " };
            descs.push(format!("coincident {}{}{}", ep_name(ca), joiner, ep_name(cb)));
            acts.push(action);
        }
        let axis_name = sketch.lines[mplan.params.axis].name.clone();
        for &ep in &mplan.syms {
            let cp = map(ep)?;
            acts.push(Action::ApplySymmetryPP { a: ep, line: mplan.params.axis, c: cp });
            descs.push(format!("symmetry {} {} {}", ep_name(ep), axis_name, ep_name(cp)));
        }
        for &i in &mplan.sym_aa {
            let (MetaEntity::Arc(src), MetaEntity::Arc(dst)) = (mplan.sources[i], copies[i]) else {
                return Err("Internal: arc-symmetry entry on a non-arc".into());
            };
            acts.push(Action::ApplySymmetryAA { a: src, line: mplan.params.axis, c: dst });
            descs.push(format!("symmetry {} {} {}",
                sketch.arcs[src].name, axis_name, sketch.arcs[dst].name));
        }
        for &i in &mplan.eq_radius {
            let (MetaEntity::Arc(src), MetaEntity::Arc(dst)) = (mplan.sources[i], copies[i]) else {
                return Err("Internal: equal-radius entry on a non-arc".into());
            };
            acts.push(Action::ApplyEqualRadius { a: src, b: dst });
            descs.push(format!("equal radius {} {}", sketch.arcs[src].name, sketch.arcs[dst].name));
        }
    }
        let watermark = runner.sketch().next_constraint_id;
    if !acts.is_empty() {
        runner.run(Action::Batch { label: "Mirror constraints".into(), actions: acts });
        if let Some(e) = runner.take_error() {
            return Err(e);
        }
        out.applied = descs;
        let names = constraint_names_since(runner.sketch(), watermark);
        if !names.is_empty() {
            out.applied.push(format!("constraints: {}", names.join(" ")));
        }
    }

    // The per-constraint cost gate is gone with the batch; one
    // whole-operation check replaces it. A mirror is exact, so a cost
    // jump means a constraint could not be satisfied.
    let quick = runner.sketch_mut().current_cost();
    let new_cost = if quick <= old_cost + 1e-6 { quick } else { runner.sketch_mut().solve().end_cost };
    if new_cost > old_cost + 1e-3 {
        let m = format!("mirror could not satisfy all constraints (cost {:.3e} -> {:.3e})", old_cost, new_cost);
        if mplan.params.strict {
            return Err(m);
        }
        out.warnings.push(m);
    }
    Ok(out)
}

/// The user-visible `C<n>` names minted at or after `watermark`.
fn constraint_names_since(sketch: &Sketch, watermark: u32) -> Vec<String> {
    let mut nids: Vec<u32> = sketch.constraint_nid_cid_pairs().into_iter()
        .map(|(nid, _)| nid).filter(|&n| n >= watermark).collect();
    nids.sort_unstable();
    nids.dedup();
    nids.into_iter().map(|n| format!("C{}", n)).collect()
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

    /// Counts group lifecycle calls; optionally refuses the n-th batch.
    struct Recorder<'a> {
        inner: &'a mut CommandContext,
        begins: usize,
        ends: usize,
        rollbacks: usize,
        fail_batch: Option<usize>,
        seen: usize,
        failed: bool,
    }

    impl ActionRunner for Recorder<'_> {
        fn sketch(&self) -> &Sketch { self.inner.sketch() }
        fn sketch_mut(&mut self) -> &mut Sketch { self.inner.sketch_mut() }
        fn run(&mut self, action: Action) -> Created {
            if matches!(action, Action::Batch { .. }) {
                self.seen += 1;
                if Some(self.seen) == self.fail_batch {
                    self.failed = true;
                    return Created::Nothing;
                }
            }
            self.inner.run(action)
        }
        fn run_unchecked(&mut self, action: Action) -> Created { self.inner.run_unchecked(action) }
        fn take_error(&mut self) -> Option<String> {
            if std::mem::take(&mut self.failed) { Some("refused for the test".into()) } else { self.inner.take_error() }
        }
        fn begin_group(&mut self) { self.begins += 1; self.inner.begin_group() }
        fn end_group(&mut self) { self.ends += 1; self.inner.end_group() }
        fn rollback_group(&mut self) { self.rollbacks += 1; self.inner.rollback_group() }
    }

    fn line_plan(ctx: &CommandContext) -> MirrorPlan {
        let sources = vec![MetaEntity::Line(ctx.sketch.lines.refs().next().unwrap())];
        let axis = ctx.sketch.lines.refs().nth(1).unwrap();
        plan(&ctx.sketch, &sources, &MirrorParams { axis, noconstraint: false, strict: false }).unwrap()
    }

    #[test]
    fn apply_ends_the_group_on_success() {
        let mut ctx = ctx("add_line 0,0 2,0; add_line 5,-5 5,5 noconnect nocursor");
        let p = line_plan(&ctx);
        let mut rec = Recorder { inner: &mut ctx, begins: 0, ends: 0, rollbacks: 0, fail_batch: None, seen: 0, failed: false };
        let out = apply(&mut rec, &p).unwrap();
        assert_eq!((rec.begins, rec.ends, rec.rollbacks), (1, 1, 0));
        assert_eq!(out.mirrored.len(), 1);
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn failed_apply_rolls_back_without_end() {
        let mut ctx = ctx("add_line 0,0 2,0; add_line 5,-5 5,5 noconnect nocursor");
        let p = line_plan(&ctx);
        let mut rec = Recorder { inner: &mut ctx, begins: 0, ends: 0, rollbacks: 0, fail_batch: Some(2), seen: 0, failed: false };
        assert!(apply(&mut rec, &p).is_err());
        assert_eq!((rec.begins, rec.ends, rec.rollbacks), (1, 0, 1));
        assert_eq!(rec.inner.sketch.lines.refs().count(), 2, "rolled back to the sources");
    }

    #[test]
    fn plan_drops_the_axis_and_refuses_empty() {
        let ctx = ctx("add_line 0,0 2,0; add_line 5,-5 5,5 noconnect nocursor");
        let axis = ctx.sketch.lines.refs().nth(1).unwrap();
        let params = MirrorParams { axis, noconstraint: false, strict: false };
        let p = plan(&ctx.sketch, &[MetaEntity::Line(ctx.sketch.lines.refs().next().unwrap()), MetaEntity::Line(axis)], &params).unwrap();
        assert_eq!(p.sources.len(), 1, "the axis is dropped from the set");
        assert!(plan(&ctx.sketch, &[MetaEntity::Line(axis)], &params).is_err());
    }

    #[test]
    fn preview_mirrors_lines_points_and_arcs() {
        let ctx = ctx("add_line 0,0 2,0; add_point 1,3; add_circle 0,2 1; add_line 5,-5 5,5 noconnect nocursor");
        let axis = ctx.sketch.lines.refs().nth(1).unwrap();
        let sources = vec![
            MetaEntity::Line(ctx.sketch.lines.refs().next().unwrap()),
            MetaEntity::Point(ctx.sketch.points.refs().next().unwrap()),
            MetaEntity::Arc(ctx.sketch.arcs.refs().next().unwrap()),
        ];
        let p = plan(&ctx.sketch, &sources, &MirrorParams { axis, noconstraint: false, strict: false }).unwrap();
        let polys = preview_polylines(&ctx.sketch, &p, 16);
        assert_eq!(polys.len(), 3);
        let near = |p: vect2d, x: f64, y: f64| (p.x - x).abs() < 1e-9 && (p.y - y).abs() < 1e-9;
        assert_eq!(polys[0].len(), 2);
        assert!(near(polys[0][0], 10.0, 0.0) && near(polys[0][1], 8.0, 0.0), "{:?}", polys[0]);
        assert_eq!(polys[1].len(), 1);
        assert!(near(polys[1][0], 9.0, 3.0), "{:?}", polys[1]);
        assert_eq!(polys[2].len(), 17, "circle sampled with n segments");
        for pt in &polys[2] {
            let d = ((pt.x - 10.0).powi(2) + (pt.y - 2.0).powi(2)).sqrt();
            assert!((d - 1.0).abs() < 1e-9, "on the mirrored circle: {:?}", pt);
        }
    }
}

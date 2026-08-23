//! Candidate-row builders for constraint probes.
//!
//! Each function returns the Jacobian row(s) the constraint WOULD
//! contribute at the current geometry, up to overall scale -- the
//! direction is what a span test needs. Tested against
//! `arael::rank::RankResult::reduces_rank`, this answers "would adding
//! this constraint reduce DOF" in microseconds, without mutating the
//! sketch or recomputing rank.
//!
//! The rows mirror the constraint bodies in constraints.rs /
//! entities.rs exactly, including the normalisation-term derivatives
//! (the probes run at generic geometry where those terms are not
//! small). The guarded flip-barrier rows are omitted: at the moment a
//! constraint would be applied its dir_sign is captured from the
//! current geometry, which leaves the barrier inactive. Fixed
//! parameters carry index u32::MAX and are skipped by the span test.
//! Row-direction equivalence against the macro-generated Jacobian is
//! pinned by tests/probe.rs.

use crate::{Arc, Line};
use arael::vect::vect2d;

fn push(row: &mut Vec<(u32, f64)>, base: u32, offset: u32, v: f64) {
    if base != u32::MAX && v != 0.0 {
        row.push((base + offset, v));
    }
}

/// A position and the sparse derivative of its coordinates over the
/// params: `x` holds d(pos.x)/d(param), `y` holds d(pos.y)/d(param).
/// Fixed params are omitted.
pub struct PosJac {
    pub pos: vect2d,
    pub x: Vec<(u32, f64)>,
    pub y: Vec<(u32, f64)>,
}

/// Which anchor of an arc a probe position refers to.
#[derive(Clone, Copy)]
pub enum ArcAnchor { Center, Start, End, Mid }

/// Position held directly by a 2D param (point, line endpoint).
pub fn point_pos(index: u32, value: vect2d) -> PosJac {
    let mut x = Vec::with_capacity(1);
    let mut y = Vec::with_capacity(1);
    push(&mut x, index, 0, 1.0);
    push(&mut y, index, 1, 1.0);
    PosJac { pos: value, x, y }
}

/// Midpoint of a line body.
pub fn line_midpoint_pos(l: &Line) -> PosJac {
    let mut x = Vec::with_capacity(2);
    let mut y = Vec::with_capacity(2);
    push(&mut x, l.p1.index(), 0, 0.5);
    push(&mut x, l.p2.index(), 0, 0.5);
    push(&mut y, l.p1.index(), 1, 0.5);
    push(&mut y, l.p2.index(), 1, 0.5);
    PosJac {
        pos: vect2d::new(
            (l.p1.value.x + l.p2.value.x) * 0.5,
            (l.p1.value.y + l.p2.value.y) * 0.5,
        ),
        x, y,
    }
}

/// An arc anchor position: center, or an ellipse point at the start,
/// end, or halfway parametric angle, differentiated over center,
/// radius, radius_b, rotation, and the angle param(s).
pub fn arc_anchor_pos(a: &Arc, which: ArcAnchor) -> PosJac {
    let mut x = Vec::new();
    let mut y = Vec::new();
    push(&mut x, a.center.index(), 0, 1.0);
    push(&mut y, a.center.index(), 1, 1.0);
    if matches!(which, ArcAnchor::Center) {
        return PosJac { pos: a.center.value, x, y };
    }
    let t = match which {
        ArcAnchor::Start => a.start_angle.value,
        ArcAnchor::End => a.end_angle.value,
        ArcAnchor::Mid => (a.start_angle.value + a.end_angle.value) * 0.5,
        ArcAnchor::Center => unreachable!(),
    };
    let (r, rb, rot) = (a.radius.value, a.radius_b.value, a.rotation.value);
    let (st, ct) = t.sin_cos();
    let (sr, cr) = rot.sin_cos();
    // pos = center + Rot(rot) * (r cos t, rb sin t)
    push(&mut x, a.radius.index(), 0, ct * cr);
    push(&mut y, a.radius.index(), 0, ct * sr);
    push(&mut x, a.radius_b.index(), 0, -st * sr);
    push(&mut y, a.radius_b.index(), 0, st * cr);
    push(&mut x, a.rotation.index(), 0, -r * ct * sr - rb * st * cr);
    push(&mut y, a.rotation.index(), 0, r * ct * cr - rb * st * sr);
    let dt_x = -r * st * cr - rb * ct * sr;
    let dt_y = -r * st * sr + rb * ct * cr;
    match which {
        ArcAnchor::Start => {
            push(&mut x, a.start_angle.index(), 0, dt_x);
            push(&mut y, a.start_angle.index(), 0, dt_y);
        }
        ArcAnchor::End => {
            push(&mut x, a.end_angle.index(), 0, dt_x);
            push(&mut y, a.end_angle.index(), 0, dt_y);
        }
        ArcAnchor::Mid => {
            push(&mut x, a.start_angle.index(), 0, dt_x * 0.5);
            push(&mut y, a.start_angle.index(), 0, dt_y * 0.5);
            push(&mut x, a.end_angle.index(), 0, dt_x * 0.5);
            push(&mut y, a.end_angle.index(), 0, dt_y * 0.5);
        }
        ArcAnchor::Center => unreachable!(),
    }
    PosJac {
        pos: vect2d::new(
            a.center.value.x + r * ct * cr - rb * st * sr,
            a.center.value.y + r * ct * sr + rb * st * cr,
        ),
        x, y,
    }
}

// a - b per coordinate, entries merged by param index.
fn diff_row(a: &[(u32, f64)], b: &[(u32, f64)]) -> Vec<(u32, f64)> {
    let mut map = std::collections::BTreeMap::new();
    for &(i, v) in a {
        *map.entry(i).or_insert(0.0) += v;
    }
    for &(i, v) in b {
        *map.entry(i).or_insert(0.0) -= v;
    }
    map.into_iter().filter(|&(_, v)| v != 0.0).collect()
}

/// The two rows of a coincidence between two probe positions:
/// `a.x - b.x` and `a.y - b.y`.
pub fn coincident_rows(a: &PosJac, b: &PosJac) -> [Vec<(u32, f64)>; 2] {
    [diff_row(&a.x, &b.x), diff_row(&a.y, &b.y)]
}

/// Row of a point-on-line constraint: probe position `q` against the
/// infinite line of `host`, `((q - p1) x d) / |d|`.
pub fn on_line_row(host: &Line, q: &PosJac) -> Vec<(u32, f64)> {
    let dx = host.p2.value.x - host.p1.value.x;
    let dy = host.p2.value.y - host.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-300);
    let ux = q.pos.x - host.p1.value.x;
    let uy = q.pos.y - host.p1.value.y;
    let f = ux * dy - uy * dx;
    // r = f/len; dr = df/len - f/len^2 * dlen
    let c = 1.0 / len;
    let s = f / (len * len);
    let mut row = Vec::with_capacity(6 + q.x.len() + q.y.len());
    push(&mut row, host.p1.index(), 0, (-dy + uy) * c + s * dx / len);
    push(&mut row, host.p1.index(), 1, (dx - ux) * c + s * dy / len);
    push(&mut row, host.p2.index(), 0, -uy * c - s * dx / len);
    push(&mut row, host.p2.index(), 1, ux * c - s * dy / len);
    for &(i, v) in &q.x {
        row.push((i, v * dy * c));
    }
    for &(i, v) in &q.y {
        row.push((i, -v * dx * c));
    }
    row
}

/// Row of a point-on-arc constraint: probe position `q` against the
/// implicit ellipse `u^2/r^2 + v^2/rb^2 - 1` of `host`, where (u, v)
/// is `q - center` in the ellipse frame.
pub fn on_arc_row(host: &Arc, q: &PosJac) -> Vec<(u32, f64)> {
    let (r, rb, rot) = (
        host.radius.value.max(1e-300),
        host.radius_b.value.max(1e-300),
        host.rotation.value,
    );
    let (sr, cr) = rot.sin_cos();
    let px = q.pos.x - host.center.value.x;
    let py = q.pos.y - host.center.value.y;
    let u = px * cr + py * sr;
    let v = -px * sr + py * cr;
    let fu = 2.0 * u / (r * r);
    let fv = 2.0 * v / (rb * rb);
    let gx = fu * cr - fv * sr; // dF/d(q.x)
    let gy = fu * sr + fv * cr; // dF/d(q.y)
    let mut row = Vec::with_capacity(5 + q.x.len() + q.y.len());
    push(&mut row, host.center.index(), 0, -gx);
    push(&mut row, host.center.index(), 1, -gy);
    push(&mut row, host.radius.index(), 0, -2.0 * u * u / (r * r * r));
    push(&mut row, host.radius_b.index(), 0, -2.0 * v * v / (rb * rb * rb));
    push(&mut row, host.rotation.index(), 0, 2.0 * u * v * (1.0 / (r * r) - 1.0 / (rb * rb)));
    for &(i, vv) in &q.x {
        row.push((i, vv * gx));
    }
    for &(i, vv) in &q.y {
        row.push((i, vv * gy));
    }
    row
}

/// Row of the `horizontal` self-constraint: `p1.y - p2.y`.
pub fn horizontal_row(l: &Line) -> Vec<(u32, f64)> {
    let mut row = Vec::with_capacity(2);
    push(&mut row, l.p1.index(), 1, 1.0);
    push(&mut row, l.p2.index(), 1, -1.0);
    row
}

/// Row of the `vertical` self-constraint: `p1.x - p2.x`.
pub fn vertical_row(l: &Line) -> Vec<(u32, f64)> {
    let mut row = Vec::with_capacity(2);
    push(&mut row, l.p1.index(), 0, 1.0);
    push(&mut row, l.p2.index(), 0, -1.0);
    row
}

/// Main row of `Perpendicular`: `(d1 . d2) / mlen`,
/// `mlen = (|d1| + |d2|) / 2`.
pub fn perpendicular_row(a: &Line, b: &Line) -> Vec<(u32, f64)> {
    let dx1 = a.p2.value.x - a.p1.value.x;
    let dy1 = a.p2.value.y - a.p1.value.y;
    let dx2 = b.p2.value.x - b.p1.value.x;
    let dy2 = b.p2.value.y - b.p1.value.y;
    let len1 = (dx1 * dx1 + dy1 * dy1).sqrt().max(1e-300);
    let len2 = (dx2 * dx2 + dy2 * dy2).sqrt().max(1e-300);
    let mlen = (len1 + len2) / 2.0;
    let g = dx1 * dx2 + dy1 * dy2;
    // d(g/mlen) = dg/mlen - g/mlen^2 * dmlen
    let c = 1.0 / mlen;
    let q = g / (mlen * mlen) * 0.5;
    let mut row = Vec::with_capacity(8);
    push(&mut row, a.p1.index(), 0, -dx2 * c + q * dx1 / len1);
    push(&mut row, a.p1.index(), 1, -dy2 * c + q * dy1 / len1);
    push(&mut row, a.p2.index(), 0, dx2 * c - q * dx1 / len1);
    push(&mut row, a.p2.index(), 1, dy2 * c - q * dy1 / len1);
    push(&mut row, b.p1.index(), 0, -dx1 * c + q * dx2 / len2);
    push(&mut row, b.p1.index(), 1, -dy1 * c + q * dy2 / len2);
    push(&mut row, b.p2.index(), 0, dx1 * c - q * dx2 / len2);
    push(&mut row, b.p2.index(), 1, dy1 * c - q * dy2 / len2);
    row
}

/// The two rows of `Collinear`: each endpoint of `b` against the
/// infinite line of `a`, `((q - a.p1) x d) / |d|`.
pub fn collinear_rows(a: &Line, b: &Line) -> [Vec<(u32, f64)>; 2] {
    [
        on_line_row(a, &point_pos(b.p1.index(), b.p1.value)),
        on_line_row(a, &point_pos(b.p2.index(), b.p2.value)),
    ]
}

/// True when any of the candidate rows would reduce DOF.
pub fn any_reduces_rank(rank: &arael::rank::RankResult, rows: &[Vec<(u32, f64)>]) -> bool {
    rows.iter().any(|r| rank.reduces_rank(r))
}

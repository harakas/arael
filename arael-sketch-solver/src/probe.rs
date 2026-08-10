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

use crate::Line;

fn push(row: &mut Vec<(u32, f64)>, base: u32, offset: u32, v: f64) {
    if base != u32::MAX && v != 0.0 {
        row.push((base + offset, v));
    }
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
    let dx = a.p2.value.x - a.p1.value.x;
    let dy = a.p2.value.y - a.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-300);
    let one_row = |qx: f64, qy: f64, q_idx: u32| -> Vec<(u32, f64)> {
        let ux = qx - a.p1.value.x;
        let uy = qy - a.p1.value.y;
        let f = ux * dy - uy * dx;
        // r = f/len; dr = df/len - f/len^2 * dlen
        let c = 1.0 / len;
        let s = f / (len * len);
        let mut row = Vec::with_capacity(6);
        push(&mut row, a.p1.index(), 0, (-dy + uy) * c + s * dx / len);
        push(&mut row, a.p1.index(), 1, (dx - ux) * c + s * dy / len);
        push(&mut row, a.p2.index(), 0, -uy * c - s * dx / len);
        push(&mut row, a.p2.index(), 1, ux * c - s * dy / len);
        push(&mut row, q_idx, 0, dy * c);
        push(&mut row, q_idx, 1, -dx * c);
        row
    };
    [
        one_row(b.p1.value.x, b.p1.value.y, b.p1.index()),
        one_row(b.p2.value.x, b.p2.value.y, b.p2.index()),
    ]
}

/// True when any of the candidate rows would reduce DOF.
pub fn any_reduces_rank(rank: &arael::rank::RankResult, rows: &[Vec<(u32, f64)>]) -> bool {
    rows.iter().any(|r| rank.reduces_rank(r))
}

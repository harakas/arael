//! Multi-root fixture: two roots at different precisions in one
//! crate. The export puts both shims in one capi crate (each in its
//! own module; symbols are root-prefixed) and each root's header in a
//! nested namespace (`cxx_mr::line` / `cxx_mr::decay`), so one C++
//! translation unit can use both. The entities are generic over the
//! float and each root instantiates them at its precision, so the
//! export spells them concretely per root.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::utils::Float;

/// Root A (f64): fit y = k * x through the observations.
#[arael::model]
#[arael(constraint(root.hb, {
    [ob.y - line.k * ob.x]
}))]
#[derive(Default)]
pub struct Ob<T: Float> {
    pub x: T,
    pub y: T,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Line {
    pub k: Param<f64>,
    pub obs: std::vec::Vec<Ob<f64>>,
    pub hb: SelfBlock<Line>,
}

/// Root B (f32): per-cell value pulled to its target against a weak
/// zero prior, so the optimum is a nonzero-cost compromise.
#[arael::model]
#[arael(constraint(hb, {
    [(cell.v - cell.t) * cell.w,
     cell.v * 0.5]
}))]
#[derive(Default)]
pub struct Cell<T: Float> {
    pub v: Param<T>,
    pub t: T,
    pub w: T,
    pub hb: SelfBlock<Cell<T>, T>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Default)]
pub struct Decay {
    pub cells: refs::Vec<Cell<f32>>,
}

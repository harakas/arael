//! The cxx-tests fixture model: a scalar-only shape covering the
//! cargo-arael skeleton -- root params behind `root.hb` observations
//! in a std::vec::Vec, own-param entities in a refs::Vec, data fields,
//! and an opaque field. Kept small; the parity test builds the same
//! problem from C++ and from Rust and compares the solves exactly.

use arael::model::{Param, SelfBlock};
use arael::refs;

/// A data-only observation of the root's line: y ~ m * x + c.
#[arael::model]
#[arael(constraint(root.hb, {
    [obs.y - (fit.m * obs.x + fit.c)]
}))]
#[derive(Default)]
pub struct Obs {
    pub x: f64,
    pub y: f64,
}

/// An entity with its own parameter pulled to its target.
#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * n.w]
}))]
#[derive(Default)]
pub struct N {
    pub v: Param<f64>,
    pub t: f64,
    pub w: f64,
    pub hb: SelfBlock<N>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Fit {
    pub m: Param<f64>,
    pub c: Param<f64>,
    pub tag: String,
    pub obs: std::vec::Vec<Obs>,
    pub items: refs::Vec<N>,
    pub hb: SelfBlock<Fit>,
}

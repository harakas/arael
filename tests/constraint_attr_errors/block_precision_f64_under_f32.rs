//! A default (f64) block under an `#[arael(root, f32)]` root: the
//! defaulted block scalar counts as concrete f64 and must match.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * 0.3]
}))]
struct N {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(root, f32)]
struct W {
    nodes: refs::Vec<N>,
}

fn main() {}

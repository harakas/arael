//! `[parent.hb, coo]`: the companion of `coo` is the constraint struct's
//! own SelfBlock, not a `root.` or `parent.` block.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint([parent.hb, coo], { [(obs.y - curve.m * obs.x - root.b) * 0.5] }))]
struct Obs {
    x: f64,
    y: f64,
}

#[arael::model]
struct Curve {
    m: Param<f64>,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root)]
struct W {
    b: Param<f64>,
    curves: refs::Vec<Curve>,
    hb: SelfBlock<W>,
}

fn main() {}

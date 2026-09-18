//! A `parent.<field>` secondary is no longer a way to place cross
//! pairs: they go to COO, and the error says so.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([hb, parent.nosuch], {
    [obs.y - (curve.m * obs.x + obs.o)]
}))]
struct Obs {
    x: f64,
    y: f64,
    o: Param<f64>,
    hb: SelfBlock<Obs>,
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
    curves: std::vec::Vec<Curve>,
}

fn main() {}

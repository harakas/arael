//! `[hb, parent.<triplet>]` with the entity held directly by the root:
//! the containing parent IS the root, and the form is `coo` either way.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([hb, parent.hbt], {
    [obs.y - (w.m * obs.x + obs.o)]
}))]
struct Obs {
    x: f64,
    y: f64,
    o: Param<f64>,
    hb: SelfBlock<Obs>,
}

#[arael::model]
#[arael(root)]
struct W {
    m: Param<f64>,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<W>,
}

fn main() {}

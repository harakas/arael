//! A `parent.<selfblock>` constraint entity with its own Params would
//! drop the (entity, parent) cross pairs -- rejected.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(parent.hb, {
    [obs.w * obs.y - curve.m * obs.x]
}))]
struct Obs {
    w: Param<f64>,
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
struct Fit {
    curves: refs::Vec<Curve>,
}

fn main() {}

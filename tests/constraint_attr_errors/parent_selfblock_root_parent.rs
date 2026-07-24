//! `parent.<selfblock>` where the containing parent IS the root:
//! rejected with a pointer to the `root.<selfblock>` spelling.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(parent.hb, {
    [obs.y - fit.m * obs.x]
}))]
struct Obs {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(root)]
struct Fit {
    m: Param<f64>,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<Fit>,
}

fn main() {}

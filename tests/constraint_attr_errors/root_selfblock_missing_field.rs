//! A `root.<field>` primary block must name a field that exists on the root.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(root.nosuch, { [e.y - root.a * e.x] }))]
struct E {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(root)]
struct Fit {
    a: Param<f64>,
    hb: SelfBlock<Fit>,
    data: std::vec::Vec<E>,
}

fn main() {}

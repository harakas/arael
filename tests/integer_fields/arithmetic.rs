// An integer field read as a number in a constraint body.
use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(hb, { [p.x * p.k - p.t] }))]
struct P {
    x: Param<f64>,
    k: u32,
    t: f64,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(root)]
struct R {
    ps: std::vec::Vec<P>,
}

fn main() {}

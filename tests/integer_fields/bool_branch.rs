// A bool field as a `branch` condition: a `match` picks the arm instead.
use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(hb, { [branch(q.on, q.x - 1.0, q.x - 2.0)] }))]
struct Q {
    x: Param<f64>,
    on: bool,
    hb: SelfBlock<Q>,
}

#[arael::model]
#[arael(root)]
struct R {
    qs: std::vec::Vec<Q>,
}

fn main() {}

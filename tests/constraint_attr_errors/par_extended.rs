//! The extended hooks write the model's own blocks, which a threaded
//! solve never reads: `par` and `extended` cannot combine.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(hb, { [p.x - 1.0] }))]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(root, par, extended)]
struct R {
    ps: arael::refs::Vec<P>,
}

fn main() {}

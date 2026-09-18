//! `coo` names the solve's COO list in a constraint, so a block field
//! cannot be called that.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(coo, { [(n.x - n.t) * 2.0] }))]
struct N {
    x: Param<f64>,
    t: f64,
    coo: SelfBlock<N>,
}

#[arael::model]
#[arael(root)]
struct W {
    items: std::vec::Vec<N>,
}

fn main() {}

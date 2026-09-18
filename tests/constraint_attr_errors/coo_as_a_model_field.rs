//! A `Coo` is the solve's, not the model's: declaring one as a field is
//! rejected, with the keyword and the hook named.

use arael::model::{Coo, Param, SelfBlock};

#[arael::model]
#[arael(constraint(hb, { [(n.x - n.t) * 2.0] }))]
struct N {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<N>,
    spare: Coo<f64>,
}

#[arael::model]
#[arael(root)]
struct W {
    items: std::vec::Vec<N>,
}

fn main() {}

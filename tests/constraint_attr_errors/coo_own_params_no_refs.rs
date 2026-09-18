//! `constraint(coo, ..)` on a struct with params of its own and no refs:
//! the N-ary form couples the entities the refs point at, so this body's
//! params would have no block. Used to compile and contribute nothing.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(coo, { [(c.v - 1.0) * 0.3] }))]
struct C {
    v: Param<f64>,
    hb: SelfBlock<C>,
}

#[arael::model]
#[arael(root)]
struct W {
    xs: std::vec::Vec<C>,
}

fn main() {}

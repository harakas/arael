//! `[hb, coo]` with a body that reads no co-entity params: there are no
//! cross pairs for COO to hold.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([hb, coo], { [(e.x - e.t) * 2.0] }))]
struct E {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<E>,
}

#[arael::model]
#[arael(root)]
struct W {
    b: Param<f64>,
    items: std::vec::Vec<E>,
    hb: SelfBlock<W>,
}

fn main() {}

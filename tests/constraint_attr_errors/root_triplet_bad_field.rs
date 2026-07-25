//! `[hb, root.<field>]` must name a TripletBlock field on the root --
//! the (entity, root) cross pairs live there.

use arael::model::{Param, SelfBlock, TripletBlock};

#[arael::model]
#[arael(constraint([hb, root.nosuch], {
    [e.x - w.b - e.t]
}))]
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
    hbt: TripletBlock<f64>,
}

fn main() {}

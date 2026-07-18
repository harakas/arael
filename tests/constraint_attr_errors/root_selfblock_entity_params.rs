//! A `root.<selfblock>` constraint may touch only root params: an entity
//! with its own Param fields would form (entity, root) cross pairs the
//! root's SelfBlock cannot hold, and dropping them is never acceptable.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(root.hb, { [e.y - root.a * e.x - e.w] }))]
struct E {
    x: f64,
    y: f64,
    w: Param<f64>,
    hb: SelfBlock<E>,
}

#[arael::model]
#[arael(root)]
struct Fit {
    a: Param<f64>,
    hb: SelfBlock<Fit>,
    data: std::vec::Vec<E>,
}

fn main() {}

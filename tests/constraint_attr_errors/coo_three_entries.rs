//! `[hb, coo, root.hb]`: `coo` takes one companion, the entity's own
//! SelfBlock.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([hb, coo, root.hb], { [(e.x - root.b) * 2.0] }))]
struct E {
    x: Param<f64>,
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

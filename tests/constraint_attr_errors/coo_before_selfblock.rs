//! `[coo, hb]`: the COO list goes second, behind the entity's SelfBlock.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint([coo, hb], { [(e.x - root.b) * 2.0] }))]
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

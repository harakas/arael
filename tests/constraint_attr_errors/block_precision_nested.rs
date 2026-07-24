//! The precision check walks the containment tree transitively: an f32
//! block two levels below the root is still named.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(nf.v - nf.t) * 0.3]
}))]
struct Nf {
    v: Param<f32>,
    t: f32,
    hb: SelfBlock<Nf, f32>,
}

#[arael::model]
struct Group {
    nodes: refs::Vec<Nf>,
}

#[arael::model]
#[arael(root)]
struct W {
    groups: std::vec::Vec<Group>,
}

fn main() {}

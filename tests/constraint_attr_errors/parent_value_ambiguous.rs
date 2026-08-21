//! `parent.` in an entity type held under several containment paths:
//! "the parent" is ambiguous.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(gnode.x - gnode.t) * parent.pw]
}))]
struct GNode {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<GNode>,
}

#[arael::model]
struct PodA {
    pw: f64,
    nodes: refs::Arena<GNode>,
}

#[arael::model]
struct PodB {
    pw: f64,
    nodes: refs::Arena<GNode>,
}

#[arael::model]
#[arael(root)]
struct Net {
    pods_a: std::vec::Vec<PodA>,
    pods_b: std::vec::Vec<PodB>,
}

fn main() {}

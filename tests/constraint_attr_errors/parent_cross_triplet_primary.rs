//! `parent.<triplet>` in the PRIMARY slot: a parent-owned TripletBlock
//! is a secondary block; the error names the `[hb, parent.<field>]` form.

use arael::model::{Param, SelfBlock, TripletBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    [node.x - node.d]
}))]
struct Node {
    x: Param<f64>,
    d: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(parent.hbt, {
    [b.x - a.x - slink.d]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
}

#[arael::model]
struct Pair {
    links: std::vec::Vec<SLink>,
    hbt: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

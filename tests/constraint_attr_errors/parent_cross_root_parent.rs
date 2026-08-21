//! `parent.<crossblock>` with the constraint held directly by the root:
//! the containing parent IS the root.

use arael::model::{CrossBlock, Param, SelfBlock};
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
#[arael(constraint(parent.hb, {
    [b.x - a.x - slink.d]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<SLink>,
    hb: CrossBlock<Node, Node>,
}

fn main() {}

//! A `parent.<crossblock>` constraint struct with its own Param fields:
//! their cross pairs against the referenced entities would be dropped.

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
    [b.x - a.x - slink.bias]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    bias: Param<f64>,
    hb_self: SelfBlock<SLink>,
}

#[arael::model]
struct Pair {
    links: std::vec::Vec<SLink>,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

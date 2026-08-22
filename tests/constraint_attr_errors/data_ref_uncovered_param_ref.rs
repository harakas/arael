//! D4: a third ref to a PARAM-BEARING type that the block declaration
//! does not cover -- its params would lose cross pairs, so it errors.

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
#[arael(constraint(hb, {
    [b.x - a.x - link.d + c.x]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = root.nodes)] c: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<Link>,
}

fn main() {}

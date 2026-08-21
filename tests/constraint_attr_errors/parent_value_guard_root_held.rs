//! `parent.` in the GUARD of a constraint held directly by the root.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    [node.x - node.t]
}))]
struct Node {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, guard = parent.enabled, {
    [b.x - a.x - link.d]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    enabled: bool,
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<Link>,
}

fn main() {}

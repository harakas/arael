//! Parent-refs form with Param fields on the parent: the parent of a
//! shared CrossBlock is a plain container.

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
    [parent.b.x - parent.a.x - plink.d]
}))]
struct PLink {
    d: f64,
}

#[arael::model]
struct Pair {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    scale: Param<f64>,
    links: std::vec::Vec<PLink>,
    hb: CrossBlock<Node, Node>,
    hb_self: SelfBlock<Pair>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

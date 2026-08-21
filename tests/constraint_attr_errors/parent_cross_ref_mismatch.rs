//! The constraint's Ref fields must be exactly [Ref<A>, Ref<B>] of the
//! parent's CrossBlock<A, B>, in declaration order.

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
struct Other {
    y: Param<f64>,
    hb: SelfBlock<Other>,
}

#[arael::model]
#[arael(constraint(parent.hb, {
    [n.x - o.y - slink.d]
}))]
struct SLink {
    #[arael(ref = root.others)] o: Ref<Other>,
    #[arael(ref = root.nodes)] n: Ref<Node>,
    d: f64,
}

#[arael::model]
struct Pair {
    links: std::vec::Vec<SLink>,
    hb: CrossBlock<Node, Other>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    others: refs::Arena<Other>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

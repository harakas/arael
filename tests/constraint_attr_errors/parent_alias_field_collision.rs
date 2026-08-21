//! `parent = <name>` colliding with a field of the constraint struct.

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
#[arael(constraint(parent.hb, parent = d, {
    [parent.b.x - parent.a.x - plink.d]
}))]
struct PLink {
    d: f64,
}

#[arael::model]
struct Pair {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    links: std::vec::Vec<PLink>,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

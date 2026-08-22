//! D2: a data-ref path chaining through a parent ref outside the
//! parent-refs form (the constraint holds its own entity refs).

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
struct Tag {
    off: f64,
}

#[arael::model]
#[arael(constraint(hb, {
    [node.x - node.d]
}))]
struct Node {
    x: Param<f64>,
    d: f64,
    tags: refs::Arena<Tag>,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(parent.hb, {
    [b.x - a.x - slink.d + t.off]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = parent.n.tags)] t: Ref<Tag>,
    d: f64,
}

#[arael::model]
struct Pair {
    #[arael(ref = root.nodes)] n: Ref<Node>,
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

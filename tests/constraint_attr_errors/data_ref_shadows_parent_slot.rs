//! A data-ref field named like a parent ref of the parent-refs form
//! would shadow the bound entity local -- rejected.

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
    [parent.b.x - parent.a.x - plink.d + a.off]
}))]
struct PLink {
    #[arael(ref = parent.a.tags)] a: Ref<Tag>,
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

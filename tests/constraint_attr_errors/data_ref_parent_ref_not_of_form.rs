//! D1: a data-ref path `parent.<x>.<coll>` where `<x>` is a ref field
//! of the parent but not one the parent-refs form binds.

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
    [parent.b.x - parent.a.x - plink.d + t.off]
}))]
struct PLink {
    #[arael(ref = parent.c.tags)] t: Ref<Tag>,
    d: f64,
}

#[arael::model]
struct Pair {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = root.nodes)] c: Ref<Node>,
    links: std::vec::Vec<PLink>,
    #[arael(cross = (a, b))]
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

//! D3: a resolve path whose head is neither `root`, `parent`, nor a
//! Ref field of the constraint struct.

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
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - link.d + t.off]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = nosuch.tags)] t: Ref<Tag>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    tags: refs::Arena<Tag>,
    links: std::vec::Vec<Link>,
}

fn main() {}

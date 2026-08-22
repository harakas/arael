//! D5: a resolve path ending on a plain field -- a ref indexes a
//! collection, so this errors naming the field and the rule.

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
    #[arael(ref = root.isigma)] t: Ref<Tag>,
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    isigma: f64,
    tags: refs::Arena<Tag>,
    links: std::vec::Vec<Link>,
}

fn main() {}

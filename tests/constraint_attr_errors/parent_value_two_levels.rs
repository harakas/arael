//! `parent.parent.` -- one level only.

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
#[arael(constraint(hb, {
    [(b.x - a.x - glink.d) * parent.parent.scale]
}))]
struct GLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
struct Group {
    isigma: f64,
    links: std::vec::Vec<GLink>,
}

#[arael::model]
struct Region {
    scale: f64,
    groups: std::vec::Vec<Group>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    regions: std::vec::Vec<Region>,
}

fn main() {}

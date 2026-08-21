//! Reading a parent Param through the data-only `parent.` binding:
//! its derivative pairs would be dropped; the error names the coupling
//! forms that hold them.

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
    [(b.x - a.x - glink.d) * parent.scale]
}))]
struct GLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
struct Group {
    scale: Param<f64>,
    links: std::vec::Vec<GLink>,
    hb: SelfBlock<Group>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    groups: std::vec::Vec<Group>,
}

fn main() {}

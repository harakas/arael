//! `parent.` in a constraint held directly by the root: no containing
//! parent; the error points at `root.<field>`.

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
    [(b.x - a.x - link.d) * parent.isigma]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    isigma: f64,
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<Link>,
}

fn main() {}

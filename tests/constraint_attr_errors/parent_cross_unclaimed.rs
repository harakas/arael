//! A CrossBlock declared on a plain struct that no constraint claims
//! via `parent.<field>`: it would sit inert, looking like a wired
//! accumulator.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs;

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
struct Pair {
    w: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

fn main() {}

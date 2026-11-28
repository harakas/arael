//! A CrossBlock that no constraint writes is rejected, on a constraint
//! struct as on a plain one: `spare` is in no block list.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::Ref;

#[arael::model]
#[arael(constraint(hb, { [(n.x - n.t) * 2.0] }))]
struct N {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(constraint(hb, { [(b.x - a.x - link.d) * 0.5] }))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<N>,
    #[arael(ref = root.nodes)] b: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
    spare: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: arael::refs::Vec<N>,
    links: std::vec::Vec<Link>,
}

fn main() {}

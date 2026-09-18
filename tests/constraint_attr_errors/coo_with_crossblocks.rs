//! `coo` cannot share a block list with CrossBlock fields: name the
//! blocks whose pairs are tiled, or `coo` alone for all of them.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, { [(n.x - n.t) * 2.0] }))]
struct N {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(constraint([hb_ab, coo], { [(a.x + b.x + c.x - tri.s) * 0.5] }))]
struct Tri {
    #[arael(ref = root.nodes)] a: Ref<N>,
    #[arael(ref = root.nodes)] b: Ref<N>,
    #[arael(ref = root.nodes)] c: Ref<N>,
    s: f64,
    #[arael(cross = (a, b))] hb_ab: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: refs::Vec<N>,
    tris: std::vec::Vec<Tri>,
}

fn main() {}

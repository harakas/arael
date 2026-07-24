//! A generic model instantiated at f32 under an f64 root: the
//! instantiation sets the block precision, so the holding field's
//! spelling is checked against the root.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(g.v - g.t) * 0.3]
}))]
struct G<T: arael::utils::Float> {
    v: Param<T>,
    t: T,
    hb: SelfBlock<G<T>, T>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: refs::Vec<G<f32>>,
}

fn main() {}

//! A body path starting at an unknown identifier -- here `tie.d` where
//! the self alias is the lowercased struct name `tiev` -- used to be
//! spliced verbatim into generated code and fail with a bare rustc
//! "cannot find value" pointing at the macro. Now a macro error listing
//! the available bindings.

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};

#[arael::model]
struct N {
    v: Param<f64>,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - tie.d) * 1.5]
}))]
struct TieV {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: refs::Vec<N>,
    ties: std::vec::Vec<TieV>,
}

fn main() {}

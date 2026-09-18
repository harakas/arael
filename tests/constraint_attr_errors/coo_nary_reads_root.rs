//! An N-ary `coo` constraint whose body reads a root param: the root joins
//! an entity only through `[hb, coo]`, and this struct has no SelfBlock.

use arael::model::{Param, SelfBlock};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, { [(n.x - n.p) * 0.1] }))]
struct N {
    x: Param<f64>,
    p: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(constraint(coo, { [(d.x - root.s) * 0.3] }))]
struct D {
    #[arael(ref = root.ns)] d: Ref<N>,
}

#[arael::model]
#[arael(root)]
struct W {
    s: Param<f64>,
    ns: refs::Vec<N>,
    xs: std::vec::Vec<D>,
    hb: SelfBlock<W>,
}

fn main() {}

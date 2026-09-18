//! `[coo, coo]`: the COO list is one place; used to compile and drop the
//! root coupling the body asked for.

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
#[arael(constraint([coo, coo], { [(b.x - root.s) * 0.3] }))]
struct B {
    #[arael(ref = root.ns)] b: Ref<N>,
}

#[arael::model]
#[arael(root)]
struct W {
    s: Param<f64>,
    ns: refs::Vec<N>,
    xs: std::vec::Vec<B>,
    hb: SelfBlock<W>,
}

fn main() {}

//! `constraint(coo, ..)` on a struct with refs AND params of its own, whose
//! body reads the latter: the N-ary form has no block for them.

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
#[arael(constraint(coo, { [(n.x - dv.v) * 0.3] }))]
struct Dv {
    #[arael(ref = root.ns)] n: Ref<N>,
    v: Param<f64>,
    hb: SelfBlock<Dv>,
}

#[arael::model]
#[arael(root)]
struct W {
    ns: refs::Vec<N>,
    xs: std::vec::Vec<Dv>,
}

fn main() {}

//! A TripletBlock has no per-thread copy: a `par` root holding one
//! anywhere is rejected.

use arael::model::{Param, SelfBlock, TripletBlock};
use arael::refs::Ref;

#[arael::model]
#[arael(constraint(hb, { [p.x - 1.0] }))]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(constraint(hb, { [a.x + b.x + c.x] }))]
struct T {
    #[arael(ref = root.ps)] a: Ref<P>,
    #[arael(ref = root.ps)] b: Ref<P>,
    #[arael(ref = root.ps)] c: Ref<P>,
    hb: TripletBlock<f64>,
}

#[arael::model]
#[arael(root, par)]
struct R {
    ps: arael::refs::Vec<P>,
    ts: std::vec::Vec<T>,
}

fn main() {}

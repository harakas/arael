//! `cross = (...)` requires exactly two ref field names.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::Ref;

#[arael::model]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.x - b.x)]
}))]
struct C {
    #[arael(ref = root.ps)]
    a: Ref<P>,
    #[arael(ref = root.ps)]
    b: Ref<P>,
    #[arael(cross = (a))]
    hb: CrossBlock<P, P>,
}

fn main() {}

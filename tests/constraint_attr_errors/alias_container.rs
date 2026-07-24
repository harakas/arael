//! `use ... as` alias on a container: containers are recognized by
//! literal last path segment, so `RVec<P>` would read as an opaque data
//! field and every constraint on `P` under it would be silently dropped.
//! An unrecognized wrapper holding a registered model type is rejected.

use arael::model::{Param, SelfBlock};
use arael::refs::Vec as RVec;

#[arael::model]
#[arael(constraint(hb, {
    [(p.v - p.t) * 0.5]
}))]
struct P {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(root)]
struct W {
    ps: RVec<P>,
}

fn main() {}

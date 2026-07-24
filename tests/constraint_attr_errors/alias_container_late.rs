//! The same alias hole with the holder expanded BEFORE the held type:
//! at the holder's expansion `P` is not yet registered, so the suspect
//! wrapper is recorded and fires when `P` turns out to be a model type.

use arael::model::{Param, SelfBlock};
use arael::refs::Vec as RVec;

#[arael::model]
struct Holder {
    ps: RVec<P>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(p.v - p.t) * 0.5]
}))]
struct P {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<P>,
}

fn main() {}

//! A cross/triplet constraint struct held as a single instance (direct
//! field here) has no collection to iterate -- its sweep used to be
//! silently skipped; now rejected at expansion.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{Ref, Vec as RVec};

#[arael::model]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(constraint(hb, { [(a.x - b.x) * 2.0] }))]
struct Tie {
    #[arael(ref = root.points)]
    a: Ref<P>,
    #[arael(ref = root.points)]
    b: Ref<P>,
    hb: CrossBlock<P, P>,
}

#[arael::model]
#[arael(root)]
struct W {
    points: RVec<P>,
    the_tie: Tie,
}

fn main() {}

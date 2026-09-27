//! Under `rayon` a root is read by every sweep thread at once, so it must
//! be `Sync` unless it says `seq`; the error names the way out.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, { [(n.x - n.p) * 0.1] }))]
struct N {
    x: Param<f64>,
    p: f64,
    hb: SelfBlock<N>,
}

#[arael::model]
#[arael(root)]
struct W {
    ns: refs::Vec<N>,
    #[arael(skip)]
    hits: std::cell::Cell<u32>,
}

fn main() {}

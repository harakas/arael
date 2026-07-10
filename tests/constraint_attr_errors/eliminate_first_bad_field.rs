//! eliminate_first must name a parameter-bearing field of the root
//! struct -- a typo or a plain-data field name is a hard error, not a
//! silently empty hint.

#[allow(unused_imports)]
use arael::model::{Model, Param, SelfBlock};

#[arael::model]
#[arael(root, eliminate_first(landmark))]
struct R {
    x: Param<f64>,
    landmarks: arael::refs::Vec<L>,
    hb: SelfBlock<R>,
}

#[arael::model]
struct L {
    p: Param<f64>,
    hb: SelfBlock<L>,
}

fn main() {}

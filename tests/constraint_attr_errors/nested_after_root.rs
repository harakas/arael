//! A NESTED holder's collection element defined after the root: the
//! syntax-side ordering guard only sees root-level fields, so this used
//! to be a silent drop (params serialized, constraint never generated).
//! The registration-time guard walks the whole reachable tree.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
struct Group {
    items: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct W {
    groups: refs::Vec<Group>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(p.v - p.t) * 2.0]
}))]
struct P {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<P>,
}

fn main() {}

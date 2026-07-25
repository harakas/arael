//! An Option-held entity defined AFTER the root: the root's expansion
//! has already consumed the constraint stash, so `P`'s constraint would
//! be silently dropped (params still serialize -- the model would solve
//! quietly wrong). The registration-time ordering guard rejects it; the
//! syntax-side guard cannot, since `Option<P>` with an unregistered `P`
//! is indistinguishable from plain `Option<Data>`.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(m.v - m.t) * 0.5]
}))]
struct M {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<M>,
}

#[arael::model]
#[arael(root)]
struct W {
    items: refs::Vec<M>,
    maybe: Option<P>,
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

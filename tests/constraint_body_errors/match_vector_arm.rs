//! Every arm of a body `match` yields the same kind of value: a vector
//! arm after a scalar one is rejected with both kinds named, not
//! spliced or silently narrowed.

use arael::model::{Param, SelfBlock};
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = m.p - m.target;
    [match m.kind {
        0 => d.x,
        _ => d,
    }]
}))]
struct M {
    p: Param<vect3d>,
    target: vect3d,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

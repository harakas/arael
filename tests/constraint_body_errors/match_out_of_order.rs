//! Match arm patterns must be `0, 1, ...` in order: the arm order is the
//! generated match order, so a body must read with the same numbering.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let e = m.x - m.t;
    [match m.kind {
        1 => e,
        0 => 2.0 * e,
    }]
}))]
struct M {
    x: Param<f64>,
    t: f64,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

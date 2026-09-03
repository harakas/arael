//! Match arm guards have no meaning in a body `match` (the arms are
//! selected by the integer alone) and are rejected at the guard.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let e = m.x - m.t;
    [match m.kind {
        0 if true => e,
        _ => 2.0 * e,
    }]
}))]
struct M {
    x: Param<f64>,
    t: f64,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

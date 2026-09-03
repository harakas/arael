//! The `_` arm of a body `match` is the default and must come last.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let e = m.x - m.t;
    [match m.kind {
        0 => e,
        _ => 3.0 * e,
        1 => 2.0 * e,
    }]
}))]
struct M {
    x: Param<f64>,
    t: f64,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

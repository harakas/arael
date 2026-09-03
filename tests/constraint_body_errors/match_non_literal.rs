//! Match arm patterns are integer literals (or a final `_`); a binding
//! pattern is rejected instead of being read as a symbol.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let e = m.x - m.t;
    [match m.kind {
        0 => e,
        other => 2.0 * e,
    }]
}))]
struct M {
    x: Param<f64>,
    t: f64,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

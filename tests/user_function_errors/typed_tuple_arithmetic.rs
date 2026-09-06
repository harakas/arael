//! A tuple is destructured or returned as the residual rows; computing
//! with one should fail at the operator.

use arael::model::{Param, SelfBlock};
use arael::sym::E;

#[arael::function]
fn pair(x: E) -> (E, E) {
    let y = x * 2.0;
    (x, y)
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "p", {
    let t = pair(m.x);
    [t + 1.0]
}))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

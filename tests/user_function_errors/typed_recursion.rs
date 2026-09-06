//! A typed `#[arael::function]` is inlined at every call, so one that
//! calls itself should fail at the constraint that calls it.

use arael::model::{Param, SelfBlock};
use arael::sym::E;

#[arael::function]
fn again(x: E) -> E {
    let y = x * 2.0;
    again(y)
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "a", {
    [again(m.x)]
}))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

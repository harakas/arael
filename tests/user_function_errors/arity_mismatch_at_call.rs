//! Declaring a 1-arg user fn and calling it with 2 args from a
//! constraint body should fire an arity-mismatch error spanned on
//! the call site.

use arael::model::{Param, SelfBlock};
use arael_sym::E;

#[arael::function]
fn unary(x: E) -> E { x * x }

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "c", {
    [unary(m.x, m.y) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    y: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

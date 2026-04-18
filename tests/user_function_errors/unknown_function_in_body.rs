//! A user-fn body that references a name which is neither a built-in
//! nor another registered user-fn should fail when the fn is called
//! from a constraint body (that's when `parse_with_functions` runs).

use arael::model::{Param, SelfBlock};
use arael_sym::E;

#[arael::function]
fn bad_body(x: E) -> E {
    totally_unknown_fn(x)
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "c", {
    [bad_body(m.x) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

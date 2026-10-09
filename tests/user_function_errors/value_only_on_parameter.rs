//! A function declared without a derivative, applied to a parameter in
//! a residual: the Jacobian needs the derivative it does not have.

use arael::model::{Param, SelfBlock};

#[arael::function(foo)]
fn foo_eval(k: f64) -> f64 { k }

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [foo(m.x)] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

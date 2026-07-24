//! A malformed `derivs` expression used to be silently dropped from the
//! function bag: the function vanished, and a CALLER of it failed with
//! a misleading unknown-function error. Now the registration itself
//! errors, naming the function and the offending string.

use arael::model::{Param, SelfBlock};

#[arael::function(f, derivs = [3.0 ** x])]
fn f_eval(x: f64) -> f64 { 3.0 * x }

#[arael::function(g, derivs = [f(x)])]
fn g_eval(x: f64) -> f64 { 1.5 * x * x }

#[arael::model]
#[arael(constraint(hb, {
    [g(m.x) - 1.0]
}))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

#[arael::model]
#[arael(root)]
struct W {
    items: std::vec::Vec<M>,
}

fn main() {}

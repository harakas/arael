//! A root-taking function called without `root` as its first argument.

use arael::model::{Param, SelfBlock};

#[arael::function(f, derivs = [1.0])]
fn f_eval<R>(root: &R, x: f64) -> f64 { let _ = root; x }

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [f(m.x, m.x)] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

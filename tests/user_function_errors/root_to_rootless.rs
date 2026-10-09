//! `root` handed to a function that takes none.

use arael::model::{Param, SelfBlock};

#[arael::function(f, derivs = [1.0, 1.0])]
fn f_eval(x: f64, y: f64) -> f64 { x + y }

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [f(root, m.x)] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

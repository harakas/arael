//! A root-taking function whose root type is not the model's root.

use arael::model::{Param, SelfBlock};

struct Other { k: f64 }

#[arael::function(f, derivs = [root.k])]
fn f_eval(root: &Other, x: f64) -> f64 { root.k * x }

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [f(root, m.x)] }))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

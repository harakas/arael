//! A root-taking function in a `symbolic =` field: a component's
//! precompute has no root.

use arael::model::Param;

struct Scene;

#[arael::function(f, derivs = [1.0])]
fn f_eval(root: &Scene, x: f64) -> f64 { let _ = root; x }

#[arael::model]
#[arael(component)]
struct Off {
    d: Param<f64>,
    #[arael(symbolic = f(root, d))]
    c: f64,
}

fn main() {}

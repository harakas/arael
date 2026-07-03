//! A typoed keyword after `root,` must be rejected -- `jacobain` used to
//! be silently ignored, dropping the jacobian feature.

#[allow(unused_imports)]
use arael::model::Model;

#[arael::model]
#[arael(root, jacobain)]
struct R {
    w: f64,
}

fn main() {}

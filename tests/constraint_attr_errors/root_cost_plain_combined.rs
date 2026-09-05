//! `cost_plain` names the default cost sum and cannot be combined with
//! the other cost accumulation keywords.

#[allow(unused_imports)]
use arael::model::Model;

#[arael::model]
#[arael(root, cost_plain, cost_kahan)]
struct R {
    w: f64,
}

fn main() {}

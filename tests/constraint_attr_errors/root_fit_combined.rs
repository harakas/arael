//! `fit(...)` cannot be combined into the root attribute -- the docs
//! used to show this form and it compiled as a silent no-op (no fit
//! code generated). It must be a standalone #[arael(fit(...))].

#[allow(unused_imports)]
use arael::model::{Model, Param, SelfBlock};

#[arael::model]
#[arael(root, fit(y = k * x + c))]
struct R {
    k: Param<f64>,
    c: Param<f64>,
    x: f64,
    y: f64,
    hb: SelfBlock<R>,
}

fn main() {}

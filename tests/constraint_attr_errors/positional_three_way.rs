//! N >= 3 positional block lists must use the bracketed form.

use arael::model::{CrossBlock, Param, SelfBlock};

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb_a, hb_b, hb_c, {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb_a: SelfBlock<M>,
    hb_b: CrossBlock<M, M>,
    hb_c: CrossBlock<M, M>,
}

fn main() {}

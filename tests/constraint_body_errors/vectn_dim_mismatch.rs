//! Dimension-mismatched multiply: caught symbolically with both dims.

use arael::matrix::matrixd;
use arael::model::{Param, SelfBlock};
use arael::vect::vectd;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let p = m.h * m.v;
    [p[0] * m.isigma]
}))]
struct M {
    v: Param<vectd<5>>,
    h: matrixd<2, 4>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

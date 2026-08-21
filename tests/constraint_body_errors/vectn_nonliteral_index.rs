//! Runtime-varying component indices are impossible under compile-time
//! differentiation: the index must be a literal.

use arael::model::{Param, SelfBlock};
use arael::vect::vectd;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [m.v[m.k] * m.isigma]
}))]
struct M {
    v: Param<vectd<4>>,
    k: usize,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

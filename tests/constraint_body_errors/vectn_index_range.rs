//! Index out of range on a vect<T, N> read: the error names the dim.

use arael::model::{Param, SelfBlock};
use arael::vect::vectd;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [m.v[4] * m.isigma]
}))]
struct M {
    v: Param<vectd<4>>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

//! Const-generic dims must be literals: the macro reads them at
//! expansion time.

use arael::model::{Param, SelfBlock};
use arael::vect::vect;

#[arael::model]
struct G<const N: usize> {
    v: Param<vect<f64, N>>,
    hb: SelfBlock<G<N>>,
}

#[arael::model]
#[arael(root)]
struct M {
    gs: std::vec::Vec<G<4>>,
}

fn main() {}

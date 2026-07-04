//! A typoed field on a registered entity must be rejected with a
//! suggestion -- it used to become a free symbol spliced into generated
//! code, failing later in rustc at the root struct with no constraint
//! context (or silently compiling against an unintended field).

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [(m.gama - 1.0) * m.isigma]
}))]
struct M {
    gamma: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

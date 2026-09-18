//! The `(<local_self_block>, coo)` positional shape: every N >= 2 block
//! list must be bracketed (`[<local>, coo]`).

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, coo, {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

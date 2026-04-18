//! A 2-item positional list is only allowed when the second item is
//! `root.<triplet>`. Two bare local blocks must be bracketed.

use arael::model::{CrossBlock, Param, SelfBlock};

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb_pose, hb_other, {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb_pose: SelfBlock<M>,
    hb_other: CrossBlock<M, M>,
}

fn main() {}

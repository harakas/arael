//! A `match` in a body has scalar arms only: a vector arm is rejected
//! with the arm's type named, not spliced or silently narrowed.

use arael::model::{Param, SelfBlock};
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = m.p - m.target;
    [match m.kind {
        0 => d.x,
        _ => d,
    }]
}))]
struct M {
    p: Param<vect3d>,
    target: vect3d,
    kind: u32,
    hb: SelfBlock<M>,
}

fn main() {}

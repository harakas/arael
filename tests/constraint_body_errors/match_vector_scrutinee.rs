//! A body `match` switches on a scalar; a vector scrutinee is rejected
//! with its type named.

use arael::model::{Param, SelfBlock};
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = m.p - m.target;
    [match d {
        0 => d.x,
        _ => d.y,
    }]
}))]
struct M {
    p: Param<vect3d>,
    target: vect3d,
    hb: SelfBlock<M>,
}

fn main() {}

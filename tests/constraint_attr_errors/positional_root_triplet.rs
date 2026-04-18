//! The `(<local_self_block>, root.<triplet>)` positional shape used to
//! be accepted as a grandfathered form. It now must be bracketed
//! (`[<local>, root.<triplet>]`) like every other N >= 2 block list.

use arael::model::{Param, SelfBlock, TripletBlock};

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, root.hbt, {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
    hbt: TripletBlock<f64>,
}

fn main() {}

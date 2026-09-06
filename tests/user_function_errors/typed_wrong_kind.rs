//! A typed `#[arael::function]` called with an argument of another kind
//! than its parameter declares should fail at the call, naming the
//! parameter and both kinds.

use arael::model::{Param, SelfBlock};
use arael::sym::{vect3sym, E};

#[arael::function]
fn depth(p: vect3sym) -> E {
    let z = p.z;
    z
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "d", {
    [depth(m.x)]
}))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

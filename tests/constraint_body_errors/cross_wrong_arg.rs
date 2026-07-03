//! `.cross()` on a Vec3 requires a Vec3 argument.

use arael::model::{Param, SelfBlock};
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let c = m.v.cross(m.isigma);
    [c.x * m.isigma]
}))]
struct M {
    v: Param<vect3d>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}

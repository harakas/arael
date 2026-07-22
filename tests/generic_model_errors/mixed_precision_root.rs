// One root cannot hold two instantiations of the same generic entity:
// entities are resolved by bare type name.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::utils::Float;
use arael::vect::vect2;

#[arael::model]
#[arael(constraint(hb, {
    [pt.pos.x, pt.pos.y]
}))]
struct Pt<T: Float> {
    pos: Param<vect2<T>>,
    hb: SelfBlock<Pt<T>, T>,
}

#[arael::model]
#[arael(root)]
struct World {
    pts32: refs::Vec<Pt<f32>>,
    pts64: refs::Vec<Pt<f64>>,
}

fn main() {}

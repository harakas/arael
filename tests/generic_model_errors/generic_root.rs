// A #[arael(root)] struct must be concrete: the root is where generated
// solver code becomes real, at one precision.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::utils::Float;
use arael::vect::vect2;

#[arael::model]
struct Pt<T: Float> {
    pos: Param<vect2<T>>,
    hb: SelfBlock<Pt<T>, T>,
}

#[arael::model]
#[arael(root)]
struct World<T: Float> {
    pts: refs::Vec<Pt<T>>,
}

fn main() {}

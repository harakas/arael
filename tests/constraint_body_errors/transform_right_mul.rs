//! A transform acts from the left: `point * transform` is refused.

use arael::model::SelfBlock;
use arael::transform::TransformParam;
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let p = m.x * m.r2w;
    [p.x]
}))]
struct M {
    r2w: TransformParam<f64>,
    x: vect3d,
    hb: SelfBlock<M>,
}

fn main() {}

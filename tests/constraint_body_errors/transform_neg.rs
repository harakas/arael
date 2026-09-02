//! A transform has no negation; `.inv()` is its inverse.

use arael::model::SelfBlock;
use arael::transform::TransformParam;
use arael::vect::vect3d;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let t = -m.r2w;
    let p = t * m.x;
    [p.x]
}))]
struct M {
    r2w: TransformParam<f64>,
    x: vect3d,
    hb: SelfBlock<M>,
}

fn main() {}

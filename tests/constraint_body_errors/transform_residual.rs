//! A transform is not a residual.

use arael::model::SelfBlock;
use arael::transform::TransformParam;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [m.r2w]
}))]
struct M {
    r2w: TransformParam<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

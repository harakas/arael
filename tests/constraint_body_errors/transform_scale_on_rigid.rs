//! `.scale_factor` exists on a ScaledTransformParam only.

use arael::model::SelfBlock;
use arael::transform::TransformParam;

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [m.r2w.inv().scale_factor]
}))]
struct M {
    r2w: TransformParam<f64>,
    hb: SelfBlock<M>,
}

fn main() {}

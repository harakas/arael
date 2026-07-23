// `Hidden` is pub in the model crate but has a private field, so it was
// excluded from the bundle; using it as a model field names the reason.

use arael::model::{Param, SelfBlock};
use export_models::Hidden;

export_models::arael_import!();

#[arael::model]
struct Uses {
    h: Hidden,
    x: Param<f64>,
    hb: SelfBlock<Uses>,
}

fn main() {}

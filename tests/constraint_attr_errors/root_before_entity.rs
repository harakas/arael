//! A #[arael::model] struct defined AFTER the root is invisible to the
//! root's expansion (macro expansion is top-down file order): its
//! constraints were silently dropped. The root now rejects collection
//! element types with no registered layout.

use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(root)]
struct R {
    items: refs::Vec<Later>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(later.x - 1.0)]
}))]
struct Later {
    x: Param<f64>,
    hb: SelfBlock<Later>,
}

fn main() {}

//! An `Option<Entity>` location is single-instance: the same entity type
//! also living in a collection would leave one location's constraints
//! silently unevaluated -- rejected at expansion, like direct fields.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(constraint(hb, { [(p.x - 1.0) * 2.0] }))]
struct P {
    x: Param<f64>,
    hb: SelfBlock<P>,
}

#[arael::model]
#[arael(root)]
struct W {
    maybe: Option<P>,
    items: std::vec::Vec<P>,
}

fn main() {}

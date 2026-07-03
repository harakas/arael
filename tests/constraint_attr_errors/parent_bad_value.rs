//! `parent =` expects an entity field name; a non-ident value used to be
//! silently dropped, leaving the constraint parented to the default.

use arael::model::{Param, SelfBlock};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, parent = "lm", {
    [(m.x - 1.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn main() {}
